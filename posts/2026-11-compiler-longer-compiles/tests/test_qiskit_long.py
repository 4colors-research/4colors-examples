"""Offline checks that the Qiskit-long code runs and keeps the operator. Tiny circuits only; never a measurement.

    python -m pytest -q tests

Timings printed by the test runner are not results and are never quoted.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import qiskit_long as ql  # noqa: E402  (sets the one-thread environment before Qiskit loads)

import numpy as np  # noqa: E402
from qiskit import QuantumCircuit, qpy  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402


def fidelity(a: Operator, b: Operator) -> float:
    """|Tr(A^dagger B)| / d: 1 for equal operators up to a global phase."""
    d = a.dim[0]
    return abs(np.trace(a.data.conj().T @ b.data)) / d


def qaoa_ring(n=5, p=2):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for a in range(n):
            for b in range(a + 1, n):          # complete graph: forces routing on a line
                qc.rzz(0.3 + 0.1 * layer + 0.01 * (a + b), a, b)
        qc.rx(0.7 - 0.1 * layer, range(n))
    return qc


def mixed(n=5):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(0.2 * (i + 1), i)
    qc.cx(0, 4)
    qc.cx(1, 3)
    qc.ccx(0, 2, 4)
    qc.cp(0.4, 3, 0)
    qc.swap(1, 4)
    qc.cx(2, 0)
    return qc


class MemberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = ql.Ctx(ql.load_target("generic:5:line:7"), approximation_degree=1.0, qubits_initially_zero=False)

    def check(self, member, qc, k=2, seed=11):
        prep = ql.prepare(qc)
        if not member.applies(prep):
            return None
        out = member.build(self.ctx, k, seed, prep).run(qc)
        f = fidelity(Operator(qc), Operator.from_circuit(out))
        self.assertGreater(f, 1 - 1e-6, f"{member.name}: operator changed (fidelity {f})")
        names = set(out.count_ops()) - {"barrier", "measure"}
        self.assertTrue(names <= {"cz", "rz", "sx", "x"}, f"{member.name}: gates outside the target {names}")
        for inst in out.data:                   # every two-qubit gate on a coupler of the line
            if ql.is_two_qubit(inst):
                a, b = sorted(out.find_bit(q).index for q in inst.qubits)
                self.assertEqual(b - a, 1, f"{member.name}: gate on ({a}, {b})")
        return out

    def test_every_member_keeps_the_operator(self):
        ran = []
        for m in ql.available_members("all"):
            if m.needs:                         # the AI member lives in its own env (see the post)
                continue
            for qc in (qaoa_ring(), mixed()):
                with self.subTest(member=m.name, circuit=qc.num_qubits):
                    if self.check(m, qc) is not None:
                        ran.append(m.name)
        self.assertIn("o3-swapstrat-line", ran)  # the QAOA input has commuting ZZ blocks
        self.assertIn("o3-kitchen-sink-k", ran)

    def test_effort_changes_the_passes(self):
        from qiskit.transpiler.passes import MinimumPoint, SabreLayout

        found = {}

        def grab(task):
            subs = getattr(task, "tasks", None)
            if subs is None:
                found.setdefault(type(task).__name__, []).append(task)
                return
            for t in subs:
                grab(t)
        m = next(x for x in ql.MEMBERS if x.name == "o3-kitchen-sink-k")
        pm = m.build(self.ctx, 8, 3, ql.prepare(mixed()))
        for stage in (pm.layout, pm.optimization):
            grab(stage.to_flow_controller())
        sl = found["SabreLayout"][0]
        self.assertIsInstance(sl, SabreLayout)
        self.assertEqual((sl.layout_trials, sl.swap_trials, sl.max_iterations), (160, 160, 7))
        self.assertTrue(all(isinstance(x, MinimumPoint) and x.backtrack_depth == 40 for x in found["MinimumPoint"]))
        for name in ("CommutativeOptimization", "CommutativeInverseCancellation", "Optimize1qGatesSimpleCommutation"):
            self.assertIn(name, found)
        found.clear()
        la = next(x for x in ql.MEMBERS if x.name == "o3-sabre-lookahead-k").build(self.ctx, 2, 3, None)
        for stage in (la.layout, la.routing):
            grab(stage.to_flow_controller())
        self.assertTrue(found["SabreLayout"][0].skip_routing)
        self.assertEqual({s.heuristic for s in found["SabreSwap"]}, {"lookahead"})
        found.clear()
        de = next(x for x in ql.MEMBERS if x.name == "o3-dense-k").build(self.ctx, 1, 3, None)
        for stage in (de.layout, de.routing):
            grab(stage.to_flow_controller())
        self.assertIn("DenseLayout", found)
        self.assertNotIn("SabreLayout", found)
        self.assertIn("VF2PostLayout", found)          # kept: the default stages, not layout_method="dense"

    def test_lift_keeps_the_operator(self):
        qc = qaoa_ring(4, 1)
        lifted, blocks = ql.lift_rzz_blocks(qc)
        self.assertEqual(blocks, 1)
        self.assertGreater(fidelity(Operator(qc), Operator(lifted)), 1 - 1e-9)
        self.assertEqual(ql.lift_rzz_blocks(mixed())[1], 0)


class Fake:
    def __init__(self, name, scales, prior=1.0):
        self.name, self.scales, self.cost_prior = name, scales, prior


class SchedulerTests(unittest.TestCase):
    def test_rotation_escalation_and_fit(self):
        a, b = Fake("a", True), Fake("b", False)
        s = ql.Scheduler([a, b], rotate=1)
        self.assertEqual(s.next(100.0), (b, 1))        # rotated start; nothing measured: the first trial runs
        s.done(b, 1, 2.0, True)
        self.assertEqual(s.next(100.0), (a, 1))        # predicted 2 s from the reference, fits
        s.done(a, 1, 2.0, True)
        self.assertEqual(s.next(100.0), (b, 1))        # b does not scale
        s.done(b, 1, 2.0, True)
        self.assertEqual(s.next(100.0), (a, 2))        # doubled (4 s) fits in a third of 100 s
        s.done(a, 2, 4.0, True)
        self.assertEqual(s.next(10.0)[0], b)
        s.done(b, 1, 2.0, True)
        self.assertEqual(s.next(10.0), (a, 2))         # 8 s > 10/3: no escalation, 4 s fits

    def test_nothing_fits_and_dead_members(self):
        a, b = Fake("a", True), Fake("b", True)
        s = ql.Scheduler([a, b])
        s.next(50.0)
        s.done(a, 1, 30.0, True)
        s.next(50.0)
        s.done(b, 1, 1.0, False)                       # failed: never retried
        self.assertEqual(s.next(31.0), (a, 1))
        self.assertIsNone(s.next(29.0))                # a needs 30 s, b is dead: stop, do not start what cannot end


class RunSmokeTest(unittest.TestCase):
    def test_tiny_run(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "in.qpy")
            with open(src, "wb") as fh:
                qpy.dump(qaoa_ring(4, 1), fh)
            rec = ql.run({"qpy": str(src), "id": "t", "backend": "generic:4:line:3", "budget": 0.6, "workers": 1,
                          "cpus": None, "seed": 1, "members": "o3,o3-sabre-k", "approximation_degree": 1.0,
                          "qubits_initially_zero": False, "out": d})
            self.assertIsNotNone(rec["final"]["2q"])
            rows = [json.loads(x) for x in Path(d, "trials.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["t_end"] <= 0.6 for r in rows if r["counted"]))
            with open(Path(d, "best-2q-t.qpy"), "rb") as fh:
                best = qpy.load(fh)[0]
            self.assertEqual(ql.metrics(best)["two_qubit_gates"], rec["final"]["2q"]["two_qubit_gates"])
            self.assertEqual(rec["threads"], 1)

    def test_tiny_threaded_run(self):
        """E7's shape: one worker with two Rayon threads runs and records its thread setting."""
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "in.qpy")
            with open(src, "wb") as fh:
                qpy.dump(qaoa_ring(4, 1), fh)
            rec = ql.run({"qpy": str(src), "id": "t", "backend": "generic:4:line:3", "budget": 0.6, "workers": 1,
                          "threads": 2, "cpus": None, "seed": 1, "members": "o3", "approximation_degree": 1.0,
                          "qubits_initially_zero": False, "out": d})
            self.assertIsNotNone(rec["final"]["2q"])
            self.assertEqual(rec["threads"], 2)
            self.assertEqual(rec["thread_env"]["RAYON_NUM_THREADS"], "2")


class ThreadEnvTests(unittest.TestCase):
    def test_main_runs_are_single_threaded(self):
        env = ql.worker_env()
        self.assertEqual(env, ql.THREAD_ENV)
        self.assertTrue(all(v in ("1", "FALSE") for v in env.values()))

    def test_e7_raises_rayon_only(self):
        env = ql.worker_env(4)
        self.assertEqual(env["RAYON_NUM_THREADS"], "4")
        self.assertEqual({k: v for k, v in env.items() if k != "RAYON_NUM_THREADS"},
                         {k: v for k, v in ql.THREAD_ENV.items() if k != "RAYON_NUM_THREADS"})
        with self.assertRaises(ValueError):
            ql.worker_env(0)

    def test_cpus_per_worker(self):
        self.assertIsNone(ql.worker_cpus(None, 0))
        self.assertEqual(ql.worker_cpus([4, 5, 6, 7], 2), {6})
        self.assertEqual(ql.worker_cpus([4, 5, 6, 7], 0, threads=4), {4, 5, 6, 7})
        self.assertEqual(ql.worker_cpus([4, 5, 6, 7], 1, threads=2), {6, 7})
        with self.assertRaises(ValueError):
            ql.worker_cpus([4, 5], 0, threads=4)

    def test_cli_checks_cpus_per_thread(self):
        with self.assertRaises(SystemExit):
            ql.main(["--qpy", "x.qpy", "--id", "t", "--budget", "1", "--workers", "1", "--threads", "4",
                     "--cpus", "4,5", "--out", "o"])


if __name__ == "__main__":
    unittest.main()
