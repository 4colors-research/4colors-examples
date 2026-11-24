"""Offline tests of the post's model code: circuit builders, the scorer, the plateau reader, the selection rule and
the plumbing of run.py and plot.py. Nothing here compiles on the service or measures anything.

    python -m pytest -q tests
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import sys

_POST = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_POST), str(__import__("pathlib").Path(__file__).resolve().parents[3] / "shared")]


import postload  # noqa: E402  (blog/tests/postload.py; blog/tests is on sys.path from conftest)


def load(name):
    """Import <post dir>/<name>.py under a name unique to this post (blog/tests/postload.py), so posts with a
    run.py or plot.py of their own never collide when their tests run in one session."""
    return postload.load(_POST, name)

import qiskit_long as ql   # first: sets the one-thread environment before Qiskit loads

circuits = load("circuits")
score = load("score")
plateau = load("plateau")
run = load("run")
plot = load("plot")


# ---- circuits ------------------------------------------------------------------------------------------

def test_every_post_circuit_is_gate_level_and_deterministic():
    from qiskit import qasm3

    for cid in circuits.CIRCUITS:
        qc, f = circuits.build(cid)
        assert all(len(i.qubits) <= 2 for i in qc.data), cid
        assert {i.operation.name for i in qc.data} <= circuits.BASIC, cid
        assert f["input_2q"] > 0 and f["qubits"] == qc.num_qubits
    a, _ = circuits.build("zzkernel-n20")
    b, _ = circuits.build("zzkernel-n20")
    assert qasm3.dumps(a) == qasm3.dumps(b)


def test_sizes_match_the_design():
    want = {"qaoa_ba3_27_p3": 27, "qaoa_ba3_100_p3": 100, "tfim2d-n36": 36, "uccsd-n16": 16, "hea_full-n16": 16,
            "zzkernel-n20": 20, "portfolio_xy-n20": 20, "qft-n32": 32, "grover-n8": 8, "qae-n9": 9}
    for cid, n in want.items():
        assert circuits.build(cid)[1]["qubits"] == n
    assert circuits.build("tfim2d-n36")[1]["input_2q"] == 3 * 60       # 3 steps x 60 grid edges
    assert circuits.build("qft-n32")[1]["input_2q"] == 32 * 31 // 2
    assert circuits.build("zzkernel-n20")[1]["input_2q"] == 2 * 190
    assert not set(circuits.PILOT) & set(circuits.CIRCUITS)              # the pilot never uses the post's circuits


def test_export_round_trips(tmp_path):
    from qiskit import qasm3, qpy
    from qiskit.quantum_info import Operator

    rows = circuits.export(tmp_path, ["grover-n8"])
    assert rows[0]["sha256_qpy"] == circuits.sha256(tmp_path / "grover-n8.qpy")
    with open(tmp_path / "grover-n8.qpy", "rb") as fh:
        a = qpy.load(fh)[0]
    b = qasm3.loads((tmp_path / "grover-n8.qasm").read_text())
    assert Operator(a).equiv(Operator(b))
    assert json.loads((tmp_path / "circuits.json").read_text())[0]["circuit_id"] == "grover-n8"


def test_excitation_generators_are_hermitian_and_keep_the_particle_number():
    from qiskit.quantum_info import SparsePauliOp

    n = 4
    num = sum(((SparsePauliOp.from_sparse_list([("", [], 0.5)], n)
                - SparsePauliOp.from_sparse_list([("Z", [p], 0.5)], n)) for p in range(n)))
    for cre, ann in (([2], [0]), ([3, 2], [1, 0])):
        G = circuits.excitation_generator(n, cre, ann).to_matrix()
        assert np.allclose(G, G.conj().T) and np.abs(G).max() > 0
        N = num.to_matrix()
        assert np.allclose(G @ N, N @ G)


def test_uccsd_keeps_the_particle_number():
    from qiskit.quantum_info import Statevector

    qc = circuits.uccsd(3, 1, 1)                      # 6 spin orbitals, 2 electrons
    sv = Statevector(qc).probabilities_dict()
    assert all(k.count("1") == 2 for k, p in sv.items() if p > 1e-9)
    assert sum(p for k, p in sv.items() if k.count("1") == 2) == pytest.approx(1.0)


def test_grover_amplifies_the_marked_string():
    from qiskit.quantum_info import Statevector

    p = Statevector(circuits.grover(4, marked=0b1011, iterations=2)).probabilities()
    assert p[0b1011] > 0.8


def test_qae_peaks_at_the_amplitude():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    m, ns = 5, 1
    qc = circuits.qae(m, ns, seed=3)
    theta = circuits._angles(3, 2 * ns - 1, 0.2, 1.2)[0]
    a = math.sin(theta / 2) ** 2                       # probability that the state qubit is 1
    probs = Statevector(qc).probabilities(list(range(m)))
    y = int(np.argmax(probs))
    est = math.sin(math.pi * y / 2 ** m) ** 2
    assert abs(est - a) < 0.1


def test_portfolio_xy_keeps_the_budget():
    from qiskit.quantum_info import Statevector

    sv = Statevector(circuits.portfolio_xy(6, budget=3, p=1)).probabilities_dict()
    assert all(k.count("1") == 3 for k, p in sv.items() if p > 1e-9)


# ---- the scorer ------------------------------------------------------------------------------------------

def test_scorer_agrees_with_the_driver_on_fakefez():
    from qiskit import transpile

    target = ql.load_target("FakeFez")
    qc, _ = circuits.build("grover-n8")
    out = transpile(qc, target=target, optimization_level=1, seed_transpiler=1)
    a, b = score.score(out, target), ql.metrics(out, target)
    for k in ("two_qubit_gates", "two_qubit_depth", "depth"):
        assert a[k] == b[k]
    assert a["estimated_esp"] == pytest.approx(b["estimated_esp"], rel=1e-9)


def test_approximation_is_zero_for_an_exact_compile_and_sees_a_wrong_one():
    from qiskit import transpile

    target = ql.load_target("generic:6:line:3")
    qc = circuits.qaoa_3reg(4, p=1)
    out = transpile(qc, target=target, optimization_level=3, seed_transpiler=2, qubits_initially_zero=False)
    assert score.approximation(qc, out) == pytest.approx(0.0, abs=1e-8)
    bad = out.copy()
    bad.rz(0.3, bad.layout.final_index_layout()[0])
    assert score.approximation(qc, bad) > 1e-3


def test_approximation_is_not_measured_on_big_circuits():
    qc, _ = circuits.build("qft-n32")
    assert score.approximation(qc, qc) is None


# ---- plateau ---------------------------------------------------------------------------------------------

def test_plateau_reader():
    pts = [(0.4, 120), (0.9, 100), (1.5, 99), (2.0, 99), (40.0, 98)]
    assert plateau.best_so_far(pts) == [(0.4, 120), (0.9, 100), (1.5, 99), (40.0, 98)]
    assert plateau.last_improvement_s(pts) == 40.0
    assert plateau.time_to_within(pts, 0.03) == 0.9                     # 100 <= 98 x 1.03
    assert plateau.time_to_within(pts, 0.0105) == 1.5                  # 99 <= 98 x 1.0105
    assert plateau.value_at(pts, 0.1) is None and plateau.value_at(pts, 10) == 99
    assert plateau.late_gain(pts, 60) == pytest.approx(1 - 98 / 99)
    assert plateau.advice(pts, 60) == "longer may pay"                  # 99 -> 98 after 30 s: 1.01 %
    assert plateau.advice(pts, 10) == "plateau"
    assert plateau.advice([(8.0, 10)], 10) == "no answer"


# ---- the selection rule -----------------------------------------------------------------------------------

def _q(cid, b, seed, v, *, hind=None, status="finished", esp=0.1, wall=None):
    return {"kind": "result", "arm": "qiskit-long", "circuit_id": cid, "budget_s": b, "seed": seed, "select": "2q",
            "status": status, "two_qubit_gates": v if status == "finished" else None, "best_member_2q": hind,
            "estimated_esp": esp, "wall_s": wall or b, "over_budget": status != "finished", "approximation": 0.0}


def _o(cid, b, seed, v, **kw):
    return {"kind": "result", "arm": "4colors", "circuit_id": cid, "budget_s": b, "seed": seed,
            "two_qubit_gates": v, "estimated_esp": 0.2, "verification_label": "verified", **kw}


def test_selection_rule():
    rows = [_o("a", 10, s, v) for s, v in ((1, 90), (2, 91), (3, 92))]
    rows += [_q("a", 10, s, v) for s, v in ((1, 100), (2, 101), (3, 102))]
    c = plot.cell(rows, "a", 10, "2q")
    assert c["win"] and c["strict"]
    rows += [_o("b", 10, s, 99) for s in (1, 2, 3)] + [_q("b", 10, s, 100) for s in (1, 2, 3)]
    assert not plot.cell(rows, "b", 10, "2q")["win"]                      # 1 % better: inside the 2 % tie band
    rows += [_o("c", 10, s, 50) for s in (1, 2, 3)] + [_q("c", 10, s, 100, hind=49) for s in (1, 2, 3)]
    assert not plot.cell(rows, "c", 10, "2q")["win"]                      # the best single member counts
    rows += [_o("d", 1, s, 50) for s in (1, 2, 3)] + [_q("d", 1, 1, None, status="over_budget")]
    assert "over_budget" in plot.cell(rows, "d", 1, "2q")["why"]
    rows += [_o("e", 1, 1, 50, budget_floor_applied=True)] + [_q("e", 1, 1, 100)]
    assert "floor" in plot.cell(rows, "e", 1, "2q")["why"]
    rows += [_o("f", 10, 1, None, refusal_code="invalid_option")] + [_q("f", 10, 1, 100)]
    assert not plot.cell(rows, "f", 10, "2q")["win"]
    rows += [_o("g", 10, 1, 95), _o("g", 10, 2, 120), _o("g", 10, 3, 90)] + [_q("g", 10, s, 100) for s in (1, 2, 3)]
    g = plot.cell(rows, "g", 10, "2q")
    assert g["win"] and not g["strict"]                                   # median wins, the worst seed does not
    rows += [_o("h", 10, 1, 50, verification_label="unverified")] + [_q("h", 10, 1, 100)]
    assert "verified" in plot.cell(rows, "h", 10, "2q")["why"]
    rows += [_o("i", 10, 1, 50, verification_label="approximate", verification_distance=1e-3, d_cell=1e-4)]
    rows += [_q("i", 10, 1, 100)]
    assert not plot.cell(rows, "i", 10, "2q")["win"]                      # approximated more than Qiskit did
    shown = [c["circuit_id"] for c in plot.select_cells(rows, "2q") if c["win"]]
    assert shown == ["a", "g"]


def test_cell_settings():
    rows = [_q("a", 10, 1, 100, wall=9.2), {**_q("a", 10, 1, 100, wall=9.2), "select": "esp"}]
    cs = run.cell_settings(rows, "a", 1, 10)
    assert cs["t_cell"] == pytest.approx(9.2) and cs["options"] == {"approximation_mode": "exact"}
    rows = [{**r, "approximation": 3e-4} for r in rows]
    rows[1]["approximation"] = 1e-4
    assert run.cell_settings(rows, "a", 1, 10)["options"] == {"max_approximation_error": 1e-4}
    rows[0]["approximation"] = None                                       # too big to check: exact, flagged
    cs = run.cell_settings(rows, "a", 1, 10)
    assert cs["options"] == {"approximation_mode": "exact"} and not cs["d_measured"]
    short = [_q("b", 1, 1, 10, wall=0.4)]
    cs = run.cell_settings(short, "b", 1, 1)
    assert cs["t_cell"] == 1.0 and cs["budget_floor_applied"]
    assert "skip" in run.cell_settings(short, "zz", 1, 1)


def test_sensitivity_rows_never_set_the_cell():
    """E6 (qubits_initially_zero) and E7 (threads > 1) rows are reported apart: they never set T_cell or d_cell."""
    main = _q("a", 10, 1, 100, wall=9.2)
    e6 = {**_q("a", 10, 1, 90, wall=4.0), "qubits_initially_zero": True, "approximation": 0.3}
    e7 = {**_q("a", 10, 1, 95, wall=3.0), "threads": 4, "approximation": 0.3}
    assert run.is_sensitivity(e6) and run.is_sensitivity(e7) and not run.is_sensitivity(main)
    cs = run.cell_settings([main, e6, e7], "a", 1, 10)
    assert cs["t_cell"] == pytest.approx(9.2) and cs["options"] == {"approximation_mode": "exact"}
    assert "skip" in run.cell_settings([e6, e7], "a", 1, 10)


# ---- plumbing: run.py and plot.py on a fake Qiskit driver and canned compiles ---------------------------

def fake_run(args):
    """Stands in for qiskit_long.run in the plumbing test only: writes the input circuit as its 'best' answer.
    It does not transpile; the real driver is covered by test_qiskit_long.py."""
    from qiskit import qpy

    d = Path(args["out"])
    d.mkdir(parents=True, exist_ok=True)
    with open(args["qpy"], "rb") as fh:
        qc = qpy.load(fh)[0]
    m = ql.metrics(qc)
    trials = [{"worker": 0, "member": "o3", "k": 1, "seed": 1, "t_start": 0.0, "t_end": 0.5, "ok": True,
               "counted": True, **m},
              {"worker": 1, "member": "o3-sabre-k", "k": 2, "seed": 2, "t_start": 0.0, "t_end": 0.8, "ok": True,
               "counted": True, **m}]
    with open(d / "trials.jsonl", "w") as fh:
        fh.writelines(json.dumps({"id": args["id"], **t}) + "\n" for t in trials)
    for key in ql.SELECT_KEYS:
        with open(d / f"best-{key}-{args['id']}.qpy", "wb") as fh:
            qpy.dump(qc, fh)
    fin = {**trials[0], "layouts": None}
    pt = [0.5, "o3", 1, 1, m["two_qubit_gates"], m["two_qubit_depth"], m["depth"], m["estimated_esp"]]
    return {"final": {"2q": fin, "esp": fin}, "traj": {"2q": [pt], "esp": [pt]}, "wall_s": 0.95 * args["budget"],
            "trials": 2, "trials_counted": 2, "failures": 0, "members": ["o3", "o3-sabre-k"], "versions": {}}


def test_sensitivity_flags_reach_the_driver_and_the_rows(tmp_path):
    """--qubits-initially-zero 1 (E6) and --workers 1 --threads N (E7) go to qiskit_long.run and into every row."""
    seen = []

    def spy(args):
        seen.append(dict(args))
        return fake_run(args)

    e6 = tmp_path / "e6"
    run.main(["--out", str(e6), "--arm", "qiskit-long", "--circuits", "grover-n8", "--budgets", "1", "--seeds", "1",
              "--qubits-initially-zero", "1"], run=spy)
    assert seen[-1]["qubits_initially_zero"] is True and seen[-1]["threads"] == 1
    rows = [json.loads(x) for x in (e6 / "results.jsonl").read_text().splitlines()]
    res = [r for r in rows if r.get("kind") == "result"]
    assert res and all(r["qubits_initially_zero"] is True and r["threads"] == 1 for r in res)
    e7 = tmp_path / "e7"
    run.main(["--out", str(e7), "--arm", "qiskit-reseed", "--circuits", "grover-n8", "--budgets", "1",
              "--seeds", "1", "--workers", "1", "--threads", "4", "--cpus", "4,5,6,7"], run=spy)
    assert seen[-1]["threads"] == 4 and seen[-1]["workers"] == 1 and seen[-1]["cpus"] == [4, 5, 6, 7]
    assert seen[-1]["qubits_initially_zero"] is False
    rows = [json.loads(x) for x in (e7 / "results.jsonl").read_text().splitlines()]
    assert all(r["threads"] == 4 for r in rows if r.get("arm") == "qiskit-reseed")
    main_q = tmp_path / "q"                                   # defaults: the main runs are unchanged
    run.main(["--out", str(main_q), "--arm", "qiskit-long", "--circuits", "grover-n8", "--budgets", "1",
              "--seeds", "1"], run=spy)
    assert seen[-1]["threads"] == 1 and seen[-1]["qubits_initially_zero"] is False


@pytest.mark.parametrize("argv", [
    ["--arm", "qiskit-long", "--workers", "1", "--threads", "4", "--cpus", "4,5"],     # too few cpus for E7
    ["--arm", "qiskit-long", "--threads", "0"],
    ["--arm", "4colors", "--qubits-initially-zero", "1"],                           # Qiskit arms only
    ["--arm", "4colors", "--threads", "2"],
])
def test_sensitivity_flags_are_checked(argv):
    with pytest.raises(SystemExit):
        run.parse(["--out", "x", *argv])


def test_run_and_plot_plumbing(tmp_path, canned):
    q = tmp_path / "q"
    run.main(["--out", str(q), "--arm", "qiskit-long", "--circuits", "grover-n8", "--budgets", "1,10",
              "--seeds", "1"], run=fake_run)
    rows = [json.loads(x) for x in (q / "results.jsonl").read_text().splitlines()]
    res = [r for r in rows if r.get("kind") == "result"]
    assert {(r["select"], r["budget_s"]) for r in res} == {("2q", 1.0), ("esp", 1.0), ("2q", 10.0), ("esp", 10.0)}
    assert all(r["approximation"] == pytest.approx(0.0, abs=1e-9) for r in res)
    assert any(r.get("kind") == "time_share" for r in rows)
    r = tmp_path / "r"
    run.main(["--out", str(r), "--arm", "qiskit-reseed", "--circuits", "grover-n8", "--budgets", "1,10",
              "--seeds", "1"], run=fake_run)
    rr = [json.loads(x) for x in (r / "results.jsonl").read_text().splitlines()]
    assert any(x.get("arm") == "qiskit-default" for x in rr)
    assert {x["budget_s"] for x in rr if x.get("arm") == "qiskit-reseed"} == {1.0, 10.0}
    o = tmp_path / "o"
    run.main(["--out", str(o), "--arm", "4colors", "--qiskit", str(q), "--circuits", "grover-n8",
              "--budgets", "1,10", "--seeds", "1", "--stream"])
    budgets = [c["time_budget_s"] for c in canned.calls]
    assert budgets == [pytest.approx(1.0), pytest.approx(9.5), pytest.approx(9.5)]   # floor, T_cell, the stream
    ours = [json.loads(x) for x in (o / "results.jsonl").read_text().splitlines()]
    res = [x for x in ours if x.get("kind") == "result"]
    assert len(res) == 2 and all(x["options"] == {"approximation_mode": "exact"} for x in res)
    assert [x["budget_floor_applied"] for x in res] == [True, False]
    assert all(c.get("approximation_mode") == "exact" and c.get("verify") == "wait" for c in canned.calls[:2])
    assert any(x.get("kind") == "incumbent" for x in ours)
    figs = tmp_path / "figs"
    plot.main([str(o), str(q), str(r), "--figs", str(figs)])
    assert (figs / "selection.md").exists() and (figs / "table.md").exists()
