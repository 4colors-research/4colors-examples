"""Offline tests on synthetic tiny circuits (no compiler, no service, no records).

    python3 -m unittest discover -s tests -v                 # stdlib only
    <env with qiskit>/bin/python -m unittest discover -s tests   # also runs the Qiskit cross-checks
"""
import math
import os
import random
import sys
import unittest
import xml.etree.ElementTree as ET
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sbs  # noqa: E402

try:
    import qiskit  # noqa: F401
    HAVE_QISKIT = True
except Exception:  # pragma: no cover
    HAVE_QISKIT = False

HDR = 'OPENQASM 3.0;\ninclude "stdgates.inc";\n'


def phys(body: str, nbits: int = 0) -> sbs.Circuit:
    return sbs.parse_qasm(HDR + (f"bit[{nbits}] meas;\n" if nbits else "") + body)


SWAP_CZ = """rz(pi/2) $1; sx $1; rz(pi/2) $1;
cz $0, $1;
rz(pi/2) $0; sx $0; rz(pi/2) $0; rz(pi/2) $1; sx $1; rz(pi/2) $1;
cz $0, $1;
rz(pi/2) $0; sx $0; rz(pi/2) $0; rz(pi/2) $1; sx $1; rz(pi/2) $1;
cz $0, $1;
rz(pi/2) $1; sx $1; rz(pi/2) $1;
"""  # H-conjugated CZs = CX(0,1) CX(1,0) CX(0,1) = SWAP


class Parse(unittest.TestCase):
    def test_three_emitter_styles(self):
        a = phys("cz $3, $5;\nsx $3;\nmeas[0] = measure $5;\n", 1)
        b = sbs.parse_qasm(HDR + "bit[1] c;\nqubit[8] q;\ncz q[3], q[5];\nsx q[3];\nc[0] = measure q[5];\n")
        c = sbs.parse_qasm('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[8];\ncreg c[1];\n'
                           "cz q[3],q[5];\nsx q[3];\nmeasure q[5] -> c[0];\n")
        for x in (a, b, c):
            self.assertEqual([(o.name, o.qubits) for o in x.ops], [("cz", (3, 5)), ("sx", (3,)), ("measure", (5,))])
            self.assertEqual(x.measured(), {0: 5})

    def test_annotation_and_gphase(self):
        c = sbs.parse_qasm("OPENQASM 3.0;\n@fourcolors.initial_layout 4 2 7\ninclude \"stdgates.inc\";\n"
                           "gphase(0.5);\nrz(3*pi/2) $4;\n")
        self.assertEqual(sbs.initial_layout(c, None, 3), [4, 2, 7])
        self.assertAlmostEqual(c.global_phase, 0.5)
        self.assertAlmostEqual(c.ops[0].params[0], 1.5 * math.pi)

    def test_angles(self):
        for s, v in [("pi/2", math.pi / 2), ("-pi", -math.pi), ("3*pi/4", 0.75 * math.pi), ("1.5e-3", 1.5e-3),
                     ("-(pi/2)", -math.pi / 2), ("π/4", math.pi / 4)]:
            self.assertAlmostEqual(sbs._num(s), v)
        with self.assertRaises(ValueError):
            sbs._num("__import__('os')")

    def test_dynamic_refused(self):
        with self.assertRaises(ValueError):
            phys("if (meas[0]) x $0;\n", 1)


class Metrics(unittest.TestCase):
    def test_counts_and_depths(self):
        c = phys("sx $0;\ncz $0, $1;\nrz(0.1) $2;\ncz $1, $2;\nx $0;\nbarrier $0, $1, $2;\ncz $0, $1;\n"
                 "meas[0] = measure $0;\n", 1)
        m = sbs.metrics(c)
        self.assertEqual(m["two_qubit_gates"], 3)
        self.assertEqual(m["two_qubit_depth"], 3)       # barrier aligns: layer 1, 2, then 3
        self.assertEqual(m["one_qubit_pulses"], 2)      # sx + x; rz is virtual
        self.assertEqual(m["rz"], 1)
        # depth: sx(1) cz(2) | rz(1) cz12(3) | x(3) | barrier aligns to 3 | cz(4) ; measure excluded
        self.assertEqual(m["depth"], 4)
        self.assertEqual(m["depth_with_measure"], 5)
        self.assertEqual(sbs.coupler_counts(c), {(0, 1): 2, (1, 2): 1})
        self.assertEqual(sbs.layers(c), [(1, 0, 1), (2, 1, 2), (3, 0, 1)])


class Blocks(unittest.TestCase):
    def test_swap_is_swap_class(self):
        bl = sbs.blocks(phys(SWAP_CZ))
        self.assertEqual(len(bl), 1)
        self.assertEqual((bl[0].n2, bl[0].min_2q, bl[0].swap_class, bl[0].slack), (3, 3, True, 0))

    def test_cz_pair_cancels(self):
        bl = sbs.blocks(phys("cz $0, $1;\nrz(0.3) $0;\ncz $0, $1;\n"))
        self.assertEqual((bl[0].n2, bl[0].min_2q, bl[0].swap_class, bl[0].slack), (2, 0, False, 2))

    def test_cx_class(self):
        bl = sbs.blocks(phys("cz $0, $1;\nsx $0;\n"))
        self.assertEqual((bl[0].min_2q, bl[0].swap_class), (1, False))

    def test_swap_times_cz_needs_two(self):
        bl = sbs.blocks(phys(SWAP_CZ + "cz $0, $1;\n"))
        self.assertEqual((bl[0].n2, bl[0].min_2q, bl[0].swap_class, bl[0].slack), (4, 2, False, 2))

    def test_other_pair_splits_blocks(self):
        bl = sbs.blocks(phys("cz $0, $1;\ncz $1, $2;\ncz $0, $1;\n"))
        self.assertEqual([b.pair for b in bl], [(0, 1), (1, 2), (0, 1)])
        self.assertEqual([b.layer for b in bl], [1, 2, 3])

    def test_cx_direction_and_rzz(self):
        a = sbs.blocks(phys("cx $1, $0;\n"))[0]
        self.assertEqual(a.min_2q, 1)
        b = sbs.blocks(phys("rzz(0.4) $0, $1;\n"))[0]
        self.assertEqual(b.min_2q, 2)
        sw = sbs.blocks(phys("cx $0, $1;\ncx $1, $0;\ncx $0, $1;\n"))[0]
        self.assertTrue(sw.swap_class)

    def test_summary(self):
        s = sbs.block_summary(sbs.blocks(phys(SWAP_CZ + "cz $1, $2;\ncz $1, $2;\n")))
        self.assertEqual((s["blocks"], s["swap_class_blocks"], s["slack_2q"], s["blocks_with_slack"]), (2, 1, 2, 1))


    # Block.kind on tiny synthetic circuits, one per class (the timeline colors use it).
    CX01 = "rz(pi/2) $1; sx $1; rz(pi/2) $1;\ncz $0, $1;\nrz(pi/2) $1; sx $1; rz(pi/2) $1;\n"

    def kinds(self, body):
        return [(b.n2, b.min_2q, b.kind) for b in sbs.blocks(phys(body))]

    def test_kind_standalone_swap(self):
        self.assertEqual(self.kinds(SWAP_CZ), [(3, 3, "swap")])
        self.assertEqual(self.kinds("swap $0, $1;\n"), [(1, 3, "swap")])

    def test_kind_fused_cx_plus_swap(self):
        # CX then SWAP compiled to 2 CZ: CZ(0,1) . H(0) . CZ(0,1) . H on both = SWAP x CX up to 1q gates
        body = ("rz(pi/2) $1; sx $1; rz(pi/2) $1;\ncz $0, $1;\n"
                "rz(pi/2) $0; sx $0; rz(pi/2) $0; rz(pi/2) $1; sx $1; rz(pi/2) $1;\ncz $0, $1;\n"
                "rz(pi/2) $0; sx $0; rz(pi/2) $0;\n")
        b = sbs.blocks(phys(body))[0]
        self.assertEqual((b.n2, b.min_2q, b.swap_class, b.min_2q_without_swap, b.kind), (2, 2, False, 1, "fused"))
        # the same block written as the two logical gates it replaces
        self.assertEqual(self.kinds(self.CX01 + SWAP_CZ), [(4, 2, "fused")])

    def test_kind_fused_cphase_plus_swap(self):
        # CP(t) + SWAP needs 3 CZ but is not SWAP-class: a move fused into a controlled phase (QFT on a line)
        b = sbs.blocks(phys("cp(0.3) $0, $1;\nswap $0, $1;\n"))[0]
        self.assertEqual((b.min_2q, b.swap_class, b.min_2q_without_swap, b.kind), (3, False, 2, "fused"))

    def test_kind_gates_without_move(self):
        self.assertEqual(self.kinds("cz $0, $1;\n"), [(1, 1, "gate")])
        self.assertEqual(self.kinds(self.CX01), [(1, 1, "gate")])
        self.assertEqual(self.kinds("rzz(0.4) $0, $1;\n"), [(1, 2, "gate")])     # 2 CZ, but no move in it
        self.assertEqual(self.kinds("cp(0.3) $0, $1;\n"), [(1, 2, "gate")])
        self.assertEqual(self.kinds("cz $0, $1;\nrz(0.3) $0;\ncz $0, $1;\n"), [(2, 0, "gate")])

    def test_kind_summary_and_timeline_colors(self):
        body = SWAP_CZ + "cz $1, $2;\n" + "cp(0.3) $2, $3;\nswap $2, $3;\n"
        c = phys(body)
        bl = sbs.blocks(c)
        self.assertEqual([b.kind for b in bl], ["swap", "gate", "fused"])
        s = sbs.block_summary(bl)
        self.assertEqual((s["swap_class_blocks"], s["fused_move_blocks"], s["two_qubit_gates_in_fused_blocks"]),
                         (1, 1, 2))
        svg = sbs.timeline_svg(c, bl, title="t")
        strokes = [e.get("stroke") for e in ET.fromstring(svg).iter("{http://www.w3.org/2000/svg}line")
                   if e.find("{http://www.w3.org/2000/svg}title") is not None]
        self.assertEqual(Counter(strokes), Counter({sbs.SWAPC: 3, sbs.WORK: 1, sbs.FUSED: 2}))


class Layouts(unittest.TestCase):
    def setUp(self):
        self.inp = sbs.parse_qasm(HDR + "bit[2] meas;\nqubit[2] q;\ncx q[0], q[1];\n"
                                  "meas[0] = measure q[0];\nmeas[1] = measure q[1];\n")

    def test_tracking_consistent(self):
        # logical 0 at phys 0, logical 1 at phys 2; SWAP (0,1) moves logical 0 next to it.
        out = sbs.parse_qasm("OPENQASM 3.0;\n@fourcolors.initial_layout 0 2\n" + 'include "stdgates.inc";\n'
                             "bit[2] meas;\n" + SWAP_CZ + "cz $1, $2;\nmeas[0] = measure $1;\nmeas[1] = measure $2;\n")
        rep = sbs.inspect(self.inp, out)
        self.assertEqual(rep["final_layout"], [1, 2])
        self.assertTrue(rep["swap_tracking"]["consistent"])
        self.assertEqual(rep["swap_tracking"]["moved_logical"], 1)
        self.assertEqual(rep["_paths"][0], [(0, 0), (1, 1)])

    def test_fused_move_is_tracked(self):
        # SWAP fused with the logical CZ on the same pair: not SWAP-class, but a 'fused' block, so tracking
        # follows the move without any greedy resolution.
        out = sbs.parse_qasm("OPENQASM 3.0;\n@fourcolors.initial_layout 0 1\n" + 'include "stdgates.inc";\n'
                             "bit[2] meas;\n" + SWAP_CZ + "cz $0, $1;\nmeas[0] = measure $1;\nmeas[1] = measure $0;\n")
        rep = sbs.inspect(self.inp, out)
        self.assertTrue(rep["swap_tracking"]["consistent"])
        self.assertEqual(rep["swap_tracking"]["hidden_moves"], 0)
        self.assertEqual(rep["blocks"]["histogram"][0]["min_2q"], 2)
        self.assertEqual((rep["blocks"]["fused_move_blocks"], rep["blocks"]["swap_class_blocks"]), (1, 0))

    def test_unexplained_move_is_resolved(self):
        # The final layout says the qubits swapped, but the only block on (0, 1) is an RZZ (2 CZ, no move in
        # it): structural tracking disagrees and the greedy resolution has to assume a hidden move.
        out = sbs.parse_qasm("OPENQASM 3.0;\n@fourcolors.initial_layout 0 1\n" + 'include "stdgates.inc";\n'
                             "bit[2] meas;\nrzz(0.4) $0, $1;\nmeas[0] = measure $1;\nmeas[1] = measure $0;\n")
        st = sbs.inspect(self.inp, out)["swap_tracking"]
        self.assertFalse(st["consistent"])
        self.assertEqual(st["hidden_moves"], 1)
        self.assertTrue(st["consistent_after_resolve"])

    def test_input_swaps_are_not_routing(self):
        # The input itself swaps its two qubits; a compiler that just relabels needs no move at all.
        inp = sbs.parse_qasm(HDR + "bit[2] meas;\nqubit[2] q;\ncx q[0], q[1];\nswap q[0], q[1];\n"
                             "meas[0] = measure q[0];\nmeas[1] = measure q[1];\n")
        out = sbs.parse_qasm("OPENQASM 3.0;\n@fourcolors.initial_layout 3 4\n" + 'include "stdgates.inc";\n'
                             "bit[2] meas;\ncz $3, $4;\nmeas[0] = measure $4;\nmeas[1] = measure $3;\n")
        self.assertEqual(sbs.final_layout(inp, out), {0: 3, 1: 4})
        self.assertTrue(sbs.inspect(inp, out)["swap_tracking"]["consistent"])

    def test_rival_layout_json(self):
        out = phys("cz $5, $6;\n")
        self.assertEqual(sbs.initial_layout(out, {"initial": [5, 6, 9, 10]}, 2), [5, 6])


class Esp(unittest.TestCase):
    def test_product(self):
        cal = {"qubits": [{"qubit": 0, "readout": {"error": 0.1}, "gates": {"sx": {"error": 0.01}, "rz": {"error": 0}}},
                          {"qubit": 1, "readout": {"error": 0.2}, "gates": {}}],
               "couplers": [{"qubits": [1, 0], "error": 0.05}]}
        c = phys("sx $0;\nrz(1) $0;\ncz $0, $1;\nmeas[0] = measure $0;\nmeas[1] = measure $1;\n", 2)
        self.assertAlmostEqual(sbs.esp(c, cal), 0.99 * 0.95 * 0.9 * 0.8)
        cal["couplers"][0]["error"] = 1.0
        self.assertEqual(sbs.esp(c, cal), 0.0)


class Svg(unittest.TestCase):
    def test_well_formed(self):
        # a 2x6 toy "rows + bridges" device: rows 0-5 and 7-12, bridge 6 between 2 and 9
        cps = [(i, i + 1) for i in range(5)] + [(i, i + 1) for i in range(7, 12)] + [(2, 6), (6, 9)]
        c = phys(SWAP_CZ + "cz $1, $2;\ncz $2, $6;\n")
        bl = sbs.blocks(c)
        for s in (sbs.device_svg(13, cps, sbs.coupler_counts(c), title="toy <a&b>", marked=[0, 1], ring=[2],
                                 path=[0, 1, 2]),
                  sbs.timeline_svg(c, bl, title="toy")):
            ET.fromstring(s)
        pos = sbs.lattice_positions(13, cps)
        self.assertEqual(pos[6], (2.0, 1.0))

    def test_heavy_hex_from_coupling_map(self):
        # two rows of 8 (ids 0-7, 10-17) joined by bridges 8 (3-8-12) and 9 (7-9-16); the row edge (14, 15)
        # is broken and qubit 13 has no couplers at all (disabled)
        cps = [(i, i + 1) for i in range(7)] + [(10, 11), (11, 12), (15, 16), (16, 17)]
        cps += [(3, 8), (8, 12), (7, 9), (9, 16)]
        pos = sbs.heavy_hex_positions(18, cps)
        self.assertEqual(pos[0], (0.0, 0.0))
        self.assertEqual(pos[8], (3.0, 1.0))
        self.assertEqual(pos[12], (3.0, 2.0))
        self.assertEqual(pos[10], (1.0, 2.0))
        self.assertEqual(pos[15], (6.0, 2.0))       # reached through bridge 9, not through the broken edge
        self.assertEqual(pos[13], (4.0, 2.0))       # disabled: put back between 12 and 14
        self.assertEqual(len(set(pos.values())), 18)
        for a, b in cps:
            self.assertEqual(abs(pos[a][0] - pos[b][0]) + abs(pos[a][1] - pos[b][1]), 1.0)
        # not a heavy-hex numbering (a triangle): no lattice, the fallback is used
        self.assertIsNone(sbs.heavy_hex_positions(3, [(0, 1), (1, 2), (0, 2)]))
        self.assertEqual(len(sbs.device_positions(3, [(0, 1), (1, 2), (0, 2)])), 3)

    def test_timelines_share_scale(self):
        a = phys("cz $0, $1;\n")
        b = phys("cz $0, $1;\ncz $1, $2;\ncz $2, $3;\n")
        sa = ET.fromstring(sbs.timeline_svg(a, sbs.blocks(a), n_layers=3, n_rows=4))
        sb = ET.fromstring(sbs.timeline_svg(b, sbs.blocks(b), n_layers=3, n_rows=4))
        self.assertEqual((sa.get("width"), sa.get("height")), (sb.get("width"), sb.get("height")))


@unittest.skipUnless(HAVE_QISKIT, "qiskit not importable")
class QiskitCrossCheck(unittest.TestCase):
    """Same numbers as Qiskit on random tiny circuits: depth(filter) semantics and minimal CNOT counts."""

    def test_depths_match_qiskit(self):
        from qiskit import QuantumCircuit, qasm3
        rnd = random.Random(7)
        for trial in range(30):
            qc = QuantumCircuit(4, 2)
            for _ in range(25):
                r = rnd.random()
                a, b = rnd.sample(range(4), 2)
                if r < 0.35:
                    qc.cz(a, b)
                elif r < 0.55:
                    qc.sx(a)
                elif r < 0.75:
                    qc.rz(rnd.uniform(-3, 3), a)
                elif r < 0.85:
                    qc.x(a)
                elif r < 0.92:
                    qc.barrier([a, b])
                else:
                    qc.measure(a, rnd.randrange(2))
            mine = sbs.metrics(sbs.parse_qasm(qasm3.dumps(qc)))
            two = lambda inst: inst.operation.num_qubits == 2 and inst.operation.name not in sbs.NON_GATES
            self.assertEqual(mine["two_qubit_gates"], sum(1 for i in qc.data if two(i)), trial)
            self.assertEqual(mine["two_qubit_depth"], qc.depth(two), trial)
            self.assertEqual(mine["depth"], qc.depth(lambda i: i.operation.name not in sbs.NON_GATES), trial)

    def test_min_cnot_matches_qiskit_decomposer(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import CXGate
        from qiskit.quantum_info import Operator
        from qiskit.synthesis import TwoQubitBasisDecomposer
        dec = TwoQubitBasisDecomposer(CXGate())
        rnd = random.Random(11)
        seen = set()
        for trial in range(60):
            k = rnd.choice([0, 1, 2, 3, 4])
            qc = QuantumCircuit(2)
            lines = []
            for _ in range(k):
                for q in (0, 1):
                    if rnd.random() < 0.7:
                        t = rnd.choice([0.0, math.pi / 2, math.pi, rnd.uniform(-3, 3)])
                        qc.rz(t, q)
                        qc.sx(q)
                        lines += [f"rz({t!r}) ${q};", f"sx ${q};"]
                qc.cz(0, 1)
                lines.append("cz $0, $1;")
            if not lines or rnd.random() < 0.1:
                qc.cz(0, 1)
                qc.rz(0.7, 1)
                qc.cz(0, 1)
                lines += ["cz $0, $1;", "rz(0.7) $1;", "cz $0, $1;"]
            qc.rz(0.2, 0)
            lines.append("rz(0.2) $0;")
            bl = sbs.blocks(phys("\n".join(lines) + "\n"))
            ref = dec.num_basis_gates(Operator(qc).data)
            self.assertEqual(bl[0].min_2q, ref, (trial, lines))
            seen.add(ref)
        self.assertTrue({0, 1, 2, 3} <= seen, seen)

    def test_fused_kind_matches_qiskit(self):
        """min_2q_without_swap = Qiskit's CX count of SWAP x U, on random blocks that end with a SWAP."""
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import CXGate
        from qiskit.quantum_info import Operator
        from qiskit.synthesis import TwoQubitBasisDecomposer
        dec = TwoQubitBasisDecomposer(CXGate())
        rnd = random.Random(5)
        kinds = set()
        for trial in range(40):
            qc = QuantumCircuit(2)
            lines = []
            for _ in range(rnd.choice([1, 2])):
                t = rnd.choice([math.pi / 2, rnd.uniform(-3, 3)])
                g = rnd.choice(["cz", "cp", "rzz"])
                if g == "cz":
                    qc.cz(0, 1)
                    lines.append("cz $0, $1;")
                else:
                    getattr(qc, g)(t, 0, 1)
                    lines.append(f"{g}({t!r}) $0, $1;")
                q = rnd.randrange(2)
                qc.sx(q)
                lines.append(f"sx ${q};")
            if rnd.random() < 0.6:
                qc.swap(0, 1)
                lines.append("swap $0, $1;")
            b = sbs.blocks(phys("\n".join(lines) + "\n"))[0]
            sw = QuantumCircuit(2)
            sw.swap(0, 1)
            self.assertEqual(b.min_2q, dec.num_basis_gates(Operator(qc).data), (trial, lines))
            self.assertEqual(b.min_2q_without_swap, dec.num_basis_gates(Operator(qc.compose(sw)).data),
                             (trial, lines))
            kinds.add(b.kind)
        self.assertTrue({"gate", "fused"} <= kinds, kinds)


if __name__ == "__main__":
    unittest.main()
