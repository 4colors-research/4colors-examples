"""Offline tests for the custom-chips post (no API call)."""
import json

import pytest

import circuitkit as ck

import os

_POST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import postload  # noqa: E402  (blog/tests/postload.py; blog/tests is on sys.path from conftest)


def load(name):
    """Import <post dir>/<name>.py under a name unique to this post (blog/tests/postload.py), so posts with a
    run.py or plot.py of their own never collide when their tests run in one session."""
    return postload.load(_POST, name)


chips = load("chips")


def valid(chip):
    """A valid target description: format version 2, couplers listed once as [a, b] with a < b < n, connected."""
    assert set(chip) == {"format_version", "n_qubits", "two_qubit_gate", "directed", "couplers"}
    assert chip["format_version"] == 2 and chip["two_qubit_gate"] == "cz" and chip["directed"] is False
    pairs = chips.coupler_list(chip)
    n = chip["n_qubits"]
    assert all(0 <= a < b < n for a, b in pairs) and len(set(pairs)) == len(pairs)
    assert ck.is_connected(n, pairs)
    json.dumps(chip)                                   # sendable as is
    return pairs


@pytest.mark.parametrize("name", list(chips.TAX_CHIPS))
def test_tax_chips_are_valid(name):
    valid(chips.TAX_CHIPS[name]())


def test_shapes():
    assert len(valid(chips.grid(5, 5))) == 2 * 5 * 4
    assert len(valid(chips.ring(25))) == 25
    assert len(valid(chips.ladder(13))) == 13 + 2 * 12
    hh = chips.heavy_hex(3, 7)
    pairs = valid(hh)
    deg = {}
    for a, b in pairs:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    assert max(deg.values()) == 3 and hh["n_qubits"] == 21 + 2 + 2 == 25
    assert chips.all_to_all(25)["couplers"] == "all_to_all" and len(chips.coupler_list(chips.all_to_all(25))) == 300


@pytest.mark.parametrize("side", [20, 32, 50])
def test_big_grids(side):
    g = chips.grid(side, side)
    assert len(valid(g)) == 2 * side * (side - 1) and g["n_qubits"] == side * side


def test_qiskit_target_matches_the_chip():
    chip = chips.heavy_hex(3, 7)
    qt = chips.qiskit_target(chip)
    assert qt.num_qubits == chip["n_qubits"]
    assert ck.device_couplers(qt) == [tuple(e) for e in chip["couplers"]]
    assert ck.device_couplers(chips.qiskit_target(chips.all_to_all(5))) == \
        [(a, b) for a in range(5) for b in range(a + 1, 5)]


def test_adder_adds():
    from qiskit.quantum_info import Statevector

    qc = load("run").tax_circuits()[2][0]              # 8-bit adder, as compiled in the post
    assert qc.num_qubits == 16 and all(len(i.qubits) <= 2 for i in qc.data)
    small = ck.draper_adder(3, a=5, b=6)
    probs = Statevector(small).probabilities_dict()
    key, p = max(probs.items(), key=lambda kv: kv[1])
    bits = key[::-1]                                     # qubit 0 first
    a = int(bits[:3][::-1], 2)
    s = int(bits[3:][::-1], 2)
    assert p > 0.999 and a == 5 and s == (5 + 6) % 8


def test_metric_agrees_with_qiskit_on_a_grid():
    from qiskit import transpile

    qc = ck.bernstein_vazirani("1" * 8)
    out = transpile(qc, target=ck.qiskit_target_from_custom(chips.grid(3, 3)), optimization_level=1,
                    seed_transpiler=1)
    m = ck.circuit_metrics(out)
    assert m["two_qubit_gates"] == out.count_ops()["cz"]
    assert {(a, b) for a, b, _ in ck.coupler_counts(out)} <= set(chips.coupler_list(chips.grid(3, 3)))


def test_run_and_plot_offline(canned, tmp_path):
    run, plot = load("run"), load("plot")
    canned.refuse = lambda call: chips.description(call["target"])["n_qubits"] > 156   # canned refusal, big chip
    run.main(["--out", str(tmp_path), "--part", "scale", "--sizes", "4", "--grids", "3,13",
              "--budgets", "0.01", "--seeds", "1", "--max-qiskit-trials", "1"])
    run.main(["--out", str(tmp_path), "--part", "tax", "--budgets", "0.01", "--seeds", "1",
              "--max-qiskit-trials", "1", "--skip-qiskit"])
    run.main(["--out", str(tmp_path), "--part", "design", "--designs", "square_54_prx", "--budgets", "0.01",
              "--seeds", "1", "--max-qiskit-trials", "1"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    four = ck.results(rows, arm="4colors")
    big = [r for r in four if r["chip"] == "grid_13x13"]
    assert big and all(r["refusal_code"] == "invalid_option" and r["two_qubit_gates"] is None for r in big)
    assert ck.results(rows, arm="qiskit", chip="grid_13x13")       # Qiskit still compiles it
    tax = [r for r in four if r.get("part") == "tax"]
    assert len(tax) == 4 * len(chips.TAX_CHIPS)
    assert {r["circuit_id"] for r in tax} >= {"uccsd_lih_12"}
    design = [r for r in rows if r.get("kind") == "result" and r.get("part") == "design"]
    assert {r["arm"] for r in design} == {"4colors", "qiskit"} and {r["chip"] for r in design} == {"square_54_prx"}
    sent = [chips.description(c["target"]) for c in canned.calls]
    assert all(t["format_version"] == 2 for t in sent)          # descriptions, never a name
    assert any(not isinstance(c["target"], dict) and c["target"].calibration is None for c in canned.calls)
    # coupler_counts on every answered result row of both arms (the arms check compares it); it sums to the
    # two-qubit gate count. Qiskit's pairs are couplers of the chip row's target (the check before citing); the
    # canned 4colors answer is the unrouted input, so only its sum is checked here.
    targets = {r["chip"]: r["target"] for r in rows if r.get("kind") == "chip" and r.get("target")}
    targets["grid_13x13"] = chips.grid(13, 13)      # over the 156-qubit cap: its chip row stores no target
    answered = [r for r in rows if r.get("kind") == "result" and not r.get("refusal_code")]
    assert {r["arm"] for r in answered} == {"4colors", "qiskit"}
    for r in answered:
        cc = r["coupler_counts"]
        assert isinstance(cc, list) and sum(n for _, _, n in cc) == r["two_qubit_gates"]
        if r["arm"] == "qiskit":
            assert {(a, b) for a, b, _ in cc} <= set(chips.coupler_list(targets[r["chip"]]))
    assert all(r.get("coupler_counts") is None for r in big)          # a refusal has no circuit
    plot.main([str(tmp_path), "--figs", str(tmp_path / "figs")])
    assert {p.name for p in (tmp_path / "figs").iterdir()} == {"fig_tax.png", "fig_scale.png", "fig_design.png",
                                                              "selection.md"}


def test_selection_shows_qiskit_only_where_ours_leads():
    plot = load("plot")
    r = lambda arm, chip, g: ck.row(kind="result", circuit_id="c", arm=arm, chip=chip, part="tax", budget_s=10,
                                    two_qubit_gates=g)
    rows = [r("4colors", "a", 10), r("qiskit", "a", 12), r("4colors", "b", 12), r("qiskit", "b", 12),
            r("4colors", "c", 14), r("qiskit", "c", 11), r("4colors", "d", None), r("qiskit", "d", 9)]
    shown, left = plot.select_cells(rows, "tax")
    assert shown == {("c", "a")} and [x[1] for x in left] == ["b", "c", "d"]


@pytest.mark.parametrize("name", ["heavy_hex_156_cz", "square_108_cz", "square_54_prx", "all_to_all_56_u1q"])
def test_design_lattices_go_to_qiskit_with_their_native_gates(name):
    from qiskit import QuantumCircuit, transpile

    t = chips.design(name)
    assert t.calibration is None and t.description["n_qubits"] <= 156
    qt = chips.qiskit_target(t)
    qc = QuantumCircuit(3)
    qc.h(0); qc.cx(0, 1); qc.cx(1, 2); qc.rz(0.3, 2)
    out = transpile(qc, target=qt, optimization_level=1, seed_transpiler=1)
    g2 = chips.QISKIT_2Q[t.description["two_qubit_gate"]]
    allowed = {g2, "measure", "barrier"} | {chips.QISKIT_1Q[g] for g in (t.description.get("gates_1q")
                                                                        or chips.GATES_1Q)}
    assert set(out.count_ops()) <= allowed and out.count_ops()[g2] >= 2
    assert {(a, b) for a, b, _ in ck.coupler_counts(out)} <= set(chips.coupler_list(t))


def test_uccsd_register_and_excitations():
    assert ck.uccsd_hf_occupied(6, 4) == [0, 1, 6, 7]
    assert len(ck.uccsd_excitations(6, 4)) == 16 + 6 + 6 + 64          # the 92 of UCCSD on LiH STO-3G
    assert len(ck.uccsd_excitations(4, 2)) == 6 + 9


def test_uccsd_keeps_particle_number_and_spin_and_leaves_the_reference():
    from qiskit.quantum_info import Statevector

    qc = ck.uccsd(4, 2, seed=3, measure=False)
    probs = Statevector(qc).probabilities_dict()
    for key, p in probs.items():
        bits = key[::-1]                                             # qubit 0 first
        if p > 1e-9:
            assert bits[:4].count("1") == 1 and bits[4:].count("1") == 1
    assert max(probs.values()) < 0.99                                # not the bare reference state
    assert set(qc.count_ops()) <= {"cx", "rz", "sx", "x", "h"}


def test_uccsd_without_reference_is_the_identity_on_zero():
    """The bug the old generator had: excitations act as zero on the vacuum."""
    from qiskit.quantum_info import Statevector

    qc = ck.uccsd(4, 2, seed=3, reference=False, measure=False)
    assert abs(Statevector(qc).probabilities()[0] - 1) < 1e-9
