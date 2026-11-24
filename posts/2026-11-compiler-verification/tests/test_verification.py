"""Offline tests for the verification post (no API call)."""
import json

from conftest import load

import circuitkit as ck


def test_qft_matches_qiskit_qft():
    from qiskit.circuit.library import QFTGate
    from qiskit.quantum_info import Operator

    qc = ck.qft(4)
    full = qc.copy()
    for i in range(2):                                   # the QFT's final swaps, left out of ck.qft
        full.swap(i, 3 - i)
    assert Operator(full).equiv(Operator(QFTGate(4)))
    assert ck.circuit_metrics(qc)["two_qubit_gates"] == 4 * 3 // 2       # one cp per pair


def test_small_chip_is_a_connected_2x3_grid():
    run = load("run")
    chip = run.small_chip()
    assert chip["n_qubits"] == 6 and len(chip["couplers"]) == 7
    assert ck.is_connected(6, chip["couplers"])


def test_custom_target_is_the_clients_description_format():
    from fourcolors.quantum import targets

    chip = ck.custom_target(3, [[1, 0], [2, 1], [0, 1]])
    assert chip == targets.build_description(3, "cz", ("rz", "sx", "x"), [[0, 1], [1, 2]])
    assert chip["format_version"] == 2 and chip["directed"] is False and chip["couplers"] == [[0, 1], [1, 2]]
    assert ck.custom_target(4, "all_to_all")["couplers"] == "all_to_all"
    assert ck.qiskit_target_from_custom(chip).num_qubits == 3


def test_local_check_sees_layout_and_differences():
    from qiskit import transpile

    run = load("run")
    qc = ck.qft(5)
    out = transpile(qc, target=ck.qiskit_target_from_custom(run.small_chip()), optimization_level=3, seed_transpiler=2)
    ok, infid = run.local_check(out, qc)
    assert ok and infid < 1e-9
    broken = out.copy()
    broken.rz(0.3, 0)
    ok, infid = run.local_check(broken, qc)
    assert not ok and infid > 1e-3


def test_run_and_plot_offline(canned, tmp_path):
    run, plot = load("run"), load("plot")
    run.main(["--out", str(tmp_path), "--sizes", "4,6,12", "--seeds", "1", "--budgets", "0.01",
              "--max-qiskit-trials", "1"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    four = ck.results(rows, arm="4colors")
    assert len(four) == 3 * 2
    assert {(r["qubits"], r["options"]["approximation_mode"]) for r in four} == \
        {(n, m) for n in (4, 6, 12) for m in ("exact", "per_block")}
    small = [r for r in four if r["qubits"] <= 10]
    assert small and all(r["local_equivalent"] is True for r in small)     # canned: the input circuit comes back unchanged
    assert all(r.get("local_equivalent") is None for r in four if r["qubits"] > 10)
    assert all(c["verify"] in ("wait", "background") for c in canned.calls)
    sent = [c["target"] for c in canned.calls if c["verify"] == "wait"]
    assert isinstance(sent[0], dict) and sent[-1] is ck.heavy_hex_target()
    bg = [r for r in rows if r["kind"] == "background"]
    assert len(bg) == 1 and bg[0]["label_at_return"] and "waited_s" in bg[0]
    plot.main([str(tmp_path), "--figs", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / "fig_labels.png").exists()


def test_qiskit_counts_only_where_distance_measured(canned, tmp_path):
    """Rule D2: Qiskit rows without a measured distance (here the 12-qubit QFT on the heavy-hex device) are never
    drawn or printed; the small-chip rows, checked locally, are."""
    run, plot = load("run"), load("plot")
    run.main(["--out", str(tmp_path), "--sizes", "6,12", "--seeds", "1", "--budgets", "0.01",
              "--max-qiskit-trials", "1", "--no-background"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    q = plot.qiskit_measured(rows)
    assert {r["qubits"] for r in q} == {6}
    assert all(r["local_equivalent"] is True for r in q)         # Qiskit's own output, checked here
    big = [r for r in ck.results(rows, arm="qiskit") if r["qubits"] == 12]
    assert big and all(r.get("local_infidelity") is None for r in big)
    assert "approximate" in plot.MODE_LABEL["per_block"] and "near-identity" in plot.MODE_LABEL["per_block"]


def test_qft_max_rule():
    run = load("run")
    assert run.QFT_MAX == 32
    assert run.circuit_for(32)[1]["family"] == "qft"
    assert run.circuit_for(34)[1]["family"] == "qaoa"           # never a QFT above 32 qubits
