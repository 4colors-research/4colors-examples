"""Offline tests for the broken-couplers post (no API call)."""
import json

from conftest import load

import circuitkit as ck


def test_fez_has_failed_couplers_and_calibration_marks_them():
    sets, failed = ck.heavy_hex_coupler_sets()
    assert len(failed) == 7
    cal = ck.calibration_from_target(sets["full"])
    off = {tuple(c["qubits"]) for c in cal["couplers"] if c.get("available") is False}
    assert off == failed
    assert cal["format_version"] == 1 and len(cal["qubits"]) == 156


def test_the_told_target_is_the_clients_description_and_calibration():
    """The 4colors arms send what a customer sends: from_qiskit(FakeFez()) (description + calibration), or its
    description alone ("not told"). The failed couplers are in the description and at error 1.0 in the calibration."""
    sets, failed = ck.heavy_hex_coupler_sets()
    told = ck.heavy_hex_target()
    assert told.description["n_qubits"] == 156 and told.description["two_qubit_gate"] == "cz"
    assert {tuple(sorted(e)) for e in told.description["couplers"]} == set(ck.device_couplers(sets["full"]))
    at_one = {tuple(sorted(c["qubits"])) for c in told.calibration["couplers"] if (c.get("error") or 0) >= 0.999}
    assert at_one == failed


def test_chip_native_uses_every_coupler():
    run = load("run")
    sets, failed = ck.heavy_hex_coupler_sets()
    qc = run.chip_native(ck.device_couplers(sets["full"]))
    edges = {(a, b) for a, b, _ in ck.interaction_edges(qc)}
    assert edges == set(ck.device_couplers(sets["full"])) and failed <= edges
    assert ck.circuit_metrics(qc, failed)["gates_on_failed_couplers"] == 7


def test_esp_is_zero_on_a_failed_coupler():
    from qiskit import QuantumCircuit

    sets, failed = ck.heavy_hex_coupler_sets()
    good = QuantumCircuit(156)
    good.cz(0, 1)
    good.sx(0)
    esp = ck.estimated_esp(good, sets["full"])
    e_cz = sets["full"]["cz"][(0, 1)].error
    e_sx = sets["full"]["sx"][(0,)].error
    assert abs(esp - (1 - e_cz) * (1 - e_sx)) < 1e-12
    bad = good.copy()
    bad.cz(*sorted(failed)[0])
    assert ck.estimated_esp(bad, sets["full"]) == 0.0


def test_qaoa_has_one_rzz_per_edge():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    qc = ck.qaoa_maxcut(4, edges, p=2)
    assert qc.count_ops()["rzz"] == 8
    assert [list(e) for e in edges] == [[a, b] for a, b in
                                        [tuple(qc.find_bit(q).index for q in i.qubits) for i in qc.data
                                         if i.operation.name == "rzz"][:4]]


def test_run_and_plot_offline(canned, tmp_path):
    run, plot = load("run"), load("plot")
    run.main(["--out", str(tmp_path), "--circuits", "qaoa_8,bv_6", "--budgets", "0.01", "--seeds", "1",
              "--max-qiskit-trials", "1"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    res = ck.results(rows)
    assert {(r["arm"], r["coupler_set"]) for r in res} == {
        ("qiskit", "working"), ("qiskit", "full"), ("qiskit", "lower_of_both"),
        ("4colors", "full"), ("4colors", "full+calibration")}
    assert all("estimated_esp" in r for r in res)
    assert all(r["estimated_esp"] is not None for r in res if r["arm"] == "qiskit")   # canned 4colors rows are unmapped
    calls = canned.calls
    assert [isinstance(c["target"], dict) for c in calls] == [True, False] * 2
    assert all(r["calibration_given"] == (r["coupler_set"] == "full+calibration") for r in ck.results(rows, arm="4colors"))
    plot.main([str(tmp_path), "--figs", str(tmp_path / "figs")])
    assert {p.name for p in (tmp_path / "figs").iterdir()} == {"fig_counts.png", "fig_esp.png", "fig_failed.png",
                                                               "selection.json"}
    sel = json.loads((tmp_path / "figs" / "selection.json").read_text())
    assert sorted(sel["shown"] + sel["left_out"]) == ["bv_6", "qaoa_8"]


def test_selection_rule_keeps_only_told_ahead_of_qiskit_full():
    plot = load("plot")
    base = {"kind": "result", "budget_s": 30, "seed": 1}
    rows = [dict(base, circuit_id="a", arm="qiskit", coupler_set="full", estimated_esp=0.0),
            dict(base, circuit_id="a", arm="4colors", coupler_set="full+calibration", estimated_esp=0.2),
            dict(base, circuit_id="b", arm="qiskit", coupler_set="full", estimated_esp=0.3),
            dict(base, circuit_id="b", arm="4colors", coupler_set="full+calibration", estimated_esp=0.3),
            dict(base, circuit_id="c", arm="qiskit", coupler_set="full", estimated_esp=0.1),
            dict(base, circuit_id="c", arm="4colors", coupler_set="full", estimated_esp=0.9)]
    shown, left = plot.select(rows, 30)
    assert shown == ["a"] and left == ["b", "c"]     # ties and "not told" never count as a win


def test_check_your_own_circuits_snippet():
    from qiskit import QuantumCircuit

    check = load("check_circuit")
    sets, failed = ck.heavy_hex_coupler_sets()
    assert check.dead_couplers(sets["full"]) == failed
    a, b = sorted(failed)[0]
    qc = QuantumCircuit(156)
    qc.cz(0, 1)
    qc.cz(b, a)
    qc.cz(a, b)
    qc.barrier()
    qc.measure_all()
    rep = check.gates_on_dead_couplers(qc, sets["full"])
    assert rep["two_qubit_gates"] == 3 and rep["gates_on_dead"] == 2
    assert rep["by_coupler"] == {f"{a}-{b}": 2} and len(rep["dead_couplers"]) == 7
    assert check.gates_on_dead_couplers(qc, sets["working"])["gates_on_dead"] == 0
