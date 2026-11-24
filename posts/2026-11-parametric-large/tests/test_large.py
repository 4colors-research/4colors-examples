import json

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

import clifford_check as cc
import run
from qaoakit import ansatz as az, apikit as ak


def test_specs_have_hundreds_of_parameters():
    """The default ansatze are 100-150 qubits with up to hundreds of parameters (built, not simulated)."""
    sizes = {s: az.build(s)[0] for s in ("ma-100", "hea-ring-150x2")}
    assert sizes["ma-100"].num_qubits == 100 and len(sizes["ma-100"].parameters) == 250
    assert sizes["hea-ring-150x2"].num_qubits == 150 and len(sizes["hea-ring-150x2"].parameters) == 600


def test_clifford_values_are_multiples_of_half_pi():
    v = cc.clifford_values(50, seed=3)
    assert len(v) == 50 and all(abs((x / (np.pi / 2)) - round(x / (np.pi / 2))) < 1e-12 for x in v)
    assert v == cc.clifford_values(50, seed=3) and v != cc.clifford_values(50, seed=4)


@pytest.mark.parametrize("spec", run.DEFAULT_ANSATZ.split(","))
def test_every_default_spec_binds_to_a_clifford_circuit_at_full_size(spec):
    """At Clifford points the 100-150 qubit inputs are stabilizer circuits: the check runs at full size."""
    qc, _ = az.build(spec)
    r = cc.stabilizer_match(qc, qc, cc.clifford_values(len(qc.parameters), 1))
    assert r == dict(ok=True, method="stabilizer", active_qubits=qc.num_qubits)


def test_check_follows_routing_and_catches_a_wrong_binding():
    """A circuit routed by Qiskit onto the heavy-hex map (layout and moved qubits) passes; the same circuit bound
    with one parameter off by pi/2 fails."""
    qc, _ = az.build("hea-ring-6x2")
    out = transpile(qc, target=ak.qiskit_target(), optimization_level=3, seed_transpiler=3)
    assert out.layout is not None
    assert cc.check_many(qc, out, draws=4, seed=1)["clifford_ok"] is True
    v = cc.clifford_values(len(qc.parameters), 7)
    wrong = list(v)
    wrong[0] += np.pi / 2                      # ry on qubit 0 in the first layer: a different state
    assert cc.stabilizer_match(qc, out, v)["ok"] is True
    assert cc.stabilizer_match(qc, out, v, compiled_values=wrong)["ok"] is False


def test_qiskit_o3_output_is_clifford_block_by_block():
    """Qiskit O3 re-synthesizes blocks into rotations that are Clifford only as a whole: the block merge checks it."""
    qc, _ = az.build("ma-20")
    out = transpile(qc, target=ak.qiskit_target(), optimization_level=3, seed_transpiler=1000)
    r = cc.check_many(qc, out, draws=2, seed=1)
    assert r["clifford_ok"] is True and r["clifford_method"].startswith("stabilizer")


def test_check_makes_no_claim_on_a_non_clifford_circuit():
    a = Parameter("a")
    qc = QuantumCircuit(2)
    qc.rz(a, 0)
    qc.cx(0, 1)
    out = qc.copy()
    out.t(1)                                   # a fixed non-Clifford gate in the "compiled" circuit
    r = cc.stabilizer_match(qc, out, [np.pi / 2])
    assert r["ok"] is None and r["method"] == "not_clifford" and r["side"] == "compiled"
    many = cc.check_many(qc, out, draws=2, seed=1)
    assert many["clifford_ok"] is None and many["clifford_method"] == "not_clifford"


def test_check_sees_a_dropped_gate():
    qc, _ = az.build("ma-6")
    out = qc.copy_empty_like()
    for ins in qc.data:
        out.append(ins)
    # drop the last rx: the "compiled" circuit is missing one gate
    last = max(i for i, ins in enumerate(out.data) if ins.operation.name == "rx")
    del out.data[last]
    r = cc.check_many(qc, out, draws=8, seed=2)
    assert r["clifford_ok"] is False and r["clifford_mismatches"] >= 1


def _row(spec, budget, arm, gates, seed=1, ok=True, refused=False):
    r = dict(kind="compile", ansatz=spec, budget_s=budget, arm=arm, seed=seed, n_qubits=100, input_n_parameters=250)
    if refused:
        return dict(r, refusal_code="input_too_large")
    return dict(r, two_qubit_gates=gates, two_qubit_depth=gates // 4, clifford_ok=ok, parameters_preserved=True,
                same_parameter_objects=True, bind_s_median=0.002)


def test_selection_shows_strict_wins_with_checks_only():
    import plot

    rows = []
    for seed in (1, 2, 3):
        rows += [_row("win", 10.0, "4colors", 90, seed), _row("win", 10.0, "qiskit-o3", 120, seed),
                 _row("tie", 10.0, "4colors", 100, seed), _row("tie", 10.0, "qiskit-o3", 100, seed),
                 _row("loss", 10.0, "4colors", 130, seed), _row("loss", 10.0, "qiskit-o3", 100, seed),
                 _row("unchecked", 10.0, "4colors", 50, seed), _row("unchecked", 10.0, "qiskit-o3", 100, seed,
                                                                   ok=None if seed == 2 else True),
                 _row("refused", 10.0, "4colors", 50, seed, refused=seed == 3),
                 _row("refused", 10.0, "qiskit-o3", 100, seed)]
    shown, left = plot.select_cells(rows)
    assert [s for s, _, _ in shown] == ["win"]
    why = {s: w for s, _, w in left}
    assert set(why) == {"tie", "loss", "unchecked", "refused"}
    assert "refused" in why["refused"] and "check" in why["unchecked"]


def test_large_end_to_end_offline(canned, tmp_path):
    run.main(["--out", str(tmp_path), "--ansatz", "ma-20,hea-ring-24x1", "--budgets", "0.01", "--seeds", "1,2",
              "--bindings", "5", "--clifford-draws", "2"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 2 * 2
    assert all(r["parameters_preserved"] and r["same_parameter_objects"] and r["position_equals_object"] for r in rows)
    assert all(r["clifford_ok"] is True and r["clifford_method"].startswith("stabilizer") for r in rows)
    assert all(r["simulated"] is False and r["local_times"] == "untimed" for r in rows)
    assert all(r["approximation_mode"] == "exact" for r in rows if r["arm"] == "4colors")
    import plot

    figs = tmp_path / "figs"
    plot.main([str(tmp_path), "--figs", str(figs)])
    assert (figs / "gates.png").exists() and (figs / "bind.png").exists()
    sel = (figs / "selection.md").read_text()
    assert "ma-20" in sel and "Left out" in sel           # the canned compiler returns the input: no win, left out


def test_arms_split_into_two_records(canned, tmp_path):
    run.main(["--out", str(tmp_path / "api"), "--ansatz", "ma-6", "--budgets", "0.01", "--seeds", "1",
              "--bindings", "2", "--arms", "4colors"])
    run.main(["--out", str(tmp_path / "local"), "--ansatz", "ma-6", "--budgets", "0.01", "--seeds", "1",
              "--bindings", "2", "--arms", "qiskit-o3"])
    api = [json.loads(x) for x in (tmp_path / "api" / "results.jsonl").read_text().splitlines()]
    loc = [json.loads(x) for x in (tmp_path / "local" / "results.jsonl").read_text().splitlines()]
    assert {r["arm"] for r in api} == {"4colors"} and {r["arm"] for r in loc} == {"qiskit-o3"}
    with pytest.raises(SystemExit):
        run.main(["--out", str(tmp_path / "x"), "--arms", "qiskit-o1"])
