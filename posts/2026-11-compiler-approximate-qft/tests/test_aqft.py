"""Offline tests for the approximate-QFT post (no API call): the scorer, the AQFT inputs, the matching rule and
the scripts' plumbing on fixed canned results."""
import json

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from conftest import load

import circuitkit as ck

check = load("check")


def test_exact_circuit_scores_zero():
    r = check.infidelity(ck.qft(4), ck.qft(4), n_inputs=6, method="statevector")
    assert r["infidelity"] < 1e-12 and r["active_qubits"] == 4


def test_dropped_phases_are_measured():
    a = check.aqft(6, 2)                                   # keeps phases >= pi/4: drops pi/8 and smaller
    assert a is not None
    r = check.infidelity(a, ck.qft(6), n_inputs=16, method="statevector")
    assert 1e-4 < r["infidelity"] < 0.5 and r["worst_infidelity"] >= r["infidelity"]


def test_input_side_z_error_is_caught():
    """A Z on an input qubit is invisible to computational-basis inputs; the |+>-type inputs see it."""
    bad = QuantumCircuit(4)
    bad.z(0)
    bad.compose(ck.qft(4), inplace=True)
    basis_only = [check.fidelity_one(bad, ck.qft(4), labels, "statevector")
                  for labels in (["0"] * 4, ["1", "0", "1", "1"])]
    assert min(basis_only) > 1 - 1e-12
    assert check.infidelity(bad, ck.qft(4), n_inputs=16, method="statevector")["infidelity"] > 0.1


def test_routed_circuit_with_ancilla_scores_zero():
    """Layout and routing: a QFT on a 6-qubit line with a shuffled start still scores as exact."""
    cm = CouplingMap.from_line(6)
    out = transpile(ck.qft(5), coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"], optimization_level=1,
                    initial_layout=[4, 2, 0, 1, 5], seed_transpiler=3)
    r = check.infidelity(out, ck.qft(5), n_inputs=6, method="statevector")
    assert r["infidelity"] < 1e-9 and r["active_qubits"] >= 5


def test_mps_agrees_with_statevector():
    a = check.aqft(6, 3)
    sv = check.infidelity(a, ck.qft(6), n_inputs=4, method="statevector")["infidelity"]
    mps = check.infidelity(a, ck.qft(6), n_inputs=4, method="matrix_product_state")["infidelity"]
    assert abs(sv - mps) < 1e-9


def test_aqft_degree_and_scope():
    assert check.aqft(8, 6).count_ops()["cp"] == 27 and check.aqft(4, 6) is None
    assert check.dropped_phase_bound(8, 6) > 0
    run = load("run")
    assert [i for i, _ in run.inputs_for(12, [6, 10])] == ["qft", "aqft6", "aqft10"]
    with pytest.raises(ValueError):
        run.inputs_for(33, [6])


def _r(n, arm, inp, mode, g, inf, seed=1, **kw):
    return ck.row(kind="result", qubits=n, arm=arm, input_id=inp, mode=mode, two_qubit_gates=g, infidelity=inf,
                  seed=seed, **kw)


def test_matching_rule_wins_only():
    plot = load("plot")
    rows = [
        _r(20, "4colors", "qft", "exact", 500, 1e-12), _r(20, "4colors", "qft", "per_block", 420, 1e-8),
        _r(20, "4colors", "qft", "per_block", 430, 3e-8, seed=2),        # worst seed counts for us
        _r(20, "qiskit", "qft", "degree 1", 520, 1e-13),                 # exact vs exact (floor): ours 500 wins
        _r(20, "qiskit", "qft", "degree 0.9999", 450, 2e-8),             # ours per_block worst 3e-8 > 2e-8: exact 500 loses
        _r(20, "qiskit", "qft", "degree 0.999999", 440, 5e-8),           # per_block 430 (median 425) wins
        _r(20, "qiskit", "aqft6", "degree 1", 300, 1e-3, over_budget=True),
        _r(20, "qiskit", "aqft10", "degree 1", 600, None),
    ]
    got = {(x["qiskit"]["input_id"], x["qiskit"]["mode"]): x for x in plot.pairs(rows)}
    assert got[("qft", "degree 1")]["shown"] and got[("qft", "degree 1")]["match"]["mode"] == "exact"
    assert not got[("qft", "degree 0.9999")]["shown"] and "not fewer" in got[("qft", "degree 0.9999")]["why"]
    w = got[("qft", "degree 0.999999")]
    assert w["shown"] and w["match"]["mode"] == "per_block" and w["match"]["infidelity"] == 3e-8
    assert "reported apart" in got[("aqft6", "degree 1")]["why"]
    assert "not checked" in got[("aqft10", "degree 1")]["why"]


def test_flags():
    plot = load("plot")
    assert "approximation_spent missing" in plot.flags(dict(mode="per_block", two_qubit_gates=5))
    assert plot.flags(dict(mode="exact", verification_label="verified", infidelity=1e-3))
    assert not plot.flags(dict(mode="exact", verification_label="verified", infidelity=1e-12))


def test_run_and_plot_offline(canned, tmp_path):
    run, plot = load("run"), load("plot")
    run.main(["--out", str(tmp_path), "--sizes", "4", "--aqft-m", "1", "--seeds", "1", "--budgets", "0.01",
              "--qiskit-degrees", "0.999", "--max-qiskit-trials", "1", "--check-method", "statevector",
              "--check-inputs", "3", "--verify", "skip"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    four = ck.results(rows, arm="4colors")
    assert sorted((r["input_id"], r["mode"]) for r in four) == [("aqft1", "exact"), ("qft", "exact"),
                                                                ("qft", "per_block")]
    assert all(r["infidelity"] is not None for r in four)
    exact = [r for r in four if r["input_id"] == "qft"]
    assert all(r["infidelity"] < 1e-9 for r in exact)      # canned: the input comes back unchanged
    q = ck.results(rows, arm="qiskit")
    assert sorted(r["mode"] for r in q) == ["degree 0.999", "degree 1", "degree 1"]
    assert all(r["coupler_set"] == "full" and r["infidelity"] is not None for r in q)
    plot.main([str(tmp_path), "--figs", str(tmp_path / "figs")])
    for f in ("fig_pairs.png", "fig_own.png", "labels.md", "selection.json"):
        assert (tmp_path / "figs" / f).exists()
