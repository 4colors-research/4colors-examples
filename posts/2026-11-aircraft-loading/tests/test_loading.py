"""Offline tests of the aircraft loading post: rows, checker, reference, penalty form, QAOA circuit, selection and
plumbing. Nothing calls the service."""
import itertools
import json

import pytest

from conftest import load

L = load("loading")


@pytest.mark.parametrize("q", [12, 20, 28])
def test_paper_scale_sizes(q):
    h = L.paper_scale(q, seed=1)
    assert h.n <= q and h.n >= q - 8
    assert all(h.weights[i] <= h.caps[j] for i, j in h.pairs)


def test_rows_admit_exactly_the_valid_plans():
    h = L.generate(3, 4, seed=3)
    assert h.n <= 12
    m = L.model(h)
    for x in itertools.product((0, 1), repeat=h.n):
        assert m.feasible(list(x)) == L.check(h, x)["valid"]


def test_check_catches_each_rule():
    h = L.generate(3, 4, seed=3)
    k0 = 0
    i0, j0 = h.pairs[k0]
    two_pos = [k for k, (i, j) in enumerate(h.pairs) if i == i0][:2]
    x = [0] * h.n
    for k in two_pos:
        x[k] = 1
    assert "a container in two positions" in L.check(h, x)["problems"]
    fwd = [k for k, (i, j) in enumerate(h.pairs) if h.arms[j] == min(h.arms)]
    x = [0] * h.n; x[fwd[0]] = 1
    assert "center of gravity outside the envelope" in L.check(h, x)["problems"]   # one container at the nose


@pytest.mark.parametrize("seed", [1, 2])
def test_reference_is_the_brute_force_optimum(seed):
    h = L.paper_scale(20, seed=seed)
    ref = L.milp_reference(h, 10)
    assert ref["optimal"] and ref["reference_value"] == L.brute_force(h)
    assert L.check(h, ref["reference_x"])["valid"]


def test_qaoa_circuit_structure():
    import circuitkit as ck

    h = L.paper_scale(12, seed=1)
    Q, lin = L.penalty_qubo(h)
    assert len(Q) >= 0.8 * h.n * (h.n - 1) // 2    # squared weight and balance rows: nearly every pair interacts
    for p in (1, 2):
        qc = L.qaoa_circuit(h, p=p)
        assert qc.num_qubits == h.n
        assert ck.circuit_metrics(qc)["two_qubit_gates"] == p * len(Q)


def test_run_and_plot_offline(canned, canned_qubo, tmp_path, monkeypatch):
    run, plot = load("run"), load("plot")
    monkeypatch.setattr(run, "PAPER_QUBITS", (12,))
    monkeypatch.setattr(run, "LARGE", ())
    out = tmp_path / "aero"
    run.main(["--out", str(out), "--holds", "1", "--seeds", "1", "--budgets", "0.2", "--layers", "1",
              "--max-qiskit-trials", "1", "--qiskit-streams", "1", "--reference-time", "10"])
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    solve = next(r for r in rows if r["kind"] == "solve")
    assert solve["valid"] and solve["load_kg"] == 0                         # canned all-zeros: an empty hold
    ref = next(r for r in rows if r["kind"] == "reference")
    assert ref["optimal"] and ref["reference_valid"]
    assert {r["arm"] for r in rows if r["kind"] == "result"} == {"4colors", "qiskit-o3-best", "qiskit-o3-once"}
    plot.main([str(out), "--figs", str(tmp_path / "figs")])
    sel = (tmp_path / "figs" / "selection.md").read_text()
    assert "Cells shown: 0 of 1 (QUBO)" in sel                              # an empty hold never matches


def test_selection_rules():
    plot = load("plot")
    rows = [{"kind": "reference", "hold": "paper12", "hold_seed": 1, "optimal": True, "reference_value": 5000},
            {"kind": "solve", "hold": "paper12", "hold_seed": 1, "time_limit_s": 1.0, "n_vars": 12, "valid": True,
             "load_kg": 5000}]
    assert plot.qubo_cells(rows)[("paper12", 1.0)]["shown"]
    rows[1]["load_kg"] = 4950
    assert not plot.qubo_cells(rows)[("paper12", 1.0)]["shown"]
