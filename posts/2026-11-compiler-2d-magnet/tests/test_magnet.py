"""Offline tests of the 2D magnet post: lattice, matchings, circuit, cross-check with the report card's
generator, selection and plumbing. No call reaches the service."""
import importlib.util
import json
import os

import numpy as np
import pytest

from conftest import POST_DIR, load

tfim = load("tfim")


@pytest.mark.parametrize("rows,cols", [(4, 4), (7, 8), (3, 5)])
def test_open_grid_bonds_and_matchings(rows, cols):
    b = tfim.bonds(rows, cols)
    assert len(b) == rows * (cols - 1) + (rows - 1) * cols                 # open: no wrap-around bonds
    m = tfim.matchings(rows, cols)
    assert sorted(e for es in m.values() for e in es) == sorted(b)         # the 4 matchings partition the bonds
    for es in m.values():
        used = [q for e in es for q in e]
        assert len(used) == len(set(used))                                 # no qubit twice in one matching
    assert max(tfim.degrees(rows, cols).values()) == (4 if min(rows, cols) >= 3 else 3)


def test_circuit_counts_and_floor():
    import circuitkit as ck

    qc = tfim.circuit(4, 4, steps=1)
    assert qc.num_qubits == 16
    assert ck.circuit_metrics(qc)["two_qubit_gates"] == tfim.logical_floor(4, 4, 1) == 48
    assert tfim.logical_floor(7, 8, 2) == 2 * 2 * (7 * 7 + 6 * 8)


def test_one_step_is_the_trotter_product():
    """One step equals exp(-i dt ZZ) on every bond, then the field: checked as an operator on a 2 x 2 grid."""
    from qiskit.quantum_info import Operator, SparsePauliOp
    from scipy.linalg import expm

    dt, h = 0.2, 1.0
    qc = tfim.circuit(2, 2, steps=1, dt=dt, h=h, measure=False)
    U = np.eye(16, dtype=complex)
    for u, v in tfim.bonds(2, 2):
        lab = ["I"] * 4; lab[3 - u] = "Z"; lab[3 - v] = "Z"
        U = expm(-1j * dt * SparsePauliOp("".join(lab)).to_matrix()) @ U
    for q in range(4):
        lab = ["I"] * 4; lab[3 - q] = "X"
        U = expm(-1j * h * dt * SparsePauliOp("".join(lab)).to_matrix()) @ U
    assert Operator(qc).equiv(Operator(U))


def test_same_construction_as_the_report_card():
    """The report card's tfim2d (2 steps, square grid) is this post's circuit, gate for gate."""
    from pathlib import Path
    found = sorted(Path(POST_DIR).resolve().parent.glob("*compiler-report-card/families.py"))
    if not found:
        pytest.skip("the report-card post is not next to this one")
    path = str(found[0])
    spec = importlib.util.spec_from_file_location("rc_families_for_magnet", path)
    F = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = F
    spec.loader.exec_module(F)
    theirs, _ = F.make("tfim2d", 4, 101)
    ours = tfim.circuit(4, 4, steps=2)
    from qiskit.converters import circuit_to_dag

    assert circuit_to_dag(theirs) == circuit_to_dag(ours)     # same gates on the same qubits in the same order
    # per qubit (the report card's lowering re-serializes the commuting bonds, so the flat lists differ)


def test_magnetization_helper():
    assert tfim.magnetization_z({"0000": 3, "1111": 1}) == pytest.approx(0.5)


def test_run_and_plot_offline(canned, tmp_path):
    run, plot = load("run"), load("plot")
    out = tmp_path / "m"
    run.main(["--out", str(out), "--grids", "4x4", "--steps", "1", "--seeds", "1", "--budgets", "0.2",
              "--max-qiskit-trials", "1", "--qiskit-streams", "1"])
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    res = [r for r in rows if r["kind"] == "result"]
    assert {r["arm"] for r in res} == {"4colors", "qiskit-o3-best", "qiskit-o3-once"}
    ours = next(r for r in res if r["arm"] == "4colors")
    assert ours["two_qubit_gates_per_step"] == 48 and ours["boundary"] == "open"   # canned: the input unchanged
    assert canned.calls[0]["approximation_mode"] == "exact" and canned.calls[0]["verify"] == "wait"
    plot.main([str(out), "--figs", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / "fig_lattice.png").exists()
    sel = (tmp_path / "figs" / "selection.md").read_text()
    assert "Internal (not for publication): cells shown" in sel
    assert "Cells shown" not in plot.caption({})   # owner 23 Nov: no tallies in the caption


def test_selection_rule_shows_only_strict_wins():
    plot = load("plot")
    def r(arm, g, seed=1):
        return {"kind": "result", "grid": [4, 4], "steps": 1, "budget_s": 10.0, "arm": arm, "seed": seed,
                "qubits": 16, "logical_floor": 48, "two_qubit_gates": g, "two_qubit_depth": 4,
                "two_qubit_gates_per_step": g, "two_qubit_depth_per_step": 4, "over_budget": False}
    win = plot.cells([r("4colors", 80), r("qiskit-o3-best", 100), r("qiskit-o3-once", 120)])
    tie = plot.cells([r("4colors", 99), r("qiskit-o3-best", 100), r("qiskit-o3-once", 120)])
    assert win[((4, 4), 1)]["shown"] and not tie[((4, 4), 1)]["shown"]
    assert "4 x 4" in plot.table(win) and "4 x 4" not in plot.table(tie)
