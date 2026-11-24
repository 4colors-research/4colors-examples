"""Offline tests for the give-it-seconds post (no API call)."""
import json

from conftest import load

import circuitkit as ck


def test_barabasi_albert_graph():
    edges = ck.barabasi_albert(15, 3, seed=15)
    assert len(edges) == 3 * (15 - 3)                  # networkx: m edges per node after the first m
    assert all(a < b for a, b in edges) and len(set(edges)) == len(edges)
    assert edges == ck.barabasi_albert(15, 3, seed=15)  # fixed by the seed


def test_qaoa_cost_layer_per_edge_and_layer():
    edges = ck.barabasi_albert(10, 2, seed=1)
    qc = ck.qaoa_maxcut(10, edges, p=3)
    assert qc.count_ops()["rzz"] == 3 * len(edges)
    assert ck.circuit_metrics(qc)["two_qubit_gates"] == 3 * len(edges)
    first = [tuple(sorted(qc.find_bit(q).index for q in i.qubits)) for i in qc.data if i.operation.name == "rzz"]
    assert first[:len(edges)] == edges


def test_qaoa_state_is_mixer_times_cut_phase():
    """The circuit prepares exp(-i beta B) exp(-i gamma C) |+...+> with C the cut and B = sum X (up to a
    global phase): checked against a direct numpy construction on 4 qubits."""
    import numpy as np
    from qiskit.quantum_info import Statevector, state_fidelity

    n, edges, g, b = 4, [(0, 1), (1, 2), (2, 3), (0, 2)], 0.7, 0.3
    sv = Statevector(ck.qaoa_maxcut(n, edges, p=1, gammas=[g], betas=[b]))
    x = np.arange(2 ** n)
    bits = (x[:, None] >> np.arange(n)) & 1                      # qubit k = bit k (Qiskit order)
    cut = sum((bits[:, i] != bits[:, j]).astype(float) for i, j in edges)
    psi = np.exp(-1j * g * cut) / np.sqrt(2 ** n)
    rx = np.array([[np.cos(b), -1j * np.sin(b)], [-1j * np.sin(b), np.cos(b)]])   # exp(-i b X)
    mixer = np.array([[1.0]])
    for _ in range(n):
        mixer = np.kron(rx, mixer)
    assert state_fidelity(sv, Statevector(mixer @ psi)) > 1 - 1e-12


def test_best_within_picks_the_best_seed_in_time():
    trials = [{"cumulative_s": 1, "two_qubit_gates": 50, "two_qubit_depth": 9, "depth": 30},
              {"cumulative_s": 2, "two_qubit_gates": 40, "two_qubit_depth": 9, "depth": 30},
              {"cumulative_s": 5, "two_qubit_gates": 30, "two_qubit_depth": 9, "depth": 30}]
    assert ck.best_within(trials, 3)["two_qubit_gates"] == 40
    assert ck.best_within(trials, 10)["two_qubit_gates"] == 30
    early = ck.best_within(trials, 0.5)
    assert early["two_qubit_gates"] == 50 and early["over_budget"]


def test_run_and_plot_offline(canned, tmp_path):
    run, plot = load("run"), load("plot")
    run.main(["--out", str(tmp_path), "--sizes", "6", "--regular-n", "6", "--layers", "1",
              "--budgets", "0.01,0.02", "--seeds", "1", "--max-qiskit-trials", "2"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    inc = [r for r in rows if r["kind"] == "incumbent"]
    assert [r["index"] for r in inc] == [0, 1, 0, 1]         # two circuits, two incumbents each
    assert [r["status"] for r in inc] == ["running", "finished"] * 2
    assert all(r["t_client_s"] >= 0 for r in inc)
    res = ck.results(rows, arm="4colors")
    assert sorted({r["budget_s"] for r in res}) == [0.01, 0.02] and len(res) == 4
    trials = [r for r in rows if r["kind"] == "trial"]
    assert all(r["cumulative_s"] > 0 for r in trials)
    plot.main([str(tmp_path), "--figs", str(tmp_path / "figs")])
    assert {p.name for p in (tmp_path / "figs").iterdir()} == {"fig_budget.png", "fig_margin.png", "fig_stream.png",
                                                               "selection.md"}
    assert {r["coupler_set"] for r in rows if r.get("arm") == "qiskit"} == {"full"}      # no "working" arm
    assert all(c.get("approximation_mode") == "exact" for c in canned.calls)


def _cell(arm, cset, b, g, seed=1, over=False):
    return {"kind": "result", "circuit_id": "qaoa_ba3_15_p3", "arm": arm, "coupler_set": cset, "budget_s": b,
            "seed": seed, "two_qubit_gates": g, "over_budget": over}


def test_selection_rule_and_margin():
    plot = load("plot")
    rows = [_cell("4colors", "fakefez", 1, 90), _cell("qiskit", "full", 1, 100, over=True),      # overran: apart
            _cell("4colors", "fakefez", 3, 80), _cell("qiskit", "full", 3, 100),                  # win, 20 %
            _cell("4colors", "fakefez", 10, 100), _cell("qiskit", "full", 10, 100),               # tie
            _cell("4colors", "fakefez", 60, 60), _cell("4colors", "fakefez", 60, 70, seed=2),
            _cell("qiskit", "full", 60, 100), _cell("qiskit", "full", 60, 100, seed=2),          # win, median 65
            _cell("qiskit", "working", 60, 10)]                                                   # never compared
    shown, left = plot.select_cells(rows)
    assert shown == {"qaoa_ba3_15_p3": {3: (80, 100), 60: (65.0, 100)}}
    why = {b: w for _, b, w in left}
    assert "overran" in why[1] and why[10] == "tie or behind"
    assert plot.margin_pct(80, 100) == 20.0
    md = plot.selection_md(shown, left)
    assert "Internal (not for publication): cells shown 2 of 4." in md
    assert "Cells shown:" not in md   # owner 23 Nov: no tallies in the caption
