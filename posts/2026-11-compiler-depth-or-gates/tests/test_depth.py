"""Offline tests for the depth-or-gates post (no API call)."""
import json

from conftest import load

import circuitkit as ck


def test_circuit_names():
    run = load("run")
    qc, circ = run.build("qaoa_3reg_10_p2")
    assert circ["input_2q"] == 2 * 15 and qc.num_qubits == 10          # 3-regular on 10 nodes: 15 edges
    qc, circ = run.build("qaoa_ba2_8_p1")
    assert circ["input_2q"] == len(ck.barabasi_albert(8, 2, seed=8))
    qc, circ = run.build("adder_3b")
    assert qc.num_qubits == 6 and all(len(i.qubits) <= 2 for i in qc.data)


def test_depth_metric_matches_qiskit():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.h(0); qc.cz(0, 1); qc.cz(1, 2); qc.rz(0.1, 0); qc.barrier(); qc.measure(range(3), range(3))
    m = ck.circuit_metrics(qc)
    assert m == {"two_qubit_gates": 2, "two_qubit_depth": 2, "depth": 3, "gates_on_failed_couplers": 0}
    assert qc.depth() == 4                     # Qiskit's own depth counts the measurement layer


def test_by_depth_pick_is_on_the_published_map():
    run = load("run")
    trials = [{"kind": "trial", "coupler_set": cs, "cumulative_s": t, "two_qubit_gates": g, "two_qubit_depth": d2,
               "depth": d}
              for cs, t, g, d2, d in (("full", 1, 40, 30, 90), ("full", 2, 45, 22, 70), ("full", 9, 38, 20, 60),
                                      ("working", 1, 10, 5, 10))]
    picks = run.by_depth_rows(trials, [5, 10, 0.5])
    assert [(r["budget_s"], r["two_qubit_depth"]) for r in picks] == [(5, 22), (10, 20), (0.5, 30)]
    assert [r["over_budget"] for r in picks] == [False, False, True]
    assert all(r["coupler_set"] == "full, by two-qubit depth" for r in picks)


def test_each_qiskit_pick_meets_our_matching_objective():
    run = load("run")
    assert run.QISKIT_SET == "full"
    assert run.PAIRS == [("full", "two_qubit", "two_qubit_gates"),
                         ("full, by two-qubit depth", "depth", "two_qubit_depth")]
    assert run.OBJECTIVES == {"two_qubit": {"objective": "two_qubit"}, "depth": {"objective": "depth"}}


def _res(arm, cset, b, g, d2, seed=1, over=False):
    return {"kind": "result", "circuit_id": "c", "arm": arm, "coupler_set": cset, "budget_s": b, "seed": seed,
            "two_qubit_gates": g, "two_qubit_depth": d2, "depth": 2 * d2, "over_budget": over}


def test_selection_shows_strict_wins_only():
    plot = load("plot")
    rows = [_res("4colors", "two_qubit", 30, 100, 50), _res("qiskit", "full", 30, 120, 40),          # gates: win
            _res("4colors", "depth", 30, 110, 40), _res("qiskit", "full, by two-qubit depth", 30, 125, 40),  # tie
            _res("4colors", "two_qubit", 1, 100, 50), _res("qiskit", "full", 1, 120, 40, over=True),   # overran
            _res("4colors", "depth", 1, 110, 30), _res("qiskit", "full, by two-qubit depth", 1, 125, 35)]  # win
    shown, left = plot.select_pairs(rows)
    assert [(s["budget_s"], s["objective"], s["ours"], s["qiskit"]) for s in shown] == \
        [(1, "depth", 30, 35), (30, "two_qubit", 100, 120)]
    why = {(b, o): w for _, b, o, w in left}
    assert why[(30, "depth")] == "tie or behind" and "overran" in why[(1, "two_qubit")]
    md = plot.selection_md(shown, left, 4)
    assert "Internal (not for publication): pairs shown 2 of 4." in md
    assert "Pairs shown:" not in md   # owner 23 Nov: no tallies in the caption


def test_run_and_plot_offline(canned, tmp_path):
    run, plot = load("run"), load("plot")
    run.main(["--out", str(tmp_path), "--circuits", "qaoa_3reg_6_p1,adder_2b", "--budgets", "0.01",
              "--seeds", "1,2", "--max-qiskit-trials", "2"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    sent = [c.get("objective") for c in canned.calls]
    assert sent == ["two_qubit", "depth"] * 4                # both objectives sent explicitly
    inc = [r for r in rows if r["kind"] == "incumbent"]
    assert len(inc) == 2 * 2 * 2 * 2 and all(r["t_client_s"] >= 0 for r in inc)
    four = ck.results(rows, arm="4colors")
    assert {r["coupler_set"] for r in four} == {"two_qubit", "depth"} and len(four) == 8
    assert {r["coupler_set"] for r in ck.results(rows, arm="qiskit")} == {"full", "full, by two-qubit depth"}
    plot.main([str(tmp_path), "--figs", str(tmp_path / "figs")])
    assert {p.name for p in (tmp_path / "figs").iterdir()} == {"fig_scatter.png", "fig_pairs.png", "selection.md"}
