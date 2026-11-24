import itertools
import json
import os

import pytest

import triage

POST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import postload  # noqa: E402  (blog/tests/postload.py; blog/tests is on sys.path from conftest)


def load(name):
    """Import <post dir>/<name>.py under a name unique to this post (blog/tests/postload.py), so posts with a
    run.py or plot.py of their own never collide when their tests run in one session."""
    return postload.load(POST_DIR, name)


def test_thirty_instances_with_unique_ids_that_fit_one_device():
    insts = triage.instances()
    assert len(insts) == 30 and len({i["id"] for i in insts}) == 30
    for i in insts:
        assert i["n"] <= 156
        assert all(0 <= a < b < i["n"] for a, b in i["edges"])
        assert set(i["weights"]) == set(i["edges"])
    assert triage.instances()[0]["edges"] == insts[0]["edges"]        # seeded


def test_family_shapes():
    e, _ = triage.regular3(16, seed=1)
    deg = [sum(v in ed for ed in e) for v in range(16)]
    assert set(deg) == {3}
    e, w = triage.sk(6, seed=1)
    assert len(e) == 15 and set(w.values()) <= {-1, 1}
    e, _ = triage.grid(3, seed=1)
    assert len(e) == 12


def test_qubo_energy_is_minus_the_cut():
    e, w = triage.sk(7, seed=3)
    m = triage.qubo_model(7, e, w)
    for x in itertools.product((0, 1), repeat=7):
        assert m.energy(list(x)) == pytest.approx(-triage.cut(e, w, x))


def test_qaoa_circuit_counts():
    e, w = triage.grid(3, seed=2)
    qc = triage.qaoa_circuit(9, e, w, p=2, measure=False)
    assert qc.count_ops()["rzz"] == 2 * len(e) and qc.num_qubits == 9


def test_classical_signal_from_a_hand_made_trajectory():
    traj = [[0.01, -5], [0.4, -8], [3.0, -9]]
    s = triage.classical_signal(traj, -9, budget_s=10)
    assert s["t_final_s"] == 3.0 and s["value_at_early"] == -8 and s["early_equals_final"] is False
    s = triage.classical_signal(traj[:2], -8, budget_s=10)
    assert s["t_final_s"] == 0.4 and s["early_equals_final"] is True
    assert triage.classical_signal([], None, budget_s=10)["t_final_s"] is None


def test_verdict_boxes_and_line_crossing():
    assert triage.verdict(0.01, 0.5).startswith("classically easy:")
    assert triage.verdict(0.01, 1e-4).startswith("classically easy and too noisy")
    assert triage.verdict(5.0, 0.5) == "worth a look"
    assert triage.verdict(5.0, 1e-4).startswith("hard classically")
    assert triage.verdict(None, 0.5) == "unknown"
    assert triage.verdict(5.0, 0.05, esp_line=0.01) == "worth a look"
    assert triage.crosses_line(0.2, 0.05) and not triage.crosses_line(0.05, 0.01) and not triage.crosses_line(0.2, None)


def test_check_and_plot_plumbing_with_canned_answers(canned, canned_compiler, tmp_path):
    check = load("check")
    plot = load("plot")
    check.main(["--out", str(tmp_path / "r"), "--ids", "sk_n10", "grid_4x4", "--time-limit", "1", "--budgets", "1",
                "--max-qiskit-trials", "1"])
    rows = [json.loads(l) for l in (tmp_path / "r" / "results.jsonl").read_text().splitlines()]
    solves = [r for r in rows if r["kind"] == "solve"]
    assert len(solves) == 2 and all(r["job_id"].startswith("canned-") for r in solves)
    sk = next(r for r in solves if r["instance"] == "sk_n10")
    assert sk["cut"] == 0 and sk["exact_cut"] > 0 and sk["is_optimal"] is False      # all-zeros answer: cut 0
    assert sk["t_final_s"] == pytest.approx(0.1)                                      # the canned trajectory time
    assert {r["arm"] for r in rows if r["kind"] == "result"} == {"4colors", "qiskit"}
    assert len([r for r in rows if r["kind"] == "verdict"]) == 2
    plot.main([str(tmp_path / "r"), "--figs", str(tmp_path / "figs")])
    sel = (tmp_path / "figs" / "selection.md").read_text()
    assert "sk_n10" in sel and (tmp_path / "figs" / "fig_quadrants.png").exists()


def test_selection_draws_qiskit_only_where_4colors_is_higher():
    plot = load("plot")
    joined = {"a": {"family": "f", "n": 1, "t_final_s": 0.1, "esp_4colors": 0.3, "esp_qiskit": 0.1},
              "b": {"family": "f", "n": 1, "t_final_s": 0.1, "esp_4colors": 0.1, "esp_qiskit": 0.1},
              "c": {"family": "f", "n": 1, "t_final_s": 0.1, "esp_4colors": 0.1}}
    boxes, drawn, both = plot.selection(joined)
    assert drawn == ["a"] and both == ["a", "b"] and set(boxes) == {"a", "b", "c"}
