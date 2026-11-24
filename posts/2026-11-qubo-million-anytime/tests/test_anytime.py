import itertools
import json

import numpy as np
import pytest

import analyze
import families as F
import plot
import rivals as R
import run


def brute_min(inst):
    best = None
    for bits in itertools.product((-1, 1), repeat=inst["n"]):
        e = F.I.energy(inst, np.array(bits))
        best = e if best is None else min(best, e)
    return best


def test_torus2d_is_a_periodic_pm1_square_lattice():
    i, j, c = F.torus2d(4, seed=3)
    assert len(i) == 2 * 16 and np.all(i < j) and set(np.unique(c)) <= {-1, 1}
    deg = np.bincount(np.concatenate([i, j]), minlength=16)
    assert np.all(deg == 4)
    i2, j2, c2 = F.torus2d(4, seed=3)
    assert np.array_equal(c, c2)


def test_random_cubic_is_simple_and_3_regular():
    u, v = F.random_cubic(1000, seed=5)
    assert len(u) == 1500 and np.all(u < v)
    assert len(np.unique(u * 1000 + v)) == 1500
    assert np.all(np.bincount(np.concatenate([u, v]), minlength=1000) == 3)


def test_build_references_and_fingerprint():
    e = F.build("ea3d:L=4,seed=1")
    assert e["reference"] == pytest.approx(-1.7863 * 64) and e["reference_alt"] == pytest.approx(-1.7876 * 64)
    t = F.build("torus2d:L=5,seed=1")
    assert t["reference"] == pytest.approx(-1.4015 * 25)
    r = F.build("regular:n=20,seed=1")
    assert r["reference"] is None and r["kind"] == "maxcut"
    assert F.build("torus2d:L=5,seed=1")["sha256"] == t["sha256"] != F.build("torus2d:L=5,seed=2")["sha256"]


def test_score_gap_signs():
    t = F.build("torus2d:L=3,seed=1")
    s = np.ones(9, dtype=np.int64)
    sc = F.score(t, s)
    assert sc["energy"] == int(np.sum(t["c"]))
    assert sc["gap"] == pytest.approx((sc["energy"] - t["reference"]) / abs(t["reference"]))
    with pytest.raises(ValueError):
        F.score(t, np.zeros(9))


def test_bsb_finds_the_ground_state_of_a_tiny_glass():
    t = F.build("torus2d:L=3,seed=4")
    J = R.sb_matrix(t)
    assert np.allclose((J - J.T).toarray(), 0)
    # H(s) = sum c s_i s_j = -1/2 s'Js on every assignment
    for bits in itertools.islice(itertools.product((-1, 1), repeat=9), 0, 512, 37):
        s = np.array(bits)
        assert F.I.energy(t, s) == pytest.approx(-0.5 * s @ (J @ s))
    best = min(F.I.energy(t, R.bsb(J, 400, seed=k)) for k in range(20))
    assert best == brute_min(t)


def test_api_model_energy_matches_the_evaluator():
    t = F.build("torus2d:L=3,seed=2")
    m = run.api_model(t)
    for bits in itertools.islice(itertools.product((-1, 1), repeat=9), 0, 512, 51):
        assert m.energy(list(bits)) == F.I.energy(t, np.array(bits))


def test_run_api_plumbing_with_canned_answer(canned, tmp_path):
    canned.solution = [1] * 16
    rows = run.main(["--out", str(tmp_path), "--instance", "torus2d:L=4,seed=1", "--arms", "api",
                     "--budgets", "1", "3", "--seeds", "1", "2", "--inline"])
    solve = [r for r in rows if r["kind"] == "solve"]
    assert len(solve) == 4 and {r["budget_s"] for r in solve} == {1.0, 3.0}
    assert all(r["agrees"] and r["engine_version"] == "canned" and r["gap"] is not None for r in solve)
    assert all(r["client_first_answer_s"] is not None for r in solve)
    assert [c["time_limit_s"] for c in canned.calls] == [1.0, 1.0, 3.0, 3.0]


def test_rival_errors_are_recorded_not_dropped(tmp_path):
    rows = run.main(["--out", str(tmp_path), "--instance", "torus2d:L=3,seed=1", "--arms", "mqlib-hh",
                     "--budgets", "1", "--seeds", "1", "--mqlib", ""])
    assert [r["kind"] for r in rows] == ["instance", "error"] and "MQLib" in rows[1]["error"]


def _rows():
    rows = []
    for inst in ("a", "b"):
        for seed in range(1, 11):
            for T, ours, riv in ((1, 0.03, 0.05), (3, 0.02, 0.015), (10, 0.012, 0.0122), (30, 0.009, 0.0128)):
                rows.append(dict(kind="solve", family="ea3d", instance=inst, budget_s=T, seed=seed, setting="api",
                                 gap=ours + 1e-5 * seed, reference=-1.0, client_first_answer_s=1.5, objective=0))
                rows.append(dict(kind="solve", family="ea3d", instance=inst, budget_s=T, seed=seed,
                                 setting="neal-fill", gap=riv + 1e-5 * seed, reference=-1.0, clock_s=T * 1.2,
                                 overran=True, objective=0))
            for T, riv in ((60, 0.0125), (120, 0.0119)):
                rows.append(dict(kind="solve", family="ea3d", instance=inst, budget_s=T, seed=seed,
                                 setting="neal-fill", gap=riv, reference=-1.0, clock_s=T, objective=0))
    return rows


def test_selection_rule_shows_wins_only():
    t = {(x["family"], x["budget_s"]): x for x in analyze.table(_rows())}
    assert t[("ea3d", 1.0)]["win"] and t[("ea3d", 1.0)]["separated"]
    assert not t[("ea3d", 10.0)]["win"]          # inside the tie band
    assert not t[("ea3d", 3.0)]["win"]           # a loss is never shown
    assert t[("ea3d", 30.0)]["win"]
    assert t[("ea3d", 1.0)]["rival_overran_runs"] == 20


def test_time_to_target_is_a_bracket():
    tt = [x for x in analyze.ttt(_rows()) if x["setting"] == "neal-fill"][0]
    assert tt["factor_low"] == 6.0 and tt["factor_high"] == 12.0 and tt["never_reached"] == 0


def test_best_found_rescoring_for_max_cut():
    rows = [dict(kind="solve", family="regular", instance="r", reference=None, objective=v, budget_s=1, seed=k,
                 setting=s) for k, (v, s) in enumerate(((100, "api"), (98, "tabu"), (101, "tabu")))]
    best = analyze.rescore_best_found(rows)
    assert best == {"r": 101} and rows[0]["gap"] == pytest.approx(1 / 101) and rows[2]["gap"] == 0


def test_analyze_and_plot_end_to_end(tmp_path):
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _rows()))
    rep = analyze.main([str(p), "--out", str(tmp_path / "an")])
    assert rep["shown"] == [("ea3d", 1.0), ("ea3d", 30.0)]
    made = plot.main([str(tmp_path / "an" / "analysis.json"), "--out", str(tmp_path / "fig")])
    assert any("gap_by_budget_ea3d" in m for m in made) and any("time_to_target" in m for m in made)
