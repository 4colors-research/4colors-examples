"""Unit tests of the rebalancing model, the rolling estimates and the script plumbing (no service calls)."""
import itertools

import numpy as np

import examplekit
import market
import plot
import rebalance as rb
import run


def small(n=6, seed=3):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return rng.normal(0.05, 0.03, size=n), A @ A.T / n + np.eye(n) * 0.01


def test_model_energy_is_objective_on_every_assignment():
    mu, sigma = small()
    prev = [1, 0, 1, 0, 0, 0]
    for cost in (0.0, 0.003):
        m = rb.model(mu, sigma, 2, 1.5, prev=prev, cost=cost)
        for x in itertools.product([0, 1], repeat=6):
            want = rb.objective(mu, sigma, 2, 1.5, x, prev=prev, cost=cost)
            assert abs(m.energy(list(x)) - want) < 1e-9
            assert bool(m.feasible(list(x))) == (sum(x) == 2)


def test_changes_counts_sold_plus_bought():
    assert rb.changes([1, 1, 0, 0], [1, 0, 1, 0]) == 2
    assert rb.changes([1, 0], [1, 0]) == 0


def test_exact_best_matches_brute_force_with_and_without_cost():
    mu, sigma = small(8)
    prev = [0, 1, 0, 1, 0, 1, 0, 0]
    for cost in (0.0, 0.01):
        best, x = rb.exact_best(mu, sigma, 3, 2.0, prev=prev, cost=cost, chunk=7)
        brute = min(rb.objective(mu, sigma, 3, 2.0, c, prev=prev, cost=cost)
                    for c in itertools.product([0, 1], repeat=8) if sum(c) == 3)
        assert abs(best - brute) < 1e-12 and sum(x) == 3
        assert abs(rb.objective(mu, sigma, 3, 2.0, x, prev=prev, cost=cost) - best) < 1e-12


def test_rolling_estimate_is_the_window_mean_and_cov():
    r, names = market.simulate(5, 40, seed=1)
    assert r.shape == (40, 5) and len(names) == 5
    mu, sigma = market.estimate(r, 30, 20)
    assert np.allclose(mu, r[10:30].mean(axis=0) * 252)
    assert np.allclose(sigma, np.cov(r[10:30], rowvar=False) * 252)
    days = list(market.mornings(r, 20, 3))
    assert [d for d, _, _ in days] == [0, 1, 2, 3]
    assert np.allclose(days[-1][1], r[20:40].mean(axis=0) * 252)     # last morning uses the last 20 days


def test_prices_csv_roundtrip(tmp_path):
    p = tmp_path / "px.csv"
    p.write_text("date,A,B\n2026-01-01,100,50\n2026-01-02,110,50\n2026-01-03,x,1\n2026-01-04,121,25\n")
    r, names = market.returns_from_prices_csv(str(p))
    assert names == ["A", "B"] and r.shape == (2, 2)
    assert np.allclose(r[:, 0], np.log(1.1))


def test_run_end_to_end_warm_gets_yesterday(canned, tmp_path):
    k = 2
    canned.solution = lambda call: call["warm_start"] or [1] * k + [0] * (call["n"] - k)
    run.main(["--out", str(tmp_path / "r"), "--n", "6", "--k", str(k), "--days", "2", "--window", "20",
              "--history", "40", "--seeds", "1", "2", "--time-limit", "1", "--day0-time-limit", "1"])
    rows = examplekit.Results(tmp_path / "r" / "results.jsonl").rows()
    solves = [r for r in rows if r["kind"] == "solve"]
    assert len(solves) == 1 + 2 * 2 * 2
    assert len([r for r in rows if r["kind"] == "carry_over"]) == 2
    assert len([r for r in rows if r["kind"] == "reference"]) == 3          # n = 6 <= exact-max
    for r in solves:
        assert r["model_feasible"] and r["service_feasible"]
        assert abs(r["objective_python"] - r["solver_objective"]) < 1e-9
    warm = [c for c in canned.calls if c["warm_start"] is not None]
    assert len(warm) == 4 and all(sum(c["warm_start"]) == k for c in warm)
    held = [r for r in rows if r["kind"] == "held"]
    assert held[0]["changes"] is None and all(h["changes"] == 0 for h in held[1:])
    plot.main([str(tmp_path / "r"), "--out", str(tmp_path / "fig")])
    for f in ("rebalance_time_to_quality.png", "rebalance_carry_over.png", "rebalance_turnover.png"):
        assert (tmp_path / "fig" / f).exists()


def test_infeasible_answer_is_not_held(canned, tmp_path):
    # day 0 canned answer meets the row; later mornings return all ones (breaks sum x = k): yesterday is kept
    canned.solution = lambda call: [1, 1, 0, 0, 0, 0] if call["warm_start"] is None and len(canned.calls) == 1 \
        else [1] * call["n"]
    run.main(["--out", str(tmp_path / "r"), "--n", "6", "--k", "2", "--days", "1", "--window", "20",
              "--history", "40", "--seeds", "1", "--time-limit", "1", "--day0-time-limit", "1"])
    rows = examplekit.Results(tmp_path / "r" / "results.jsonl").rows()
    bad = [r for r in rows if r["kind"] == "solve" and r["day"] == 1]
    assert bad and all(r["model_feasible"] is False for r in bad)
    assert [r["day"] for r in rows if r["kind"] == "held"] == [0]


def test_time_to_quality_uses_exact_reference():
    rows = [{"kind": "reference", "n": 6, "day": 1, "objective_python": -0.05},
            {"kind": "solve", "n": 6, "day": 1, "arm": "cold", "model_feasible": True, "objective_python": -0.05,
             "trajectory": [[0.2, -0.049], [0.8, -0.05]]},
            {"kind": "solve", "n": 6, "day": 1, "arm": "warm", "model_feasible": True, "objective_python": -0.05,
             "trajectory": [[0.0, -0.05]]}]
    assert plot.times_to_quality(rows, 0.0) == {"cold": [0.8], "warm": [0.0]}
    assert plot.times_to_quality(rows, 5 * plot.BP) == {"cold": [0.8], "warm": [0.0]}     # -0.049 is 10 bp off
    assert plot.times_to_quality(rows, 10 * plot.BP) == {"cold": [0.2], "warm": [0.0]}
