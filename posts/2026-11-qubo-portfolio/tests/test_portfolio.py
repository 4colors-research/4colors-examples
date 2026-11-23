import itertools

import numpy as np
import pytest

import examplekit
from canned import canned_result
import frontier
import plot
import portfolio as pf


def test_synthetic_is_seeded_and_positive_definite():
    mu, s, names = pf.synthetic(20, seed=3)
    mu2, s2, _ = pf.synthetic(20, seed=3)
    assert np.array_equal(mu, mu2) and np.array_equal(s, s2) and len(names) == 20
    assert np.allclose(s, s.T) and np.linalg.eigvalsh(s).min() > 0


@pytest.mark.parametrize("q", [0.0, 1.0, 7.5])
def test_model_energy_is_the_objective_and_the_row_is_k(q):
    mu, s, _ = pf.synthetic(8, seed=1)
    m = pf.model(mu, s, 3, q)
    for bits in itertools.product((0, 1), repeat=8):
        x = list(bits)
        assert m.energy(x) == pytest.approx(pf.objective(mu, s, 3, q, x), abs=1e-12)
        assert m.feasible(x) == (sum(x) == 3)


def test_brute_force_and_swap_search_agree_with_plain_enumeration():
    mu, s, _ = pf.synthetic(10, seed=2)
    qs = [0, 2, 20]
    bf = pf.brute_force(mu, s, 4, qs)
    for q in qs:
        best = min(itertools.combinations(range(10), 4),
                   key=lambda c: pf.objective(mu, s, 4, q, [int(i in c) for i in range(10)]))
        assert bf[q][1] == list(best)
        assert pf.swap_search(mu, s, 4, q)[0] == pytest.approx(bf[q][0])


def test_frontier_moves_toward_less_risk():
    mu, s, _ = pf.synthetic(12, seed=4)
    bf = pf.brute_force(mu, s, 4, [0, 100])
    risk = {q: pf.portfolio_variance(s, [int(i in p) for i in range(12)], 4) for q, (_, p) in bf.items()}
    assert risk[100] <= risk[0]


def test_prices_csv(tmp_path):
    rng = np.random.default_rng(0)
    P = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(300, 3)), axis=0))
    lines = ["date,AAA,BBB,CCC"] + [f"2025-01-{i:03d}," + ",".join(f"{v:.4f}" for v in row) for i, row in enumerate(P)]
    lines.insert(5, "2025-bad,,1,2")                 # a broken row is skipped
    path = tmp_path / "prices.csv"
    path.write_text("\n".join(lines) + "\n")
    mu, s, names = pf.from_prices_csv(str(path))
    assert names == ["AAA", "BBB", "CCC"] and mu.shape == (3,) and s.shape == (3, 3)


def test_frontier_script_judges_and_records_each_answer(canned, tmp_path):
    # Two fixed answers written by hand: three picks (feasible), then four picks (breaks "exactly k").
    three = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    four = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    cons = {"m": 1, "feasible": True, "rows_checked": 1, "rows_violated": 0}
    canned(canned_result(three, -0.25, constraints=cons, job_id="canned-1"),
           canned_result(four, 1.5, constraints={**cons, "feasible": False, "rows_violated": 1}, job_id="canned-2"))
    frontier.main(["--out", str(tmp_path / "f"), "--n", "10", "--k", "3", "--q", "0", "4", "--seeds", "1",
                   "--time-limit", "1"])
    rows = examplekit.Results(tmp_path / "f" / "results.jsonl").rows()
    refs = {r["q"]: r for r in rows if r["kind"] == "reference"}
    ok, bad = [r for r in rows if r["kind"] == "solve"]
    assert all(r["n_rows"] == 1 for r in (ok, bad)) and refs[0]["method"] == "enumeration (exact)"
    # the script's own checks, recomputed from the 0/1 answer
    assert ok["feasible"] and ok["model_feasible"] and ok["n_picked"] == 3 and ok["picks"] == [0, 1, 2]
    assert ok["energy_python"] == pytest.approx(ok["objective"])
    assert ok["gap_abs"] == pytest.approx(ok["objective"] - refs[0]["objective"]) and ok["gap_abs"] >= -1e-12
    assert not bad["feasible"] and not bad["model_feasible"] and bad["n_picked"] == 4
    # what the service said is recorded as it said it, beside the script's numbers
    assert (ok["solver_objective"], ok["job_id"], ok["engine_version"]) == (-0.25, "canned-1", "canned-test")
    assert ok["service_constraints"]["feasible"] and not bad["service_constraints"]["feasible"]
    plot.main([str(tmp_path / "f"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "frontier_n10_k3.png").exists()


def test_frontier_script_uses_a_heuristic_reference_when_enumeration_is_too_big(canned, tmp_path):
    canned(canned_result([1, 1, 1, 1] + [0] * 8, 0.0))
    frontier.main(["--out", str(tmp_path / "h"), "--n", "12", "--k", "4", "--q", "1", "--seeds", "1",
                   "--brute-force-max", "10"])
    ref = [r for r in examplekit.Results(tmp_path / "h" / "results.jsonl").rows() if r["kind"] == "reference"][0]
    assert "heuristic" in ref["method"]
