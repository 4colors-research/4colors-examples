import itertools

import numpy as np
import pytest

import examplekit
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


def test_frontier_script_end_to_end(fake, tmp_path):
    frontier.main(["--out", str(tmp_path / "f"), "--n", "10", "--k", "3", "--q", "0", "4", "--seeds", "1",
                   "--time-limit", "1"])
    rows = examplekit.Results(tmp_path / "f" / "results.jsonl").rows()
    sol = [r for r in rows if r["kind"] == "solve"]
    assert len(sol) == 2 and all(r["n_rows"] == 1 for r in sol)
    for r in sol:
        # the stand-in enumerates, so it returns the exact optimum; every check agrees
        assert r["feasible"] and r["model_feasible"] and r["n_picked"] == 3
        assert r["gap_abs"] == pytest.approx(0, abs=1e-12)
        assert r["energy_python"] == pytest.approx(r["solver_objective"]) == pytest.approx(r["objective"])
        assert r["service_constraints"]["feasible"]
    plot.main([str(tmp_path / "f"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "frontier_n10_k3.png").exists()


def test_frontier_script_uses_a_heuristic_reference_when_enumeration_is_too_big(fake, tmp_path):
    frontier.main(["--out", str(tmp_path / "h"), "--n", "12", "--k", "4", "--q", "1", "--seeds", "1",
                   "--brute-force-max", "10"])
    ref = [r for r in examplekit.Results(tmp_path / "h" / "results.jsonl").rows() if r["kind"] == "reference"][0]
    assert "heuristic" in ref["method"]
