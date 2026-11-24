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


def test_frontier_script_end_to_end(canned, tmp_path):
    canned.solution = [1, 1, 1] + [0] * 7          # a hand-made answer: the first three assets
    frontier.main(["--out", str(tmp_path / "f"), "--n", "10", "--k", "3", "--q", "0", "4", "--seeds", "1",
                   "--time-limit", "1"])
    rows = examplekit.Results(tmp_path / "f" / "results.jsonl").rows()
    sol = [r for r in rows if r["kind"] == "solve"]
    assert len(sol) == 2 and all(r["n_rows"] == 1 for r in sol)
    for r in sol:
        # the canned pick has exactly 3 assets: every check accepts it, and it is no better than the exact optimum
        assert r["feasible"] and r["model_feasible"] and r["n_picked"] == 3
        assert r["gap_abs"] >= -1e-12
        assert r["energy_python"] == pytest.approx(r["solver_objective"]) == pytest.approx(r["objective"])
        assert r["service_constraints"]["feasible"]
    plot.main([str(tmp_path / "f"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "frontier_n10_k3.png").exists()


def test_frontier_script_uses_a_heuristic_reference_when_enumeration_is_too_big(canned, tmp_path):
    frontier.main(["--out", str(tmp_path / "h"), "--n", "12", "--k", "4", "--q", "1", "--seeds", "1",
                   "--brute-force-max", "10"])
    ref = [r for r in examplekit.Results(tmp_path / "h" / "results.jsonl").rows() if r["kind"] == "reference"][0]
    assert "heuristic" in ref["method"]


# ---- optional sector cap (off by default) ------------------------------------------------------

def test_sectors_are_seeded_and_cover_every_label():
    a, b = pf.sectors(12, 4, seed=1), pf.sectors(12, 4, seed=1)
    assert a == b and sorted(set(a)) == [0, 1, 2, 3] and all(a.count(s) == 3 for s in range(4))


def test_default_model_has_no_sector_rows():
    mu, s, _ = pf.synthetic(8, seed=1)
    assert pf.model(mu, s, 3, 1.0).m == 1


@pytest.mark.parametrize("cap", [1, 2])
def test_sector_rows_admit_exactly_the_capped_portfolios(cap):
    mu, s, _ = pf.synthetic(8, seed=1)
    lab = pf.sectors(8, 3, seed=1)
    m = pf.model(mu, s, 3, 2.0, sector_of=lab, sector_cap=cap)
    assert m.m == 1 + len(set(lab))
    for bits in itertools.product((0, 1), repeat=8):
        x = list(bits)
        assert m.feasible(x) == (sum(x) == 3 and pf.sector_ok(x, lab, cap))
        assert m.energy(x) == pytest.approx(pf.objective(mu, s, 3, 2.0, x), abs=1e-12)


def test_capped_references_match_filtered_enumeration_and_respect_the_cap():
    mu, s, _ = pf.synthetic(10, seed=2)
    lab = pf.sectors(10, 3, seed=2)
    for q in [0, 2, 20]:
        ok = [c for c in itertools.combinations(range(10), 4)
              if pf.sector_ok([int(i in c) for i in range(10)], lab, 2)]
        best = min(ok, key=lambda c: pf.objective(mu, s, 4, q, [int(i in c) for i in range(10)]))
        bf = pf.brute_force(mu, s, 4, [q], sector_of=lab, sector_cap=2)[q]
        assert bf[1] == list(best)
        f, picks = pf.swap_search(mu, s, 4, q, sector_of=lab, sector_cap=2)
        assert pf.sector_ok([int(i in picks) for i in range(10)], lab, 2) and len(picks) == 4
        assert f >= bf[0] - 1e-12


def test_impossible_cap_is_refused():
    mu, s, _ = pf.synthetic(6, seed=1)
    with pytest.raises(ValueError):
        pf.swap_search(mu, s, 4, 1.0, sector_of=pf.sectors(6, 2, seed=1), sector_cap=1)


def test_frontier_script_with_sector_cap(canned, tmp_path):
    frontier.main(["--out", str(tmp_path / "c"), "--n", "10", "--k", "4", "--q", "0", "--seeds", "1",
                   "--time-limit", "1", "--sectors", "3", "--sector-cap", "2"])
    rows = examplekit.Results(tmp_path / "c" / "results.jsonl").rows()
    sol = [r for r in rows if r["kind"] == "solve"]
    assert len(sol) == 1 and sol[0]["n_rows"] == 4 and sol[0]["sector_cap"] == 2
    # canned all-zeros answer: it picks no asset, so the row check fails, and the sector check agrees with the counts
    assert sol[0]["feasible"] is False
    assert sol[0]["sector_ok"] == (max(sol[0]["sector_counts"].values() or [0]) <= 2)
    plot.main([str(tmp_path / "c"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "frontier_n10_k4_cap2.png").exists()


def test_smoke_arms_check_capped_vs_uncapped(canned, tmp_path):
    """The smoke check for the optional sector row: capped and uncapped runs must differ."""
    arms_differ = pytest.importorskip("arms_differ")   # optional helper, skipped when absent

    base = ["--n", "10", "--k", "4", "--q", "0", "--seeds", "1", "--time-limit", "1", "--sectors", "4"]
    frontier.main(["--out", str(tmp_path / "u")] + base)
    frontier.main(["--out", str(tmp_path / "c")] + base + ["--sector-cap", "1"])
    files = [str(tmp_path / d / "results.jsonl") for d in ("u", "c")]
    # the exact command the post shows: uncapped rows record sector_cap None
    assert arms_differ.main(files + ["--arm-field", "sector_cap", "--arms", "None", "1", "--key", "q", "seed",
                                     "--compare", "n_rows", "picks", "--where", "kind=solve"]) == 0


def test_brute_force_refuses_an_impossible_cap():
    mu, s, _ = pf.synthetic(6, seed=1)
    with pytest.raises(ValueError):
        pf.brute_force(mu, s, 4, [1.0], sector_of=pf.sectors(6, 2, seed=1), sector_cap=1)


def test_smoke_arms_check_q_sweep(canned, tmp_path):
    """The smoke check command runs on these rows (q is a float in the rows). The canned answer is the
    same for q = 0 and q = 32, so the check must flag the arms as identical (the harness-bug case)."""
    arms_differ = pytest.importorskip("arms_differ")   # optional helper, skipped when absent

    frontier.main(["--out", str(tmp_path / "s"), "--n", "12", "--k", "4", "--q", "0", "32", "--seeds", "1", "2",
                   "--time-limit", "1"])
    assert arms_differ.main([str(tmp_path / "s" / "results.jsonl"), "--arm-field", "q", "--arms", "0.0", "32.0",
                             "--key", "seed", "--compare", "picks", "risk", "--where", "kind=solve"]) == 1
