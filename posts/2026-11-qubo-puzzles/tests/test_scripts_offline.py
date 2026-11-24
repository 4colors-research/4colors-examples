import pytest
import examplekit
import plot
import solve_puzzles

from test_puzzles import SMALL_SUDOKU


def test_sudoku_and_queens_end_to_end(canned, tmp_path):
    four_queens = [0] * 16
    for r, c in [(0, 1), (1, 3), (2, 0), (3, 2)]:          # a hand-made 4-queens answer
        four_queens[r * 4 + c] = 1
    import puzzles as pz

    solved = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]   # SMALL_SUDOKU's solution, by hand
    _, elim_vars = pz.sudoku_model(pz.parse_grid(SMALL_SUDOKU), eliminate=True)
    sudoku_answer = [int(solved[r][c] == d) for r, c, d in elim_vars]
    canned_by_n = {16: four_queens, len(elim_vars): sudoku_answer}
    canned.solution = lambda call: canned_by_n.get(call["n"], [0] * call["n"])
    solve_puzzles.main(["--out", str(tmp_path / "p"), "--puzzles", "sudoku", "nqueens", "--sudoku", SMALL_SUDOKU,
                        "--queens", "4", "--seeds", "1", "--time-limit", "1"])
    rows = examplekit.Results(tmp_path / "p" / "results.jsonl").rows()
    solves = [r for r in rows if r["kind"] == "solve"]
    eliminated = [r for r in solves if r.get("variant") == "eliminated"]
    assert eliminated[0]["valid"] and eliminated[0]["model_feasible"] and eliminated[0]["service_feasible"]
    q = [r for r in solves if r["puzzle"] == "nqueens"][0]
    assert q["valid"] and q["model_feasible"] and q["n_vars"] == 16
    full = [r for r in solves if r.get("variant") == "full"][0]      # canned all zeros: rejected by both checks
    assert full["valid"] is False and full["model_feasible"] is False
    plot.main([str(tmp_path / "p"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "nqueens_4.png").exists() and (tmp_path / "fig" / "sudoku.png").exists()


def test_us_map_end_to_end(canned, tmp_path):
    solve_puzzles.main(["--out", str(tmp_path / "u"), "--puzzles", "usmap", "--colors", "4", "--seeds", "1",
                        "--time-limit", "1"])
    rows = examplekit.Results(tmp_path / "u" / "results.jsonl").rows()
    ref = [r for r in rows if r["kind"] == "reference"][0]
    assert ref["min_conflicts"] == 0 and ref["n_edges"] == 107
    for r in (r for r in rows if r["kind"] == "solve"):
        assert r["n_vars"] == 196
        if r["answer"] is not None:
            assert r["every_state_colored"] == r["model_feasible"]
            if r["variant"] == "conflicts_in_objective" and r["model_feasible"]:
                assert r["n_conflicts"] == r["energy_python"]      # the objective counts conflicts
            if r["variant"] == "all_exact" and r["model_feasible"]:
                assert r["n_conflicts"] == 0                       # edge rows forbid conflicts
        else:
            assert r["reason"] == "no_feasible_found"
    plot.main([str(tmp_path / "u"), "--out", str(tmp_path / "fig")])


def test_refused_job_is_recorded(canned, tmp_path, monkeypatch):
    import fourcolors
    import fourcolors.qubo

    def refuse(*a, **k):
        raise fourcolors.errors.InvalidRequest("refused for the test", code="invalid_input")

    monkeypatch.setattr(fourcolors.qubo, "solve", refuse)
    solve_puzzles.main(["--out", str(tmp_path / "x"), "--puzzles", "nqueens", "--queens", "4", "--seeds", "1"])
    row = [r for r in examplekit.Results(tmp_path / "x" / "results.jsonl").rows() if r["kind"] == "solve"][0]
    assert row["error_code"] == "invalid_input"


def test_smoke_arms_check_on_us_map_variants(canned, tmp_path):
    """The EXPERIMENTS smoke check: the two US-map variants at k = 3 must differ (rows, and what the service says)."""
    arms_differ = pytest.importorskip("arms_differ")   # the internal arms check; not published

    solve_puzzles.main(["--out", str(tmp_path / "s"), "--puzzles", "usmap", "--colors", "3", "--seeds", "1",
                        "--time-limit", "1"])
    res = str(tmp_path / "s" / "results.jsonl")
    assert arms_differ.main([res, "--arm-field", "variant", "--arms", "conflicts_in_objective", "all_exact",
                             "--key", "colors", "seed", "--compare", "n_rows", "reason", "service_feasible",
                             "--where", "kind=solve", "puzzle=usmap"]) == 0
