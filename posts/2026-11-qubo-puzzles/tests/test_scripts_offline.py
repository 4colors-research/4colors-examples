import examplekit
import plot
import solve_puzzles

from test_puzzles import SMALL_SUDOKU


def test_sudoku_and_queens_end_to_end(fake, tmp_path):
    fake.exact_max_n = 16
    solve_puzzles.main(["--out", str(tmp_path / "p"), "--puzzles", "sudoku", "nqueens", "--sudoku", SMALL_SUDOKU,
                        "--queens", "4", "--seeds", "1", "--time-limit", "1"])
    rows = examplekit.Results(tmp_path / "p" / "results.jsonl").rows()
    solves = [r for r in rows if r["kind"] == "solve"]
    eliminated = [r for r in solves if r.get("variant") == "eliminated"]
    assert eliminated[0]["valid"] and eliminated[0]["model_feasible"] and eliminated[0]["service_feasible"]
    q = [r for r in solves if r["puzzle"] == "nqueens"][0]
    assert q["valid"] and q["n_vars"] == 16
    # the full 4x4 Sudoku model (64 variables) goes to the annealing stand-in; whatever it returns is judged
    full = [r for r in solves if r.get("variant") == "full"][0]
    assert full["valid"] == (full["model_feasible"] is True)
    plot.main([str(tmp_path / "p"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "nqueens_4.png").exists() and (tmp_path / "fig" / "sudoku.png").exists()


def test_us_map_end_to_end(fake, tmp_path):
    fake.sweeps = 40
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


def test_refused_job_is_recorded(fake, tmp_path, monkeypatch):
    import fourcolors
    import fourcolors.qubo

    def refuse(*a, **k):
        raise fourcolors.errors.InvalidRequest("refused for the test", code="invalid_input")

    monkeypatch.setattr(fourcolors.qubo, "solve", refuse)
    solve_puzzles.main(["--out", str(tmp_path / "x"), "--puzzles", "nqueens", "--queens", "4", "--seeds", "1"])
    row = [r for r in examplekit.Results(tmp_path / "x" / "results.jsonl").rows() if r["kind"] == "solve"][0]
    assert row["error_code"] == "invalid_input"
