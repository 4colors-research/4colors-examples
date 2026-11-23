"""The scripts end to end on fixed canned answers (tests/canned.py): decode, verify and record, no service."""
import examplekit
import plot
import puzzles as pz
import solve_puzzles
from canned import canned_result

from test_puzzles import SMALL_SUDOKU

SMALL_SUDOKU_SOLUTION = "1234341221434321"      # written by hand; the test checks it is the reference answer
FOUR_QUEENS = [1, 3, 0, 2]                     # column of the queen in each row


def test_sudoku_and_queens_end_to_end(canned, tmp_path):
    grid = pz.parse_grid(SMALL_SUDOKU)
    answer = pz.parse_grid(SMALL_SUDOKU_SOLUTION)
    assert answer == pz.sudoku_backtrack(grid)
    m_elim, variables = pz.sudoku_model(grid, eliminate=True)
    m_full, _ = pz.sudoku_model(grid, eliminate=False)
    right = [1 if answer[r][c] == d else 0 for r, c, d in variables]
    queens = [1 if c == FOUR_QUEENS[r] else 0 for r in range(4) for c in range(4)]
    canned(canned_result(right, 0, constraints={"feasible": True}),            # sudoku, eliminated
           canned_result([0] * m_full.n, 0, constraints={"feasible": False}),  # sudoku, full: an empty grid
           canned_result(queens, 0, constraints={"feasible": True}))           # 4-queens
    solve_puzzles.main(["--out", str(tmp_path / "p"), "--puzzles", "sudoku", "nqueens", "--sudoku", SMALL_SUDOKU,
                        "--queens", "4", "--seeds", "1", "--time-limit", "1"])
    rows = examplekit.Results(tmp_path / "p" / "results.jsonl").rows()
    solves = [r for r in rows if r["kind"] == "solve"]
    elim = [r for r in solves if r.get("variant") == "eliminated"][0]
    assert elim["valid"] and elim["model_feasible"] and elim["service_feasible"] and elim["answer"] == answer
    full = [r for r in solves if r.get("variant") == "full"][0]
    assert full["n_vars"] == 64 and not full["valid"] and full["model_feasible"] is False
    q = [r for r in solves if r["puzzle"] == "nqueens"][0]
    assert q["valid"] and q["n_vars"] == 16 and q["model_feasible"]
    plot.main([str(tmp_path / "p"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "nqueens_4.png").exists() and (tmp_path / "fig" / "sudoku.png").exists()


def test_us_map_end_to_end(canned, tmp_path):
    nodes, _ = pz.us_edges()
    every_state_color_0 = [1, 0, 0, 0] * len(nodes)
    canned(canned_result(every_state_color_0, 107),                                   # conflicts in the objective
           canned_result(None, None, reason={"code": "no_feasible_found"}))           # all exact
    solve_puzzles.main(["--out", str(tmp_path / "u"), "--puzzles", "usmap", "--colors", "4", "--seeds", "1",
                        "--time-limit", "1"])
    rows = examplekit.Results(tmp_path / "u" / "results.jsonl").rows()
    ref = [r for r in rows if r["kind"] == "reference"][0]
    assert ref["min_conflicts"] == 0 and ref["n_edges"] == 107
    obj, exact = [r for r in rows if r["kind"] == "solve"]
    assert obj["n_vars"] == exact["n_vars"] == 196
    assert obj["every_state_colored"] and obj["model_feasible"]
    assert obj["n_conflicts"] == 107 == obj["energy_python"]          # the objective counts conflicts
    assert exact["answer"] is None and exact["reason"] == "no_feasible_found"
    plot.main([str(tmp_path / "u"), "--out", str(tmp_path / "fig")])


def test_refused_job_is_recorded(canned, tmp_path, monkeypatch):
    import fourcolors
    import fourcolors.qubo

    def refuse(*a, **k):
        raise fourcolors.errors.InvalidRequest("refused for the test", code="invalid_input")

    canned()
    monkeypatch.setattr(fourcolors.qubo, "solve", refuse)
    solve_puzzles.main(["--out", str(tmp_path / "x"), "--puzzles", "nqueens", "--queens", "4", "--seeds", "1"])
    row = [r for r in examplekit.Results(tmp_path / "x" / "results.jsonl").rows() if r["kind"] == "solve"][0]
    assert row["error_code"] == "invalid_input"
