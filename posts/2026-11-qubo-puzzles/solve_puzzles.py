"""Solve the puzzles with exact rows, decode each answer and verify it in plain Python.

    python solve_puzzles.py --out runs/puzzles --time-limit 5 --seeds 1 2 3
    python solve_puzzles.py --out runs/puzzles --puzzles nqueens --queens 8 16 32

Puzzles: `sudoku` (the Wikipedia puzzle, or --sudoku "<81 digits, 0 = blank>"), `nqueens` (one model per --queens
size), `usmap` (the 48 contiguous states + DC with --colors 3 4, conflicts in the objective, plus the all-exact
variant where every edge is a row).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors
import fourcolors.qubo

import examplekit
import puzzles as pz


def solve(m, a, seed, client):
    """The result, or None with the error when the service refuses the job (recorded, not hidden)."""
    try:
        return fourcolors.qubo.solve(m, time_limit_s=a.time_limit, seed=seed, client=client), None
    except fourcolors.errors.FourColorsError as exc:
        return None, {"error_code": getattr(exc, "code", None), "error": str(exc)}


def checks(m, r):
    x = r.solution
    return {"model_feasible": None if x is None else m.feasible(x),
            "service_feasible": (r.constraints or {}).get("feasible"),
            "energy_python": None if x is None else m.energy(x)}


def main(argv=None):
    p = examplekit.base_parser(__doc__)
    p.add_argument("--puzzles", nargs="+", default=["sudoku", "nqueens", "usmap"],
                   choices=["sudoku", "nqueens", "usmap"])
    p.add_argument("--sudoku", default=pz.WIKIPEDIA_PUZZLE, help="the grid, row by row, 0 or . for a blank")
    p.add_argument("--queens", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--colors", type=int, nargs="+", default=[3, 4])
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()

    if "sudoku" in a.puzzles:
        grid = pz.parse_grid(a.sudoku)
        out.add(kind="reference", puzzle="sudoku", grid=grid, solution=pz.sudoku_backtrack(grid), method="backtracking")
        for eliminate in (True, False):
            m, variables = pz.sudoku_model(grid, eliminate=eliminate)
            for seed in a.seeds:
                r, err = solve(m, a, seed, client)
                common = dict(kind="solve", puzzle="sudoku", variant="eliminated" if eliminate else "full",
                              n_vars=m.n, n_rows=m.m, seed=seed, time_limit_s=a.time_limit)
                if r is None:
                    out.add(**common, **err)
                    continue
                filled = None if r.solution is None else pz.sudoku_decode(grid, variables, r.solution)
                row = out.add(**common, answer=filled, valid=filled is not None and pz.sudoku_valid(grid, filled),
                              **checks(m, r), **examplekit.result_fields(r))
                print(f"sudoku {row['variant']:10s} seed={seed} valid={row['valid']}")

    if "nqueens" in a.puzzles:
        for n in a.queens:
            m = pz.nqueens_model(n)
            for seed in a.seeds:
                r, err = solve(m, a, seed, client)
                common = dict(kind="solve", puzzle="nqueens", size=n, n_vars=m.n, n_rows=m.m, seed=seed,
                              time_limit_s=a.time_limit)
                if r is None:
                    out.add(**common, **err)
                    continue
                queens = None if r.solution is None else pz.nqueens_decode(n, r.solution)
                row = out.add(**common, answer=queens, valid=queens is not None and pz.nqueens_valid(n, queens),
                              **checks(m, r), **examplekit.result_fields(r))
                print(f"{n}-queens seed={seed} valid={row['valid']}")

    if "usmap" in a.puzzles:
        nodes, edges = pz.us_edges()
        for k in a.colors:
            out.add(kind="reference", puzzle="usmap", colors=k, min_conflicts=pz.min_conflicts(nodes, edges, k),
                    method="branch and bound", n_nodes=len(nodes), n_edges=len(edges))
            for all_exact in (False, True):
                m = pz.coloring_model(nodes, edges, k, all_exact=all_exact)
                for seed in a.seeds:
                    r, err = solve(m, a, seed, client)
                    common = dict(kind="solve", puzzle="usmap", colors=k,
                                  variant="all_exact" if all_exact else "conflicts_in_objective",
                                  n_vars=m.n, n_quad=m.nnz, n_rows=m.m, seed=seed, time_limit_s=a.time_limit)
                    if r is None:
                        out.add(**common, **err)
                        continue
                    colors = None if r.solution is None else pz.coloring_decode(nodes, k, r.solution)
                    conflicts = None if colors is None else pz.coloring_conflicts(edges, colors)
                    row = out.add(**common, answer=colors, conflicts=conflicts,
                                  n_conflicts=None if conflicts is None else len(conflicts),
                                  every_state_colored=colors is not None and min(colors.values()) >= 0,
                                  **checks(m, r), **examplekit.result_fields(r))
                    print(f"US map k={k} {row['variant']:22s} seed={seed} conflicts={row['n_conflicts']} "
                          f"reason={row['reason']}")


if __name__ == "__main__":
    main()
