# Sudoku, N-queens and map coloring: puzzles as constrained QUBOs

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-puzzles&utm_content=readme-top)

Hands-on companion code for the post. Everything runs through the public `fourcolors` Python client against the
4colors Optimizer for QUBO.

## What you will learn

- The one-hot pattern: a 0/1 variable for each (thing, choice), and "pick exactly one" as an exact row.
- How to write a model that is all rows and no objective (Sudoku, N-queens), and one that mixes both (map coloring:
  exact one-color rows, conflicts in the objective).
- How to decode the 0/1 answer back into a grid, a board or a map, and verify it in plain Python.
- What the service says when a puzzle has no answer at all (`reason["code"] == "no_feasible_found"`).

## The three puzzles

| Puzzle | Variables | Rows | Objective |
|---|---|---|---|
| Sudoku (the Wikipedia puzzle) | one per (cell, digit): 729, or only the digits a blank cell can still take | exactly one per cell, per (row, digit), per (column, digit), per (box, digit) | none |
| N-queens, n = 8, 16, 32 | n^2, one per square | exactly one per row and per column; at most one per diagonal | none |
| US map (48 states + DC), 3 and 4 colors | 49 x k, one per (state, color) | exactly one color per state | the number of borders whose two states share a color |

The map also runs in an "all exact" variant: every border and color is a row `x[u,c] + x[v,c] <= 1` and there is
no objective. With 4 colors that is a coloring; with 3 colors no coloring exists (Nevada and its five neighbors form an
odd wheel, and so do Kentucky and its seven): the odd wheels prove it, and the service reports `no_feasible_found`
when its time limit ends with no answer meeting every row (it does not prove infeasibility).

## Setup

```bash
pip install "fourcolors[optimizer]" matplotlib
export FOURCOLORS_API_KEY=...        # or FOURCOLORS_KEY_FILE=/path/to/key
```

## Step 1: N-queens in a dozen lines

```python
import fourcolors.qubo

n = 8
m = fourcolors.qubo.to_model({}, [0] * (n * n), n=n * n)       # no objective
for r in range(n):
    m.add_constraint({r * n + c: 1 for c in range(n)}, lo=1, hi=1)      # one queen per row
for c in range(n):
    m.add_constraint({r * n + c: 1 for r in range(n)}, lo=1, hi=1)      # one per column
# ... and at most one per diagonal: m.add_constraint({...: 1}, hi=1)   (see puzzles.nqueens_model)

r = fourcolors.qubo.solve(m, time_limit_s=1, seed=1)
queens = [(i, j) for i in range(n) for j in range(n) if r.solution[i * n + j]]
```

## Step 2: solve all three

```bash
python solve_puzzles.py --out runs/puzzles --time-limit 5 --seeds 1 2 3
```

Each answer is decoded and checked in plain Python (`puzzles.sudoku_valid`, `puzzles.nqueens_valid`,
`puzzles.coloring_conflicts`), and beside that the script stores the client's check (`m.feasible`) and the service's
(`r.constraints`). Lines read like:

```
sudoku eliminated seed=1 valid=[True/False]
32-queens seed=1 valid=[True/False]
US map k=3 conflicts_in_objective seed=1 conflicts=[N] reason=None
US map k=3 all_exact              seed=1 conflicts=None reason=no_feasible_found
```

The script also writes reference rows: the Sudoku solved by plain backtracking, and the fewest conflicts possible on
the US map with k colors, found by an exact branch and bound in plain Python ([N] with 3 colors, 0 with 4).

## Step 3: pictures

```bash
python plot.py runs/puzzles --out figures
```

`usmap_k3.png` and `usmap_k4.png` (conflicting borders drawn thick), `nqueens_8.png`, `nqueens_16.png`,
`nqueens_32.png`, `sudoku.png`.

## A modeling note: fewer variables first

`puzzles.sudoku_model(grid, eliminate=True)` drops every digit a clue already rules out, the first step of any
Sudoku solver: [N] variables instead of 729, and the clues need no rows. The script solves both versions.

## Run the tests (no API key needed)

```bash
python -m pytest -q tests
```

The tests check that the rows admit exactly the known solutions on small cases (4-queens has 2, 5-queens has 10; a
small 4x4 Sudoku has one), that the coloring objective counts conflicts on every assignment of a small graph, and run
the script end to end with an offline stand-in for the service.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

- Without a key: put `FOURCOLORS_OFFLINE=1` in front of any command above. A small local stand-in answers instead of the service, so the scripts run end to end; its answers are not 4colors results.
- With a key: get one at [account.4colors-research.com](https://account.4colors-research.com/?utm_source=github&utm_medium=examples&utm_campaign=qubo-puzzles&utm_content=readme-key), then `export FOURCOLORS_API_KEY=...` or `FOURCOLORS_KEY_FILE=/path/to/file`.
