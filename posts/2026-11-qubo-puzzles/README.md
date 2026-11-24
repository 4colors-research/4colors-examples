# Sudoku, N-queens and map coloring: puzzles as constrained QUBOs

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-puzzles&utm_content=readme-top)

## What you will learn

- The one-hot pattern: a 0/1 variable for each (thing, choice), and "pick exactly one" as an exact row.
- How to write a model that is all rows and no objective (Sudoku, N-queens), and one that mixes both (map coloring:
  exact one-color rows, conflicts in the objective).
- How to decode the 0/1 answer back into a grid, a board or a map, and verify it in plain Python.
- What the service says when a puzzle has no answer at all (`reason["code"] == "no_feasible_found"`).

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-puzzles&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-puzzles&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

## Step 1: N-queens in a dozen lines

An objective of all zeros is allowed: the model is pure rows, and any answer that meets them all is a solution.

## Step 2: solve all three

```bash
python solve_puzzles.py --out runs/puzzles --time-limit 5 --seeds 1 2 3
```

Each answer is decoded and checked in plain Python (`puzzles.sudoku_valid`, `puzzles.nqueens_valid`,
`puzzles.coloring_conflicts`), and beside that the script stores the client's check (`m.feasible`) and the service's
(`r.constraints`). Every constrained answer is checked by the service against its rows; the script checks it a second
and third time. Lines read like:

The script also writes reference rows: the Sudoku solved by plain backtracking, and the fewest conflicts possible on
the US map with k colors, found by an exact branch and bound in plain Python (2 with 3 colors, 0 with 4).

## Step 3: pictures

```bash
python plot.py runs/puzzles --out figures
```

This writes `usmap_k3.png` and `usmap_k4.png` (conflicting borders drawn thick), `nqueens_8.png`, `nqueens_16.png`,
`nqueens_32.png` and `sudoku.png`.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that the rows admit exactly the known solutions on small cases (4-queens has 2, 5-queens has 10; a
small 4x4 Sudoku has one), that the coloring objective counts conflicts on every assignment of a small graph, and run
the script's plumbing (rows, checks, plots) on fixed hand-made answers, and the smoke-check command.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
