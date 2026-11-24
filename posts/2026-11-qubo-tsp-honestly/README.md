# QUBO is the wrong encoding for TSP; here is the right tool

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-tsp-honestly&utm_content=readme-top), [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=qubo-tsp-honestly&utm_content=readme-top)

## What you will learn

- The textbook TSP QUBO: one 0/1 variable per (city, position), and why its size grows as n^3.
- How to write "each city once" and "each position once" as exact rows instead of penalty terms, and what that does
  (every constrained answer is checked) and does not fix (the size, and the landscape).
- How to decode an answer back to a tour and check it in plain Python, with the exact optimum for small n.
- When a QUBO is the right home for routing, and when a routing tool is.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-tsp-honestly&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-tsp-honestly&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" numpy matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

## Step 1: the encoding

`x[i*n + p] = 1` when city i is visited at position p. Stack the variables as an n x n matrix: a tour is a
permutation matrix, one 1 in every row (each city once) and every column (each position once).

The tour length is the sum of `d[i][j] * x[i,p] * x[j,p+1]` over every position p and every ordered pair of
different cities i, j (position n wraps to 0). That is n^2 (n - 1) quadratic terms: each of the n(n - 1)/2
distances appears 2n times, once per position and direction.

The 2n "exactly one" rules go in one of two ways:

- `tsp.exact_model(d)`: 2n exact rows, `m.add_constraint({...: 1}, lo=1, hi=1)`. The service checks every
  constrained answer against them.
- `tsp.penalty_model(d, lam)`: `lam * (sum - 1)^2` per row and column, the textbook form. It adds another
  n^2 (n - 1) quadratic terms, and a weight lam you have to choose.

| Cities n | Distances (routing input) | QUBO variables | Quadratic terms, exact rows | Quadratic terms, penalties |
|---|---|---|---|---|
| 10 | 45 | 100 | 900 | 1,800 |
| 30 | 435 | 900 | 26,100 | 52,200 |
| 100 | 4,950 | 10,000 | 990,000 | 1,980,000 |
| 1,000 | 499,500 | 1,000,000 | about 10^9 | about 2 x 10^9 |

These come from the formula (`tsp.model_size`), which the tests check against the built model.

## Step 2: one instance by hand

## Step 3: the sweep

```bash
python solve_tsp.py --out runs/tsp --sizes 5 6 8 10 12 15 20 30 --time-limit 5 --seeds 1 2 3
python plot.py runs/tsp --out figures
```

One random instance per size. Encodings: exact rows, and penalties at lam = 0.5, 1, 2 and 5 times the longest
distance. Each returned answer is decoded; one that is not a permutation counts as "not a tour". Up to n = 12 the script also
computes the exact optimum with Held-Karp and records each tour's distance from it; above 12 it records none.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check Held-Karp against brute force, the size formula against the built model, that a tour's energy equals
its length in both encodings, that the exact rows admit permutations only, that decoding rejects non-permutations, and
run the script's plumbing (rows, figures) on fixed canned answers.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
