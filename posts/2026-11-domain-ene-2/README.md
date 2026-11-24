# Split a power grid for parallel simulation: balanced partitioning with exact rows

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=domain-ene-2&utm_content=readme-top)

## What you will learn

- How to read a MATPOWER case (the format of the PGLib-OPF benchmark library) into a graph.
- How a two-way split becomes a QUBO with one exact balance row, and a k-way split a QUBO with one-hot rows plus one
  balance row per part.
- How to check each answer in plain Python (cut and balance) and against a proven optimum from a MILP on small grids.
- Why an exact row stays sparse when a penalty does not.

## The model

Buses are nodes, in-service branches are edges (parallel branches merge, their weights add). For a two-way split,
x_i = 1 puts bus i in part 1:

```
minimize  sum over branches (i, j) of  w_ij (x_i + x_j - 2 x_i x_j)        (the cut)
subject to  lo <= sum_i c_i x_i <= hi                                       (one balance row)
```

c_i is the work of bus i: 1 per bus by default, or 1 plus its number of branches (`--work degree`), a rough proxy for
the cost of a bus in a sparse solve. lo and hi allow 3% imbalance. For k parts there is one variable per (bus, part),
an exact "one part per bus" row per bus, and a balance row per part. In code:

```python
import fourcolors.qubo
import grid

g = grid.to_graph(grid.parse_matpower(open("data/pglib_opf_case1354_pegase.m").read()))
c = grid.work(g, "unit")
m = grid.bisection_model(g, c, imbalance=0.03)        # or grid.kway_model(g, c, parts=4, imbalance=0.03)
r = fourcolors.qubo.solve(m, time_limit_s=1, seed=1)
part = grid.decode(r.solution, g["n"], 2)
print(grid.cut_weight(g, part), grid.balanced(part, c, 2, 0.03))
```

## Data

The grids are from PGLib-OPF (Babaeinejadsarookolaee et al., arXiv 1908.02788), CC BY 4.0, which you download
yourself; this folder ships no grid files and the scripts download nothing. The run uses six cases:
`pglib_opf_case118_ieee`, `case300_ieee`, `case1354_pegase`, `case2869_pegase`, `case9241_pegase` and
`case13659_pegase`. Only the bus and branch tables are read; electrical data other than the branch status (and
reactance, for `--weight susceptance`) are ignored.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=domain-ene-2&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=domain-ene-2&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" matplotlib networkx scipy
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

3. Download the six `.m` files from the PGLib-OPF repository into `data/`.

What this post bills if you run Step 2 as shown: 6 grids x 2 part counts x 3 seeds x (1 s + 5 s), at most
216 search-seconds.

## Step 1: one grid

The snippet above splits the 1,354-bus grid in two. Every constrained answer is checked by the service against its
rows; check it again with `grid.balanced` and recompute the cut with `grid.cut_weight`, both in plain Python.

## Step 2: all six grids, two and four parts

```bash
python run.py --out runs/grid --case data/pglib_opf_case*.m --parts 2 4 --imbalance 0.03 \
  --budgets 1 5 --seeds 1 2 3 --milp-max-buses 400
```

For the two smallest grids the script also solves a MILP (SciPy's interface to the HiGHS solver) with a time limit
and records its status. Only a row with status `optimal` is a proven optimum; the others are left out of the gap chart.

## Step 3: pictures

```bash
python plot.py runs/grid runs/grid-sa --out figures --draw pglib_opf_case1354_pegase:4 \
  --case-file data/pglib_opf_case1354_pegase.m
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests read a hand-made six-bus case (parallel and out-of-service branches included), check that the QUBO energy
equals the cut and the rows equal the balance on every assignment (two-way) and every one-hot assignment (three-way),
that the MILP reference finds the single tie branch, that the annealer's penalty model equals cut plus penalty, and
they run the scripts' plumbing on fixed hand-made answers.

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
