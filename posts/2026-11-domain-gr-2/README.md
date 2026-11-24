# The neutral-atom problem at a million atoms: maximum independent set on unit-disk graphs

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=domain-gr-2&utm_content=readme-top)

## What you will learn

- How to generate King's-lattice unit-disk graphs at any filling, in plain Python and numpy, up to a million atoms.
- MIS two ways: with one exact row `x_u + x_v <= 1` per edge (no weight to tune), and as the textbook penalty QUBO.
- A cheap certificate that works at every size: each 2 x 2 block of the lattice is a clique, so an independent set
  holds at most one atom per non-empty block.
- How to check each answer in a few lines of numpy, and what to do with a penalty answer that breaks an edge.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=domain-gr-2&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=domain-gr-2&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" numpy scipy matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

## Step 1: the lattice and the model in a dozen lines

```python
import fourcolors.qubo
import kings

L = kings.side_for_atoms(289)                     # 19
pos, edges = kings.lattice(L, filling=0.8, seed=1)
n = len(pos)

m = fourcolors.qubo.to_model({}, [1] * n, n=n, sense="max")   # maximize the number of chosen atoms
for u, v in edges:
    m.add_constraint({u: 1, v: 1}, hi=1)                        # no two chosen atoms inside the blockade radius

r = fourcolors.qubo.solve(m, time_limit_s=1, seed=1)
chosen = [i for i in range(n) if r.solution[i]]
print(len(chosen), "atoms; at most", kings.block_bound(pos))
```

Every constrained answer is checked by the service against its rows (`r.constraints["feasible"]`), and the script
checks it again in numpy.

## Step 2: the penalty form, for comparison

The form in most papers is a plain QUBO: minimize `-sum x + P * sum_edges x_u x_v`. On an unweighted graph any P > 1
has the same optimum, so the post uses P = 2. `kings.penalty_model` builds it as a scipy sparse matrix, which is how a
million atoms fit in a laptop's memory. A penalty answer can break an edge; the script counts violated edges and then
applies the usual repair (drop one end of each), and reports both.

## Step 3: run the sizes

```bash
python run.py --out runs/gr2 --atoms 289 1000 10000 100000 1000000 --time-limit 1 10 --seeds 1 2 3
python plot.py runs/gr2 --out figures
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check the King's graph on a full 3 x 3 lattice (20 edges, MIS 4), that the numpy and pure-Python
generators give the same instance, that the HiGHS reference and the block bound agree with brute force on small
lattices, that the rows admit exactly the independent sets and the penalty form has the same optimum, and run the
scripts' plumbing (rows, checks, repair, the chart's selection rule, plots) on fixed hand-made answers.

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
