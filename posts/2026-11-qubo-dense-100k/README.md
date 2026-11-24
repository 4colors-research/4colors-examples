# Dense QUBOs at 100,000 variables: degree 4 to 256

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-dense-100k&utm_content=readme-top)

## What you will learn

- How to generate a large dense model in slices and write it straight to a compressed file the API takes
  (`dense.write_file`), without holding a second copy in memory.
- How to check, before a sweep, that a model this size is admitted on your plan: one recorded admission probe.
- How to upload once and solve by `input_id` at many budgets, so each extra budget costs no upload.
- How to re-score each returned answer yourself (`dense.energy`), and why the gap is measured to the best answer any solver
  found, not to a proven optimum.

## The models

| Family | Variables | Terms per variable | Couplings | What it stands for |
|---|---|---|---|---|
| random regular, +-1 couplings (Max-Cut form) | 100,000 | 4, 16, 64, 256 | 0.2M, 0.8M, 3.2M, 12.8M | sparse to dense graph models at a fixed size |
| fully dense, integer weights in [-100, 100] | 3,000, 5,000, 7,000 | n - 1 | 4.5M, 12.5M, 24.5M | a model where every pair interacts |

Each size has three fresh generator seeds. `dense.reg_instance` builds a degree-d graph as d random perfect
matchings (repeated pairs merged, so a few nodes end up just below d); `dense.full_instance` fills the upper
triangle block by block. Both are pure functions of the size and seed, and every row records the instance's sha256.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-dense-100k&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-dense-100k&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" numpy zstandard matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

## Step 1: size it before you build it

```bash
python run.py --out runs/plan --family reg --plan-only
```

```
  reg n=100000 degree=4: 200,000 couplings (400,000 matrix entries), 0.00 GB raw
  ...
  reg n=100000 degree=256: 12,800,000 couplings (25,600,000 matrix entries), 0.21 GB raw
```

## Step 2: one admission probe

```bash
python run.py --out runs/probe --family reg --degrees 256 --instance-seeds 101 --probe
```

## Step 3: upload once, solve at every budget

```python
import fourcolors.qubo
import dense

inst = dense.reg_instance(100_000, 256, seed=101)
dense.write_file(inst, "reg256.4cqb.zst")                       # streamed in slices, one zstd frame
input_id = fourcolors.qubo.upload("reg256.4cqb.zst")
for T in (1, 3, 10, 60, 300):
    r = fourcolors.qubo.solve(input_id=input_id, time_limit_s=T, seed=1)
    print(T, dense.energy(inst, r.solution), r.objective)       # our number beside the service's
```

`run.py` does this for every family, degree, instance and seed, records the first-answer time and the trajectory,
and checks that the service's objective equals the local re-score on each returned answer (`agrees`).

## Step 4: tables and pictures

```bash
python plot.py runs/dense/results.jsonl runs/dense-rivals/results.jsonl --out figures
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check both generators (sorted, no repeats, weights in range, deterministic by seed, blocks independent),
that the energy equals a plain-Python loop on each returned answer of small models, that the compressed file round-trips
through the client's reader with the same energy, the selection rule on hand-made rows (wins kept; ties, overruns
and missing answers left out), and run the scripts' plumbing on fixed canned answers.

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
