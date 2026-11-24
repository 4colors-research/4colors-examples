# Six million spins, one upload: a 3D spin glass on the standard plan

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-huge-spin-glass&utm_content=readme-top)

## What you will learn

- How to generate an L x L x L spin glass in slabs, with couplings from a counter-based hash, so the instance does
  not depend on slab size or order and never sits in memory whole.
- How to stream it to a zstd-compressed 4CQB file and upload it once with `fourcolors.qubo.upload(path)`.
- How to solve an uploaded model by id (`input_id=`) at several budgets, and what each row records.
- How to re-score a six-million-spin answer slab by slab against the published ground-state density.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-huge-spin-glass&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-huge-spin-glass&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" numpy scipy matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

`fourcolors[optimizer]` brings zstandard (the instance file is one zstd frame) and blake3. With blake3, the
upload's id is computed on your machine, so a model the API already holds is not sent again.

## Step 1: a first try (27,000 spins, 1 s)

```bash
python run.py --out results/L30 --L 30 --budgets 1
```

## Step 2: see the sizes before building anything

```bash
python run.py --out results/L182 --L 182 --plan-only
```

It prints the spins, the couplings, the file size, the generator's memory (slab by slab, well under 1 GB) and the
service memory to expect. Nothing is built or sent. The job's own memory check decides when it starts, and a
refused job is billed nothing.

## Step 3: generate, upload once, solve by id

```bash
python run.py --out results/L182 --L 182 --seed 1 --budgets 30 60 300 --solver-seeds 0 1 2
```

The core of it is three calls:

```python
info = spinglass.write_instance("ea3d-L182.4cqb.zst", L=182, seed=1)   # streamed slab by slab, one zstd frame
model_id = fourcolors.qubo.upload("ea3d-L182.4cqb.zst")                # streamed from disk, sent once
result = fourcolors.qubo.solve(input_id=model_id, time_limit_s=60)    # by id: nothing is re-sent
```

The script deletes the file after the upload (`--keep-file` keeps it) and runs one job per budget and solver seed.
To solve an uploaded model again later, pass its id: `python run.py --out ... --L 182 --input-id <id> --budgets 600`.
For the largest sizes, `--upload-timeout 1800` gives the API longer to check the upload.

## Step 4: re-score and plot

Each returned answer is re-evaluated by `spinglass.score`, slab by slab, from the spins themselves. The row says
whether our energy equals the service's objective (`agrees`).

```bash
python plot.py results/*/results.jsonl --out figures
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the generator, writer and scorer; they do not call the service
```

The tests check the lattice and its couplings, that slab size does not change the instance, that the streamed 4CQB
file is byte-identical to the client's own writer, the scorer against a direct energy, the size report, and the
script's rows and the plot on a fixed canned answer.

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
