# Denoise a 1440p image with one QUBO

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-4k-denoise&utm_content=readme-top)

## What you will learn

- How a binary image MRF becomes a QUBO: one variable per pixel, a linear data term, one coupling per neighbor
  pair.
- How to stream a 3.7-million-variable model to a 4CQB file in row blocks, upload it once and solve it by id.
- How to score an answer two ways: the energy gap to the exact minimum (min cut, submodular case) and the share of
  pixels that differ from the clean image.
- What changes when the prior is non-submodular, and how to report a reference that is "best found", not optimal.

## The model

For a noisy image y (each pixel flipped with probability eps) and labels x_p in {0, 1}:

E(x) = Σ_p D_p(x_p) + Σ_{p~q} w_pq [x_p ≠ x_q]

- **Data term.** D_p(l) = λ if l differs from y_p, else 0, with λ = round(10 log((1 - eps) / eps)) = 22 at
  eps = 0.10.
- **Smoothing prior.** w = β = 10 on every horizontal and vertical neighbor pair. All weights are non-negative,
  so the energy is submodular and its exact minimum is one s-t min cut. `mrf.exact_mincut` computes it with
  scipy's max-flow on a sparse graph.
- **Stripe prior** (`--variant stripes`). Inside one rectangle, horizontal pairs get w = β - γ = -10 < 0, which
  rewards neighbors that differ. The clean test image has a vertical-stripe texture there. Negative weights make
  the energy non-submodular. It is scored against the best energy found, and labeled that way.

As a QUBO, [x_p ≠ x_q] = x_p + x_q - 2 x_p x_q, so each neighbor pair contributes w to both linear terms and -2w to
their coupling. All weights are integers, so energies are exact and the service's objective can be checked to the
unit.

| image | variables | couplings | 4CQB file (raw, upper bound) | service memory (estimate) |
|---|---|---|---|---|
| 1920 x 1080 | 2,073,600 | 4,144,200 | 0.08 GB | about 2 GB |
| 2560 x 1440 | 3,686,400 | 7,368,800 | 0.15 GB | about 3.4 GB |

reports what the service used.

## Data

The test image is drawn by the script: a large "4" glyph, discs and a ring, plus the striped rectangle for the
second variant. Nothing is downloaded. To use your own picture, pass `--image file.png` (thresholded to 0/1). Use
an image you own or one in the public domain.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-4k-denoise&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-4k-denoise&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" numpy scipy pillow matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

## Step 1: a small first try

```bash
python run.py --out results/small --size 640x360 --budgets 1
```

## Step 2: check that the exact reference fits your machine

```bash
python pilot_exact.py --out results/pilot --sizes 1080p 1440p
```

This computes only the exact minimum, with no API call, and prints its wall time and peak memory. If the 1440p
max-flow is too slow or too large on your machine, use `--exact crop --crop 1024x1024` in the next step: a centered
crop becomes its own model, solved exactly and by the service, and the gap is reported for the crop. Every row says
which (`region`).

## Step 3: 1080p and 1440p, both priors

```bash
python run.py --out results/1080p --size 1080p --budgets 10 60
python run.py --out results/1440p --size 1440p --budgets 10 60
python run.py --out results/1440p-stripes --size 1440p --variant stripes --budgets 10 60
```

Each run writes the QUBO to a zstd-compressed 4CQB file in row blocks, uploads it once, solves it by id at each
budget, and deletes the file:

```python
info = mrf.write_qubo("denoise-1440p.4cqb.zst", m)          # streamed in row blocks
model_id = fourcolors.qubo.upload("denoise-1440p.4cqb.zst")  # sent once
result = fourcolors.qubo.solve(input_id=model_id, time_limit_s=60)
```

It also runs a simple local baseline, iterated conditional modes (ICM): start from the noisy image and flip any
pixel that lowers the energy, until none does. It writes the before, after, difference and exact images, and the
triptych at the top of this post.

## Step 4: plot

```bash
python plot.py results/*/results.jsonl --out figures
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model, writer, references and plots; they do not call the service
```

The tests check that the MRF energy and the QUBO energy agree, the streamed 4CQB file against the client's own
writer, the min cut against brute force on small images (and that it refuses the stripe prior), that ICM never
raises the energy, the crop model, the exact pilot's rows, the triptych layout, and the script's rows and plot on
fixed canned answers.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
