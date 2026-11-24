# Warm starts: re-solving tomorrow's problem from today's answer

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-warm-start&utm_content=readme-top)

## The model

A random 3-regular graph on n nodes (default 10,000) with integer edge weights 1..10, and the task is a weighted
Max-Cut: split the nodes in two so the cut edges weigh as much as possible. Each day, 5% of the edges get a new
weight. The graph stays the same, so yesterday's cut is still a valid cut today. `daily_maxcut.py` builds the days from
a seed; `examplekit.maxcut_qubo` writes Max-Cut as `min E(x)`, with `cut(x) = -E(x)`.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-warm-start&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-warm-start&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

What this post bills if you run Step 2 as shown: one 10 s day-0 job plus 6 days x 3 seeds x 2 arms x 3 s, at most
118 search-seconds.

## Step 1: the warm start in two lines

```python
import fourcolors.qubo

r0 = fourcolors.qubo.solve(model_today, time_limit_s=3, seed=1)                            # cold
r1 = fourcolors.qubo.solve(model_today, time_limit_s=3, seed=1, warm_start=yesterday)       # warm
```

`yesterday` is a list of 0/1 values in variable order (yesterday's `result.solution`), or a dict
`{label: value}`. Use the same seed and time limit for both, so the warm start is the only difference.

## Step 2: run the week

```bash
python warm_vs_cold.py --out runs/warm --n 10000 --days 6 --time-limit 3 --seeds 1 2 3
```

Day 0 is solved once, cold. On days 1 to 6, each seed runs a cold job and a warm job with the same time limit, and the
best cut of the day becomes tomorrow's warm start. Each day prints `day 3: best cut 78012 over 6 job(s)`.

The script also records a `carry_over` row per day: yesterday's answer scored on today's weights, before any search.
That is the number in the first line of this post.

## Step 3: time to quality, from the trajectory

Every result carries `trajectory`: each improvement the search found, with the service's own search time
(`t_search_s`). Time to quality is read from it, so nothing is timed on your side:

```python
from examplekit import time_to_target
t = time_to_target(r1.trajectory, target=-best_cut)     # energy terms: E = -cut; None if never reached
```

Before comparing warm with cold, check what the first point of a warm job is:

```bash
python first_points.py runs/warm/results.jsonl
```

It reports whether the service lists the warm start itself as the first trajectory point (at t = 0), or whether the
first point is the first improvement on it. On our run: the warm start is listed at t = 0 on every
warm job (36 jobs). That decides how to read the chart: if the warm start is listed, a warm job "reaches" any quality its
starting answer already has at 0 s.

```bash
python plot.py runs/warm --out figures --quality 0 0.001 0.005
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check the daily model, check the script's plumbing on fixed answers (warm jobs send
yesterday's answer, cold jobs none), and check the first-point verdicts.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
