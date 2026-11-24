# Rebalance every morning: warm-start today's portfolio from yesterday's

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=domain-fin-2&utm_content=readme-top)

## What you will learn

- How to run a daily re-solve loop on a rolling-window estimate of mu and Sigma.
- How to pass yesterday's holdings as `warm_start`, and score them on today's model before any search (the
  carry-over gap).
- How to read time to quality from the job's `trajectory`, against the exact optimum where one can be enumerated.
- How to add a per-name trading cost as a linear term, and see turnover fall.

## The model

Pick k of n assets, equally weighted, as in the portfolio post:

```
minimize  q * x' Sigma x / k^2  -  mu . x / k        subject to   sum x = k   (one exact row)
```

Each morning `mu` and `Sigma` are re-estimated from the last 252 trading days of daily log returns
(`market.mornings`). The drift between mornings is therefore whatever one day in and one day out does to the
estimate: nothing in the code perturbs the model by hand.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=domain-fin-2&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=domain-fin-2&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

What this post bills if you run Step 2 as shown: per size one 5 s morning-0 job plus 10 mornings x 3 seeds x 2 arms x
1 s, at most 65 search-seconds; both sizes at most 130 s.

## Step 1: one morning, cold and warm

```python
import fourcolors.qubo
import market, rebalance

returns, names = market.simulate(n=200, T=600, seed=7)
mu, Sigma = market.estimate(returns, end=len(returns), window=252)       # this morning's estimate
m = rebalance.model(mu, Sigma, k=20, q=2.0)                               # QUBO matrix + "exactly 20 names"

cold = fourcolors.qubo.solve(m, time_limit_s=1, seed=1)
warm = fourcolors.qubo.solve(m, time_limit_s=1, seed=1, warm_start=yesterday)   # yesterday: 0/1 per asset
```

Use the same seed and the same time limit for both jobs, so the warm start is the only difference. Before any
search, `m.energy(yesterday)` tells you how good yesterday's holdings already are on today's data.

## Step 2: run ten mornings

```bash
python run.py --out runs/rebalance-n200 --n 200 --k 20 --days 10 --time-limit 1 --seeds 1 2 3
python run.py --out runs/rebalance-n30  --n 30  --k 6  --days 10 --time-limit 1 --seeds 1 2 3
```

Morning 0 is solved once, cold. On mornings 1 to 10, each seed runs a cold and a warm job; the best answer of the
morning (scored in plain Python, rows checked) becomes the portfolio held into the next morning. The script writes:

- a `carry_over` row per morning: yesterday's holdings scored on today's model;
- at n = 30, a `reference` row per morning: the exact optimum by enumerating all C(30, 6) = 593,775 portfolios, so
  "reached the optimum" is a certificate there, not an estimate;
- a `held` row per morning with the number of names changed.

Every constrained answer is checked by the service against the exact row; the script checks it again with
`m.feasible`, and a morning with no feasible answer keeps yesterday's holdings.

## Step 3: a trading cost in one line

Changing a name costs money. With yesterday's holdings `y`, the number of names sold plus bought is
`sum_i x_i (1 - 2 y_i) + sum_i y_i`, linear in today's `x`, so a cost per name goes on the diagonal of the same QUBO:

```python
m = rebalance.model(mu, Sigma, k=20, q=2.0, prev=yesterday, cost=0.0005)     # 5 bp of objective per name changed
```

```bash
python run.py --out runs/rebalance-n200-cost --n 200 --k 20 --days 10 --time-limit 1 --seeds 1 2 3 --cost 0.0005
```

## Step 4: pictures

```bash
python plot.py runs/rebalance-n30 runs/rebalance-n200 --out figures
python plot.py runs/rebalance-n200 runs/rebalance-n200-cost --out figures-cost
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that the QUBO energy equals the objective (with and without the trading cost) on every assignment of
a small case, that the enumeration matches brute force, that the rolling estimate equals the window's mean and
covariance, and they run the script's plumbing on fixed hand-made answers (warm jobs send yesterday's holdings, cold
jobs none; an infeasible answer is never held).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
