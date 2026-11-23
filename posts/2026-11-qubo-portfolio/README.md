# Portfolio selection with a cardinality constraint

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-portfolio&utm_content=readme-top)

Hands-on companion code for the post. Everything runs through the public `fourcolors` Python client against the
4colors Optimizer for QUBO.

## What you will learn

- How Markowitz mean-variance selection over "pick k of n assets" becomes a QUBO: the covariance matrix is the
  quadratic part, the expected returns are the linear part.
- How to pass that QUBO as one numpy matrix, and add "exactly k assets" as one exact row instead of a penalty.
- How to trace the efficient frontier by sweeping the risk aversion q, and check each answer against the exact
  optimum (by enumeration) where that is possible.

## The model

Pick k of n assets, equally weighted. With expected annual returns `mu` and covariance `Sigma`:

```
return(x) = mu . x / k        risk(x)^2 = x' Sigma x / k^2
minimize  q * risk(x)^2 - return(x)      subject to   sum x = k
```

`q = 0` picks the k highest expected returns; a large `q` picks the least risky mix. In code:

```python
import numpy as np
import fourcolors.qubo

Q = (q / k**2) * Sigma - np.diag(mu / k)       # E(x) = x'Qx, the objective above
m = fourcolors.qubo.to_model(Q)
m.add_constraint({i: 1 for i in range(n)}, lo=k, hi=k)     # exactly k assets
r = fourcolors.qubo.solve(m, time_limit_s=1, seed=1)
```

A symmetric matrix is read as `x'Qx`, so each pair counts twice, as written.

## Data

- Default: `portfolio.synthetic(n, seed)`, a seeded three-factor model with annual returns around 8% and
  volatilities of roughly 15% to 40%. Everyone gets the same numbers.
- Your own: `--prices prices.csv`, a CSV of daily closing prices (a date column, then one column per asset) from any
  public source. Expected returns and covariance are estimated from daily log returns, annualized with 252 days.
  Estimated means are noisy; treat the frontier as an illustration, not advice.

## Setup

Start a free trial at [TRIAL SIGNUP URL] and copy your key, then:

```bash
pip install "fourcolors[optimizer]" numpy matplotlib
export FOURCOLORS_API_KEY=...        # the trial key, or FOURCOLORS_KEY_FILE=/path/to/key
```

## Step 1: a frontier you can check exactly (30 assets, pick 6)

```bash
python frontier.py --out runs/n30 --n 30 --k 6 --time-limit 5 --seeds 1 2 3
```

C(30, 6) = 593,775 portfolios is small enough to enumerate, so every answer has an exact reference. Lines read like
`q=4     seed=1 picked=6 return=... risk=... objective=... (reference ..., enumeration (exact))`.

## Step 2: a size the demos skip (200 assets, pick 20)

```bash
python frontier.py --out runs/n200 --n 200 --k 20 --time-limit 5 --seeds 1 2 3
```

C(200, 20) is far too many to enumerate; the reference is a swap local search (a heuristic), so a negative gap means
"better than the heuristic", not "better than optimal".

## Step 3: plot

```bash
python plot.py runs/n30 runs/n200 --out figures
```

`figures/frontier_n30_k6.png` and `figures/frontier_n200_k20.png`: risk against return, the reference line and the
solver's answers.

What to check: at n = 30, compare each answer with the exact optimum (`gap_abs` in results.jsonl, 0 means it
found the optimum); at n = 200, a negative gap means better than the heuristic. Every answer should pick exactly
k assets. The numbers we got are in the post.

## Run the tests

```bash
python -m pytest -q tests
```

The tests check the matrix against the plain-Python objective on every assignment of a small case, the brute force
and the swap search against plain enumeration, the CSV reader, and both script paths on fixed, hand-written answers
(`tests/canned.py`). The enumeration is the post's labeled reference optimum, not a stand-in for the service.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and
run the commands above. Every answer comes from the 4colors service.
