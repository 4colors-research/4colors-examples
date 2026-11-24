# Pick exactly k stocks: portfolio selection with a cardinality row

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-portfolio&utm_content=readme-top)

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

`q = 0` picks the k highest expected returns; a large `q` picks the least risky mix. Equal weighting is a modeling
simplification: it turns "how much of each" into "which ones", which is what a binary model can express. In code:

```python
import numpy as np
import fourcolors.qubo

Q = (q / k**2) * Sigma - np.diag(mu / k)       # E(x) = x'Qx, the objective above
m = fourcolors.qubo.to_model(Q)
m.add_constraint({i: 1 for i in range(n)}, lo=k, hi=k)     # exactly k assets
r = fourcolors.qubo.solve(m, time_limit_s=1, seed=1)
# 1 s is enough to try it. Each search runs to its time limit and is billed for it.
```

Two details make the matrix right. On binaries x_i^2 = x_i, so the diagonal `-mu/k` is the linear return term. And a
symmetric matrix is read as `x'Qx`, so each off-diagonal pair counts twice, exactly as the variance needs.

## Data

- Default: `portfolio.synthetic(n, seed)`, a seeded three-factor model with annual returns around 8% and
  volatilities of roughly 15% to 40%. Everyone gets the same numbers.
- Your own: `--prices prices.csv`, a CSV of daily closing prices (a date column, then one column per asset) from any
  public source whose terms you have checked. Expected returns and covariance are estimated from daily log returns,
  annualized with 252 days. Estimated means are noisy; treat the frontier as an illustration, not advice.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-portfolio&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-portfolio&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" numpy matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

## Step 1: a frontier you can check exactly (30 assets, pick 6)

```bash
python frontier.py --out runs/n30 --n 30 --k 6 --time-limit 5 --seeds 1 2 3
```

C(30, 6) = 593,775 portfolios is small enough to enumerate in a few seconds of numpy, so each answer has an exact
reference. The script sweeps q over 0, 0.5, 1, 2, 4, 8, 16 and 32, solves each with three seeds, and recomputes the
picks, return, risk and objective from the 0/1 answer. Lines read like:

```
q=4     seed=1 picked=6 return=0.09798368460782597 risk=0.13095416938056845 objective=-0.029387706695207516 (reference -0.029387706695207516, enumeration (exact))
```

## Step 2: a size the demos skip (200 assets, pick 20)

```bash
python frontier.py --out runs/n200 --n 200 --k 20 --time-limit 5 --seeds 1 2 3
```

C(200, 20) is far too many to enumerate. The reference here is a swap local search (one asset out, one in, until no
swap helps), which is a heuristic: it shows the answers are sensible and meet the row, not that they are optimal.

## Step 3: plot

```bash
python plot.py runs/n30 runs/n200 --out figures
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check the matrix against the plain-Python objective on every assignment of a small case, the brute force
and the swap search against plain enumeration (with and without a sector cap), that the sector rows admit exactly the
capped portfolios, the CSV reader, and every script path on fixed hand-made answers.

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
