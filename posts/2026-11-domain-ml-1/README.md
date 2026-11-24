# Pick the k best features: feature selection as a QUBO, with credit scoring as the example

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=domain-ml-1&utm_content=readme-top)

## What you will learn

- How mutual information turns a table of features into a relevance vector and a redundancy matrix.
- How mRMR becomes a QUBO, and how "exactly k" goes in as one exact row.
- How to check each returned answer three ways: the exact optimum by enumeration (small d), greedy on the same objective,
  and Peng's greedy mRMR scored on the same objective.
- How to report downstream accuracy honestly: as context, not as the headline.

## The model

For features f_1..f_d and a class label y, with mutual information I(.;.) estimated on discretized columns
(`features.discretize`, 8 quantile bins):

```
minimize  E(x) = -alpha * sum_i I(f_i; y) x_i  +  (1 - alpha) * sum_{i<j} I(f_i; f_j) x_i x_j
subject to  sum_i x_i = k
```

`alpha` weighs relevance against redundancy: near 1 it picks the k most relevant features whatever they share; lower
values push out near copies. In code:

```python
import fourcolors.qubo
import features as fs

X, y, names = fs.load_german_numeric("data/german.data-numeric")
rel, red = fs.relevance_redundancy(X, y)          # I(f_i; y) and I(f_i; f_j)
m = fs.model(rel, red, alpha=0.7, k=8)            # QUBO + the row sum x = 8
r = fourcolors.qubo.solve(m, time_limit_s=1, seed=1)
picked = [names[i] for i in fs.chosen(r.solution)]
```

## Data

Both datasets are public and fetched by you at run time; this folder ships no data and the scripts download nothing.

The first is small enough that every set of k features can be enumerated (C(24, 10) = 1,961,256), so "the optimum"
is a certificate there. The second is not: C(95, 10) is about 10^13.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=domain-ml-1&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=domain-ml-1&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

3. Download the two files from their UCI pages into `data/` (check the license line on each page first).

What this post bills if you run Step 2 as shown: 2 datasets x 3 values of k x 3 values of alpha x 3 seeds x 1 s, at
most 54 search-seconds.

## Step 1: one dataset, one k

The snippet above solves one cell. Every constrained answer is checked by the service against the row; check it
again in plain Python with `m.feasible(r.solution)`, and score it with `fs.energy(rel, red, alpha, r.solution)`.

## Step 2: the grid, with references

```bash
python run.py --out runs/fs --dataset german:data/german.data-numeric \
  --dataset csv:taiwan-bankruptcy:"Bankrupt?":data/data.csv --k 5 8 10 --alpha 0.5 0.7 0.9 --seeds 1 2 3 --time-limit 1 --auc
```

For every (dataset, k, alpha) the script writes three kinds of rows:

- `reference`: `exact` (enumeration, where C(d, k) is at most 3 million), `greedy_qubo` (forward selection on E
  itself) and `greedy_mrmr` (Peng's incremental mRMR), each scored on the same E;
- `solve`: our answer per seed, its E recomputed in plain Python, and both feasibility checks;
- with `--auc`, a 5-fold cross-validated logistic-regression AUC for each returned answer, as context.

## Step 3: pictures

```bash
python plot.py runs/fs --out figures
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that the QUBO energy equals the objective and the row admits exactly the k-subsets on a small case,
that enumeration matches brute force, that forward selection is fooled on a hand-made case where the optimum is not
greedy's, the mutual-information and AUC helpers, both loaders, and the script's plumbing on fixed hand-made answers.

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
