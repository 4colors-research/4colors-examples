"""Feature selection as a QUBO: minimum redundancy, maximum relevance (mRMR) with "exactly k features" as a row.

For features f_1..f_d and a class label y, with mutual information I(.;.) estimated on discretized columns:
    minimize  E(x) = -alpha * sum_i I(f_i; y) x_i  +  (1 - alpha) * sum_{i<j} I(f_i; f_j) x_i x_j
    subject to sum_i x_i = k                                           (one exact row)
alpha weighs relevance against redundancy (the QUBO form of mRMR used by Mücke et al., QMI 2023).

References (plain Python, labeled as such in every row):
- `exact_best`: the optimum of E by enumerating every k-subset (feasible for d up to about 25);
- `greedy_qubo`: forward selection that adds the feature lowering E the most, k times;
- `greedy_mrmr`: the classic incremental mRMR of Peng, Long and Ding (TPAMI 2005, "MID" form), scored on E.

Data loaders read files you downloaded yourself (no download here):
- `load_german_numeric(path)`: UCI Statlog (German Credit Data), the `german.data-numeric` file (24 numeric
  attributes, last column the class 1 = good, 2 = bad);
- `load_csv(path, target)`: any CSV with a header and a class column (numeric columns are used as features);
- `synthetic(rows, d, seed)`: planted relevant features plus noisy copies of them (for tests and smoke runs).
"""
from __future__ import annotations

import csv
import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np

import fourcolors.qubo


# ---- data -----------------------------------------------------------------------------------------

def load_german_numeric(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows = [line.split() for line in open(path) if line.strip()]
    A = np.asarray(rows, dtype=float)
    X, y = A[:, :-1], (A[:, -1] == 2).astype(int)              # 1 = bad credit risk
    return X, y, [f"a{i + 1:02d}" for i in range(X.shape[1])]


def load_csv(path: str, target: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        header = [h for h in rd.fieldnames if h != target]
    keep = []
    for h in header:
        try:
            [float(r[h]) for r in rows]
            keep.append(h)
        except (TypeError, ValueError):
            continue
    X = np.asarray([[float(r[h]) for h in keep] for r in rows])
    labels = sorted({r[target].strip() for r in rows})
    y = np.asarray([labels.index(r[target].strip()) for r in rows])
    return X, y, [h.strip() for h in keep]


def synthetic(rows: int, d: int, seed: int, relevant: int = 4) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """`relevant` informative features, a noisy copy of each (redundant), the rest noise; y from the informative."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(rows, relevant))
    y = (Z @ rng.uniform(0.5, 1.5, size=relevant) + rng.normal(0, 0.5, size=rows) > 0).astype(int)
    cols = [Z, Z + rng.normal(0, 0.3, size=Z.shape), rng.normal(size=(rows, max(0, d - 2 * relevant)))]
    X = np.hstack(cols)[:, :d]
    return X, y, [f"f{i:02d}" for i in range(d)]


# ---- mutual information ---------------------------------------------------------------------------

def discretize(col: np.ndarray, bins: int = 8) -> np.ndarray:
    """Integer codes: columns with <= bins distinct values keep them; others go to quantile bins."""
    vals = np.unique(col)
    if len(vals) <= bins:
        return np.searchsorted(vals, col)
    edges = np.unique(np.quantile(col, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, col, side="right")


def mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """I(a; b) in nats for two integer-coded columns (plug-in estimate)."""
    a = np.asarray(a)
    b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    joint = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(joint, (ai, bi), 1.0)
    joint /= joint.sum()
    pa, pb = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())


def relevance_redundancy(X: np.ndarray, y: np.ndarray, bins: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """(rel[i] = I(f_i; y), red[i, j] = I(f_i; f_j) with a zero diagonal)."""
    D = np.column_stack([discretize(X[:, i], bins) for i in range(X.shape[1])])
    d = D.shape[1]
    rel = np.array([mutual_info(D[:, i], y) for i in range(d)])
    red = np.zeros((d, d))
    for i, j in itertools.combinations(range(d), 2):
        red[i, j] = red[j, i] = mutual_info(D[:, i], D[:, j])
    return rel, red


# ---- the QUBO -------------------------------------------------------------------------------------

def energy(rel, red, alpha: float, x) -> float:
    x = np.asarray(x, dtype=float)
    return float(-alpha * rel @ x + (1 - alpha) * 0.5 * x @ red @ x)     # red is symmetric, zero diagonal


def model(rel, red, alpha: float, k: int):
    """QUBO dict (each pair once) + linear terms + the exact row sum x = k."""
    d = len(rel)
    quad = {(i, j): (1 - alpha) * float(red[i, j]) for i, j in itertools.combinations(range(d), 2) if red[i, j]}
    m = fourcolors.qubo.to_model(quad, [-alpha * float(r) for r in rel], n=d)
    m.add_constraint({i: 1 for i in range(d)}, lo=k, hi=k)
    return m


def chosen(x) -> List[int]:
    return [i for i, v in enumerate(x) if v]


def as_vector(idx: Sequence[int], d: int) -> List[int]:
    x = [0] * d
    for i in idx:
        x[i] = 1
    return x


# ---- references -----------------------------------------------------------------------------------

def exact_best(rel, red, alpha: float, k: int, chunk: int = 100_000) -> Tuple[float, List[int]]:
    """(optimum of E, 0/1 vector) over every k-subset, vectorized."""
    d = len(rel)
    rel = np.asarray(rel)
    red = np.asarray(red)
    best, arg = np.inf, None
    it = itertools.combinations(range(d), k)
    while True:
        block = np.array(list(itertools.islice(it, chunk)), dtype=np.int32)
        if block.size == 0:
            break
        pair = red[block[:, :, None], block[:, None, :]].sum(axis=(1, 2)) * 0.5
        vals = -alpha * rel[block].sum(axis=1) + (1 - alpha) * pair
        j = int(np.argmin(vals))
        if vals[j] < best:
            best, arg = float(vals[j]), block[j]
    return best, as_vector([int(i) for i in arg], d)


def greedy_qubo(rel, red, alpha: float, k: int) -> List[int]:
    """Forward selection on E itself: add the feature that lowers E most (ties: lowest index)."""
    d = len(rel)
    S: List[int] = []
    for _ in range(k):
        gains = [(-alpha * rel[i] + (1 - alpha) * sum(red[i, j] for j in S), i) for i in range(d) if i not in S]
        S.append(min(gains)[1])
    return as_vector(S, d)


def greedy_mrmr(rel, red, k: int) -> List[int]:
    """Peng et al. 2005 incremental mRMR (MID): start from the most relevant, then max rel_i - mean_j red_ij."""
    d = len(rel)
    S = [int(np.argmax(rel))]
    while len(S) < k:
        score = [(rel[i] - np.mean([red[i, j] for j in S]), -i) for i in range(d) if i not in S]
        S.append(-max(score)[1])
    return as_vector(S, d)


# ---- downstream context: cross-validated AUC of a logistic regression ------------------------------

def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve by ranks (ties get average ranks)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 25) -> np.ndarray:
    """Newton's method for L2-regularized logistic regression; returns weights with the intercept first."""
    A = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(A.shape[1])
    reg = np.eye(A.shape[1]) * l2
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) + reg @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + reg
        w -= np.linalg.solve(H, g)
    return w


def cv_auc(X: np.ndarray, y: np.ndarray, cols: Sequence[int], folds: int = 5, seed: int = 0) -> float:
    """Mean AUC over stratified-by-shuffle folds, features standardized on each training fold."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    out = []
    for f in range(folds):
        te = parts[f]
        tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        Xs = X[:, list(cols)]
        mu, sd = Xs[tr].mean(axis=0), Xs[tr].std(axis=0) + 1e-12
        w = logistic_fit((Xs[tr] - mu) / sd, y[tr])
        z = np.hstack([np.ones((len(te), 1)), (Xs[te] - mu) / sd]) @ w
        out.append(auc(z, y[te]))
    return float(np.mean(out))
