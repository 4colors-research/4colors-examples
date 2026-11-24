"""The morning model: pick exactly k of n assets (equal weight), mean-variance, re-estimated every day.

Same model as the portfolio post (../qubo-portfolio):
    minimize  q * x'Sigma x / k^2 - mu.x / k        subject to  sum x = k (one exact row)
plus an optional trading-cost term. Changing a name costs `cost` (in the same annual-return units as mu, per name
sold or bought). With yesterday's picks y, the number of names changed is
    changes(x, y) = sum_i x_i (1 - 2 y_i) + sum_i y_i
which is linear in x, so the cost goes on the diagonal and the model stays a QUBO with one exact row.
"""
from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np

import fourcolors.qubo


def objective(mu, sigma, k: int, q: float, x, *, prev: Optional[Sequence[int]] = None, cost: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    val = q * float(x @ np.asarray(sigma) @ x) / k ** 2 - float(np.dot(mu, x)) / k
    if prev is not None and cost:
        val += cost * changes(x, prev)
    return val


def changes(x, y) -> int:
    """Names sold plus names bought between picks y (yesterday) and x (today)."""
    return int(sum(int(a) != int(b) for a, b in zip(x, y)))


def model(mu, sigma, k: int, q: float, *, prev: Optional[Sequence[int]] = None, cost: float = 0.0):
    """The QUBO as one symmetric matrix (x'Qx counts each off-diagonal pair twice, as the variance needs) plus the
    exact row; with `prev` and `cost`, the trading cost on the diagonal and its constant as the offset."""
    mu = np.asarray(mu, dtype=float)
    lin = -mu / k
    offset = 0.0
    if prev is not None and cost:
        y = np.asarray(prev, dtype=float)
        lin = lin + cost * (1.0 - 2.0 * y)
        offset = cost * float(y.sum())
    Q = (q / k ** 2) * np.asarray(sigma, dtype=float) + np.diag(lin)
    m = fourcolors.qubo.to_model(Q, offset=offset)
    m.add_constraint({i: 1 for i in range(len(mu))}, lo=k, hi=k)
    return m


def picks(x) -> List[int]:
    return [i for i, v in enumerate(x) if v]


def exact_best(mu, sigma, k: int, q: float, *, prev=None, cost: float = 0.0,
               chunk: int = 200_000) -> Tuple[float, List[int]]:
    """(optimum, picks as 0/1) by enumerating every k-subset, vectorized. For n <= 30 (C(30, 6) = 593,775)."""
    n = len(mu)
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    y = None if prev is None else np.asarray(prev, dtype=bool)
    best, best_idx = np.inf, None
    it = itertools.combinations(range(n), k)
    while True:
        block = np.array(list(itertools.islice(it, chunk)), dtype=np.int32)
        if block.size == 0:
            break
        sub = sigma[block[:, :, None], block[:, None, :]].sum(axis=(1, 2))
        vals = q * sub / k ** 2 - mu[block].sum(axis=1) / k
        if y is not None and cost:
            held = y[block].sum(axis=1)                         # yesterday's names kept
            vals = vals + cost * ((k - held) + (y.sum() - held))
        j = int(np.argmin(vals))
        if vals[j] < best:
            best, best_idx = float(vals[j]), block[j]
    x = [0] * n
    for i in best_idx:
        x[int(i)] = 1
    return best, x
