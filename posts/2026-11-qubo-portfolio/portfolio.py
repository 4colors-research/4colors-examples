"""Markowitz mean-variance selection over binary picks, with "exactly k assets" as an exact row.

Pick k of n assets, equally weighted (1/k each). With expected annual returns mu and covariance Sigma:
    portfolio return  R(x) = mu . x / k
    portfolio risk    V(x) = x' Sigma x / k^2         (variance; risk is its square root)
    minimize          q * V(x) - R(x)                 subject to sum x = k (exact row)
q is the risk aversion: q = 0 picks the k highest returns; large q picks the k least risky mix. Sweeping q traces
the efficient frontier of k-asset equal-weight portfolios.

Data: `synthetic(n, seed)` is a seeded three-factor model with realistic scales (annual returns around 8%, volatility
15% to 40%). `from_prices_csv(path)` reads a CSV of daily closing prices (a date column, then one column per asset)
from any public source and estimates mu and Sigma from daily log returns.
"""
from __future__ import annotations

import csv
import itertools
from typing import List, Tuple

import numpy as np

import fourcolors.qubo


def synthetic(n: int, seed: int, factors: int = 3) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(mu, Sigma, names): Sigma = B B' + D, a factor model, positive definite by construction."""
    rng = np.random.default_rng(seed)
    B = rng.normal(0.0, 0.12, size=(n, factors))
    B[:, 0] = np.abs(B[:, 0]) + 0.05                # a market factor every asset loads on
    idio = rng.uniform(0.10, 0.30, size=n)
    sigma = B @ B.T + np.diag(idio ** 2)
    beta = B[:, 0] / B[:, 0].mean()
    mu = 0.02 + 0.05 * beta + rng.normal(0.0, 0.03, size=n)   # more market risk earns more, with noise
    return mu, sigma, [f"A{i:03d}" for i in range(n)]


def from_prices_csv(path: str, periods_per_year: int = 252) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(mu, Sigma, names), annualized, from daily closing prices: first column the date, then one per asset.
    Rows with a missing or non-positive price are dropped."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    names = rows[0][1:]
    prices = []
    for r in rows[1:]:
        try:
            vals = [float(v) for v in r[1:]]
        except ValueError:
            continue
        if len(vals) == len(names) and all(v > 0 for v in vals):
            prices.append(vals)
    P = np.asarray(prices)
    rets = np.diff(np.log(P), axis=0)
    return rets.mean(axis=0) * periods_per_year, np.cov(rets, rowvar=False) * periods_per_year, names


def portfolio_return(mu, x, k: int) -> float:
    return float(np.dot(mu, x)) / k


def portfolio_variance(sigma, x, k: int) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ sigma @ x) / k ** 2


def objective(mu, sigma, k: int, q: float, x) -> float:
    return q * portfolio_variance(sigma, x, k) - portfolio_return(mu, x, k)


def model(mu, sigma, k: int, q: float):
    """The QUBO as one symmetric numpy matrix (E = x'Qx; to_model folds the lower triangle onto the upper one, so
    each pair counts twice, as written) plus the exact row sum x = k."""
    Q = (q / k ** 2) * np.asarray(sigma, dtype=float) - np.diag(np.asarray(mu, dtype=float) / k)
    m = fourcolors.qubo.to_model(Q)
    m.add_constraint({i: 1 for i in range(len(mu))}, lo=k, hi=k)
    return m


def picks(x) -> List[int]:
    return [i for i, xi in enumerate(x) if xi]


# ---- references -----------------------------------------------------------------------------------

def all_portfolios(mu, sigma, k: int, chunk: int = 200_000):
    """(return, variance, index tuples) of every k-subset, vectorized; C(30, 6) = 593,775 takes a few seconds."""
    n = len(mu)
    combos = np.array(list(itertools.combinations(range(n), k)), dtype=np.int16)
    R = np.empty(len(combos))
    V = np.empty(len(combos))
    for s in range(0, len(combos), chunk):
        c = combos[s:s + chunk].astype(np.int64)
        R[s:s + chunk] = mu[c].sum(axis=1) / k
        V[s:s + chunk] = sigma[c[:, :, None], c[:, None, :]].sum(axis=(1, 2)) / k ** 2
    return R, V, combos


def brute_force(mu, sigma, k: int, qs) -> dict:
    """{q: (objective, picks)}: the exact optimum for each q, by enumeration."""
    R, V, combos = all_portfolios(mu, sigma, k)
    out = {}
    for q in qs:
        j = int(np.argmin(q * V - R))
        out[q] = (float(q * V[j] - R[j]), [int(i) for i in combos[j]])
    return out


def swap_search(mu, sigma, k: int, q: float, seed: int = 0, restarts: int = 10) -> Tuple[float, List[int]]:
    """A plain heuristic reference for sizes too large to enumerate: random starts, then best-improvement swaps
    (one asset out, one in) until none helps."""
    rng = np.random.default_rng(seed)
    n = len(mu)
    best = (np.inf, None)
    for _ in range(restarts):
        x = np.zeros(n)
        x[rng.choice(n, k, replace=False)] = 1
        f = objective(mu, sigma, k, q, x)
        while True:
            ins, outs = np.flatnonzero(x), np.flatnonzero(x == 0)
            g = sigma @ x                                  # swap i out, j in: exact change in objective below
            dR = (mu[outs][None, :] - mu[ins][:, None]) / k
            dV = (2 * (g[outs][None, :] - g[ins][:, None]) + np.diag(sigma)[ins][:, None]
                  + np.diag(sigma)[outs][None, :] - 2 * sigma[np.ix_(ins, outs)]) / k ** 2
            delta = q * dV - dR
            a, b = np.unravel_index(int(np.argmin(delta)), delta.shape)
            if delta[a, b] >= -1e-15:
                break
            x[ins[a]], x[outs[b]] = 0, 1
            f = objective(mu, sigma, k, q, x)
        if f < best[0]:
            best = (f, picks(x))
    return best
