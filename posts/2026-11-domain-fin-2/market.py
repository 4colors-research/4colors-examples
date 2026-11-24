"""Daily returns and the rolling-window estimates a desk re-computes every morning.

The drift between two mornings is not made up: each morning, expected returns mu and covariance Sigma are
re-estimated from the last `window` trading days of a return series, so today's model differs from yesterday's by
exactly one day in and one day out. That is what a real rebalancing loop sees.

Two sources:
- `simulate(n, T, seed)`: a seeded factor model of daily log returns whose loadings wander slowly (a random walk)
  and whose market volatility switches between a calm and a stressed regime. Synthetic, so the post ships no
  third-party data.
- `returns_from_prices_csv(path)`: daily log returns from a CSV of closing prices you supply (first column the
  date, then one column per asset). Check the terms of whatever source you use [CHECK].
"""
from __future__ import annotations

import csv
from typing import List, Tuple

import numpy as np

TRADING_DAYS = 252


def simulate(n: int, T: int, seed: int, factors: int = 3, drift: float = 0.02,
             stress_prob: float = 0.01) -> Tuple[np.ndarray, List[str]]:
    """(returns T x n, names). Daily log returns r_t = B_t f_t + e_t with slowly drifting loadings B_t."""
    rng = np.random.default_rng(seed)
    B = rng.normal(0.0, 0.12, size=(n, factors))
    B[:, 0] = np.abs(B[:, 0]) + 0.05                         # a market factor every asset loads on
    alpha = rng.normal(0.06, 0.04, size=n) / TRADING_DAYS     # annual expected returns around 6%
    idio = rng.uniform(0.10, 0.30, size=n) / np.sqrt(TRADING_DAYS)
    fvol = np.full(factors, 1.0 / np.sqrt(TRADING_DAYS))
    stressed = False
    out = np.empty((T, n))
    for t in range(T):
        if rng.random() < stress_prob:
            stressed = not stressed                           # regime switch: calm <-> stressed
        scale = fvol * (2.5 if stressed else 1.0)
        B = B + rng.normal(0.0, drift / np.sqrt(TRADING_DAYS), size=B.shape)
        f = rng.normal(0.0, 1.0, size=factors) * scale
        out[t] = alpha + B @ f + rng.normal(0.0, 1.0, size=n) * idio
    return out, [f"A{i:03d}" for i in range(n)]


def returns_from_prices_csv(path: str) -> Tuple[np.ndarray, List[str]]:
    """(daily log returns, names) from closing prices; rows with a missing or non-positive price are dropped."""
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
    return np.diff(np.log(np.asarray(prices)), axis=0), names


def estimate(returns: np.ndarray, end: int, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """(mu, Sigma), annualized, from returns[end - window:end] (the `window` days before morning `end`)."""
    if end < window or end > len(returns):
        raise ValueError(f"need {window} days before day {end}, have {len(returns)}")
    w = returns[end - window:end]
    return w.mean(axis=0) * TRADING_DAYS, np.cov(w, rowvar=False) * TRADING_DAYS


def mornings(returns: np.ndarray, window: int, days: int):
    """Yield (day, mu, Sigma) for day 0..days: morning `day` uses the window ending just before it."""
    first = len(returns) - days
    if first < window:
        raise ValueError("series too short for window + days")
    for d in range(days + 1):
        mu, sigma = estimate(returns, first + d, window)
        yield d, mu, sigma
