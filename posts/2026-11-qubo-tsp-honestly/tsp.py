"""The textbook n^2 one-hot TSP QUBO, two ways (exact rows and penalties), and an exact reference for small n.

Variable x[i*n + p] = 1 when city i is visited at position p. The tour length is
    sum over p, and over ordered city pairs i != j, of d[i][j] * x[i,p] * x[j,p+1 mod n],
which is n^2 (n - 1) quadratic terms. "Each city once" and "each position once" are 2n one-hot constraints:
penalties lam * (sum - 1)^2 in the penalty model, exact rows in the exact model.

Instances are random Euclidean: integer coordinates in [0, 1000), distances rounded to the nearest integer
(TSPLIB's EUC_2D rule), so every tour length is an exact integer.
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Dict, List, Optional, Tuple

import fourcolors.qubo


def random_instance(n: int, seed: int) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    rnd = random.Random(seed)
    pts = [(rnd.randrange(1000), rnd.randrange(1000)) for _ in range(n)]
    d = [[int(math.floor(math.dist(a, b) + 0.5)) for b in pts] for a in pts]
    return pts, d


def tour_length(d, tour: List[int]) -> int:
    return sum(d[tour[k]][tour[(k + 1) % len(tour)]] for k in range(len(tour)))


# ---- the QUBO -------------------------------------------------------------------------------------

def routing_input_size(n: int) -> int:
    """What a routing solver reads for the same problem: one distance per city pair."""
    return n * (n - 1) // 2


def model_size(n: int, penalty: bool) -> Dict[str, int]:
    """Variables, quadratic terms and rows of the n-city model (n >= 3), by formula; the tests check it against the
    built model."""
    quad = n * n * (n - 1)
    if penalty:
        quad += 2 * n * (n * (n - 1) // 2)      # pairs inside each city row and each position column
    return {"n_vars": n * n, "n_quad": quad, "n_rows": 0 if penalty else 2 * n}


def _distance_terms(d) -> Dict[Tuple[int, int], float]:
    n = len(d)
    Q: Dict[Tuple[int, int], float] = {}
    for p in range(n):
        q = (p + 1) % n
        for i in range(n):
            for j in range(n):
                if i != j:
                    a, b = i * n + p, j * n + q
                    key = (a, b) if a < b else (b, a)
                    Q[key] = Q.get(key, 0) + d[i][j]
    return Q


def exact_model(d):
    """Distance objective plus 2n exact rows."""
    n = len(d)
    m = fourcolors.qubo.to_model(_distance_terms(d), [0] * (n * n), n=n * n)
    for i in range(n):
        m.add_constraint({i * n + p: 1 for p in range(n)}, lo=1, hi=1)       # city i visited once
    for p in range(n):
        m.add_constraint({i * n + p: 1 for i in range(n)}, lo=1, hi=1)       # one city at position p
    return m


def penalty_model(d, lam: float):
    """Distance objective plus lam * sum of (sum - 1)^2 over the 2n one-hot groups."""
    n = len(d)
    Q = {k: float(v) for k, v in _distance_terms(d).items()}
    lin = [0.0] * (n * n)
    groups = [[i * n + p for p in range(n)] for i in range(n)] + [[i * n + p for i in range(n)] for p in range(n)]
    for g in groups:
        for a in g:
            lin[a] -= lam                                   # (sum - 1)^2 = -sum + 2 sum_{a<b} x_a x_b + 1
        for a, b in itertools.combinations(g, 2):
            Q[(a, b)] = Q.get((a, b), 0.0) + 2 * lam
    return fourcolors.qubo.to_model(Q, lin, n=n * n, offset=lam * len(groups))


def penalty_energy(d, lam: float, x) -> float:
    """The penalty objective written out directly, for tests."""
    n = len(d)
    X = [[x[i * n + p] for p in range(n)] for i in range(n)]
    length = sum(d[i][j] * X[i][p] * X[j][(p + 1) % n] for p in range(n) for i in range(n) for j in range(n) if i != j)
    pen = sum((sum(X[i]) - 1) ** 2 for i in range(n)) + sum((sum(X[i][p] for i in range(n)) - 1) ** 2 for p in range(n))
    return length + lam * pen


def encode(tour: List[int]) -> List[int]:
    n = len(tour)
    x = [0] * (n * n)
    for p, i in enumerate(tour):
        x[i * n + p] = 1
    return x


def decode(n: int, x) -> Optional[List[int]]:
    """The tour (cities in position order), or None when x is not a permutation matrix."""
    tour = []
    for p in range(n):
        cities = [i for i in range(n) if x[i * n + p]]
        if len(cities) != 1:
            return None
        tour.append(cities[0])
    return tour if len(set(tour)) == n else None


# ---- classical references -------------------------------------------------------------------------

def held_karp(d) -> Tuple[int, List[int]]:
    """The optimal tour by dynamic programming over subsets, O(2^n n^2); vectorized with numpy; fine to n = 13."""
    import numpy as np

    n = len(d)
    if n <= 3:
        tour = list(range(n))
        return tour_length(d, tour), tour
    D = np.asarray(d, dtype=np.int64)
    m = n - 1                                   # city 0 is the start; subsets of cities 1..n-1
    INF = np.iinfo(np.int64).max // 4
    dp = np.full((1 << m, m), INF, dtype=np.int64)
    parent = np.full((1 << m, m), -1, dtype=np.int64)
    for j in range(m):
        dp[1 << j, j] = D[0, j + 1]
    for mask in range(1, 1 << m):
        cur = dp[mask]
        if (cur >= INF).all():
            continue
        for k in range(m):
            if mask & (1 << k):
                continue
            # extend every end j in mask by city k
            cand = cur + D[1:, k + 1]
            j = int(np.argmin(cand))
            nm = mask | (1 << k)
            if cand[j] < dp[nm, k]:
                dp[nm, k] = cand[j]
                parent[nm, k] = j
    full = (1 << m) - 1
    ends = dp[full] + D[1:, 0]
    j = int(np.argmin(ends))
    best = int(ends[j])
    tour, mask = [], full
    while j >= 0:
        tour.append(j + 1)
        pj = int(parent[mask, j])
        mask ^= 1 << j
        j = pj
    return best, [0] + tour[::-1]


def brute_force(d) -> int:
    n = len(d)
    return min(tour_length(d, [0, *p]) for p in itertools.permutations(range(1, n)))
