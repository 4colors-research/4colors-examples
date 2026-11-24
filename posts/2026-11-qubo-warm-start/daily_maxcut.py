"""A weighted Max-Cut model that changes a little every day.

Day 0 is a random d-regular graph with integer edge weights 1..10. Each later day redraws the weight of a fixed
share of the edges (default 5%), chosen at random, on top of the day before. The graph itself never changes, so
yesterday's answer is a valid (and usually good) answer today: exactly the case a warm start is for.

Max-Cut is written as min E(x) = sum_w (2 x_i x_j - x_i - x_j), so cut(x) = -E(x) (examplekit.maxcut_qubo).
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import fourcolors.qubo

import examplekit

Edge = Tuple[int, int]


def base_graph(n: int, degree: int, seed: int) -> Tuple[List[Edge], Dict[Edge, int]]:
    edges = examplekit.random_regular(n, degree, seed)
    rnd = random.Random(seed + 1)
    return edges, {e: rnd.randint(1, 10) for e in edges}


def next_day(edges: List[Edge], weights: Dict[Edge, int], day: int, share: float, seed: int) -> Dict[Edge, int]:
    """Day `day`'s weights from the day before: round(share * |E|) edges get a new weight 1..10 (always a
    different one, so the change is real)."""
    rnd = random.Random(seed * 1_000_003 + day)
    out = dict(weights)
    for e in rnd.sample(edges, round(share * len(edges))):
        out[e] = rnd.choice([w for w in range(1, 11) if w != weights[e]])
    return out


def days(n: int, degree: int, n_days: int, share: float, seed: int):
    """[(day, edges, weights)] for day 0..n_days."""
    edges, w = base_graph(n, degree, seed)
    out = [(0, edges, w)]
    for d in range(1, n_days + 1):
        w = next_day(edges, w, d, share, seed)
        out.append((d, edges, w))
    return out


def model(n: int, edges: List[Edge], weights: Dict[Edge, int]):
    Q, lin = examplekit.maxcut_qubo(edges, weights)
    return fourcolors.qubo.to_model({k: int(v) for k, v in Q.items()}, [int(v) for v in lin], n=n)


def cut(edges, weights, x) -> int:
    return int(examplekit.cut_value(edges, x, weights))
