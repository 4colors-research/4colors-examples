"""Knapsack and number partitioning, written the way most people send them: as textbook penalty QUBOs with
integer coefficients. Plus the explicit-row model of the knapsack, the decoders, and exact references.

Knapsack (0/1, one capacity):  max sum v_i x_i  s.t.  sum w_i x_i <= C.
  Penalty QUBO: min -sum v_i x_i + lam * (sum w_i x_i + sum b_k s_k - C)^2 with binary slack bits s (coefficients
  b = 1, 2, 4, ..., cut so they sum to exactly C) and the integer lam = max v + 1. Any load <= C is matched exactly
  by some slack, so feasible answers pay no penalty; an overload by o >= 1 costs lam * o^2 > max v, more than
  dropping any one item loses, so every optimum of the QUBO is a feasible packing with the optimal value.
Number partitioning: split a_1..a_n into two sets with sums as equal as possible.
  Ising: min (sum a_i s_i)^2 = sum a_i^2 + sum_{i<j} 2 a_i a_j s_i s_j over s in {-1,+1}^n. Every coefficient stays
  an integer below 2**53 for a_i < 2**20.

Instances are generated (dev only), never downloaded.
"""
from __future__ import annotations

import heapq
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit

MAX_BITS = 20                       # a_i < 2**20: 2 a_i a_j < 2**41, far inside 2**53


# ---- knapsack -------------------------------------------------------------------------------------

def knapsack_instance(n: int, seed: int, wmax: int = 1000, spread: int = 100) -> dict:
    """Weakly correlated items (v = w + U[-spread, spread], at least 1), capacity half the total weight."""
    rng = np.random.default_rng([seed, n, 14])
    w = rng.integers(1, wmax + 1, size=n)
    v = np.maximum(1, w + rng.integers(-spread, spread + 1, size=n))
    return {"problem": "knapsack", "name": f"knap-n{n}-s{seed}", "n": n, "seed": seed,
            "w": w.tolist(), "v": v.tolist(), "cap": int(w.sum() // 2)}


def slack_bits(cap: int) -> List[int]:
    """1, 2, 4, ..., with the last cut so they sum to exactly cap: every slack 0..cap reachable, none above."""
    out, total, b = [], 0, 1
    while total + b <= cap:
        out.append(b)
        total += b
        b *= 2
    if total < cap:
        out.append(cap - total)
    return out


def knapsack_penalty_qubo(inst: dict) -> Tuple[Dict[Tuple[int, int], int], List[int], int, int, int]:
    """(Q upper-triangular dict, linear, offset, n_vars, lam), all integers: the model a modeler would send."""
    w, v, cap = inst["w"], inst["v"], inst["cap"]
    n = len(w)
    lam = max(v) + 1
    bits = slack_bits(cap)
    terms = [(i, wi) for i, wi in enumerate(w)] + [(n + k, b) for k, b in enumerate(bits)]
    lin = [-vi for vi in v] + [0] * len(bits)
    Q: Dict[Tuple[int, int], int] = {}
    for t, (i, a) in enumerate(terms):                 # lam (sum a y - C)^2 with y*y = y
        lin[i] += lam * (a * a - 2 * cap * a)
        for k, b in terms[t + 1:]:
            Q[(i, k)] = Q.get((i, k), 0) + 2 * lam * a * b
    return Q, lin, lam * cap * cap, n + len(bits), lam


def knapsack_rows_model(inst: dict):
    """The same problem with one explicit row: min -value s.t. sum w x <= C (n variables, no slack, no lam)."""
    import fourcolors.qubo

    m = fourcolors.qubo.to_model({}, [-vi for vi in inst["v"]], n=inst["n"])
    m.add_constraint({i: wi for i, wi in enumerate(inst["w"])}, hi=inst["cap"])
    return m


def knapsack_penalty_model(inst: dict):
    import fourcolors.qubo

    Q, lin, off, n, _ = knapsack_penalty_qubo(inst)
    return fourcolors.qubo.to_model(Q, lin, n=n, offset=off)


def knapsack_penalty_energy(inst: dict, x) -> int:
    """The penalty QUBO's value, straight from its definition (an independent check of the builder)."""
    w, v, cap = inst["w"], inst["v"], inst["cap"]
    n = len(w)
    lam = max(v) + 1
    load = sum(wi for wi, xi in zip(w, x[:n]) if xi) + sum(b for b, s in zip(slack_bits(cap), x[n:]) if s)
    return -sum(vi for vi, xi in zip(v, x[:n]) if xi) + lam * (load - cap) ** 2


def knapsack_judge(inst: dict, x) -> dict:
    """Decode the item bits (slack ignored) and check the packing in plain Python."""
    if x is None:
        return {"feasible": None, "value": None, "weight": None}
    items = [int(b) for b in list(x)[:inst["n"]]]
    weight = sum(wi for wi, xi in zip(inst["w"], items) if xi)
    return {"feasible": weight <= inst["cap"], "value": sum(vi for vi, xi in zip(inst["v"], items) if xi),
            "weight": weight}


def knapsack_optimum(inst: dict) -> int:
    """The exact optimum by dynamic programming over capacity (integer weights; NumPy, O(n C))."""
    best = np.zeros(inst["cap"] + 1, dtype=np.int64)
    for wi, vi in zip(inst["w"], inst["v"]):
        if wi <= inst["cap"]:
            cand = best[:-wi] + vi if wi else best + vi
            best[wi:] = np.maximum(best[wi:], cand)
    return int(best[-1])


# ---- number partitioning --------------------------------------------------------------------------

def numpart_instance(n: int, seed: int, bits: int = MAX_BITS) -> dict:
    if bits > MAX_BITS:
        raise ValueError(f"bits <= {MAX_BITS} keeps every coefficient an exact integer")
    rng = np.random.default_rng([seed, n, 13])
    a = rng.integers(1, 2 ** bits, size=n)
    return {"problem": "numpart", "name": f"npp-n{n}-b{bits}-s{seed}", "n": n, "seed": seed, "bits": bits,
            "a": a.tolist()}


def numpart_ising(inst: dict) -> Tuple[Dict[Tuple[int, int], int], int]:
    """(J upper-triangular dict, offset) with (sum a s)^2 = offset + sum J_ij s_i s_j."""
    a = inst["a"]
    J = {(i, j): 2 * a[i] * a[j] for i in range(len(a)) for j in range(i + 1, len(a))}
    return J, sum(x * x for x in a)


def numpart_model(inst: dict):
    import fourcolors.qubo

    J, off = numpart_ising(inst)
    return fourcolors.qubo.to_model(J, [0] * inst["n"], n=inst["n"], vartype="SPIN", offset=off)


def numpart_diff(inst: dict, s) -> int | None:
    """|sum of one set - sum of the other| for a +-1 answer."""
    if s is None:
        return None
    s = list(s)
    if not all(x in (-1, 1) for x in s):
        raise ValueError("number partitioning answers are +-1")
    return abs(sum(ai * si for ai, si in zip(inst["a"], s)))


def numpart_lower_bound(inst: dict) -> int:
    """The difference has the parity of the total, so it is at least sum(a) mod 2. Reaching it is a certificate."""
    return sum(inst["a"]) % 2


def karmarkar_karp(a: List[int]) -> int:
    """The textbook differencing heuristic (Karmarkar and Karp 1982): a labeled reference line, not an arm."""
    h = [-x for x in a]
    heapq.heapify(h)
    while len(h) > 1:
        x, y = -heapq.heappop(h), -heapq.heappop(h)
        heapq.heappush(h, -(x - y))
    return -h[0] if h else 0


# ---- shared ---------------------------------------------------------------------------------------

def build(problem: str, n: int, seed: int) -> dict:
    return knapsack_instance(n, seed) if problem == "knapsack" else numpart_instance(n, seed)


def reference(inst: dict) -> dict:
    if inst["problem"] == "knapsack":
        return {"optimum": knapsack_optimum(inst), "reference_kind": "exact (dynamic programming)"}
    return {"lower_bound": numpart_lower_bound(inst), "kk": karmarkar_karp(inst["a"]),
            "reference_kind": "parity lower bound; Karmarkar-Karp as a reference line"}


def score(inst: dict, ref: dict, x) -> dict:
    """The one scorer at every pick. Knapsack: gap to the DP optimum, None if infeasible or no answer.
    Number partitioning: the difference, and whether it meets the parity bound (a proof of optimality)."""
    if inst["problem"] == "knapsack":
        j = knapsack_judge(inst, x)
        gap = None if not j["feasible"] else (ref["optimum"] - j["value"]) / ref["optimum"]
        return {**j, "gap": gap}
    d = numpart_diff(inst, x)
    return {"diff": d, "proved_optimal": d is not None and d == ref["lower_bound"]}
