"""Two small constrained problems, each written two ways for the 4colors Optimizer for QUBO.

1. Assignment: n workers, n jobs, cost c[i][j]; every worker does exactly one job and every job gets exactly one
   worker. Variable x[i*n + j] = 1 when worker i does job j.
2. 0/1 knapsack: items with integer weight w[i] and value v[i], capacity W; pick items to maximize value with
   total weight <= W. Variable x[i] = 1 when item i is picked.

The "penalty" way folds each constraint into the objective as lam * (violation)^2, the textbook QUBO recipe.
The knapsack inequality needs slack bits for that. The "exact" way keeps the objective as it is and adds each
constraint as an exact row with `Model.add_constraint`, so there is no lam and no slack.

Everything that judges an answer here (objective, feasibility, penalty energy) is plain Python on the 0/1
vector, independent of the matrices, so a test can check the matrices against it.
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, List, Tuple

import fourcolors.qubo


# ---- instances ------------------------------------------------------------------------------------

def assignment_instance(n: int, seed: int) -> List[List[int]]:
    """An n x n cost matrix of integers 1..99, seeded."""
    rnd = random.Random(seed)
    return [[rnd.randint(1, 99) for _ in range(n)] for _ in range(n)]


def knapsack_instance(n: int, seed: int, capacity_ratio: float = 0.5) -> Tuple[List[int], List[int], int]:
    """(weights, values, capacity): weights 1..50, values correlated with weight (value = weight + 1..30), and the
    capacity half of the total weight. Correlated values make the instance less trivial than random ones."""
    rnd = random.Random(seed)
    w = [rnd.randint(1, 50) for _ in range(n)]
    v = [wi + rnd.randint(1, 30) for wi in w]
    return w, v, int(capacity_ratio * sum(w))


# ---- plain-Python judges --------------------------------------------------------------------------

def assignment_cost(c, x) -> int:
    n = len(c)
    return sum(c[i][j] for i in range(n) for j in range(n) if x[i * n + j])


def assignment_feasible(n: int, x) -> bool:
    rows = all(sum(x[i * n + j] for j in range(n)) == 1 for i in range(n))
    cols = all(sum(x[i * n + j] for i in range(n)) == 1 for j in range(n))
    return rows and cols


def assignment_penalty_energy(c, lam: float, x) -> float:
    """The penalty objective written out directly: cost + lam * sum of squared row and column violations."""
    n = len(c)
    pen = sum((sum(x[i * n + j] for j in range(n)) - 1) ** 2 for i in range(n))
    pen += sum((sum(x[i * n + j] for i in range(n)) - 1) ** 2 for j in range(n))
    return assignment_cost(c, x) + lam * pen


def knapsack_value(v, x) -> int:
    return sum(vi for vi, xi in zip(v, x) if xi)


def knapsack_weight(w, x) -> int:
    return sum(wi for wi, xi in zip(w, x) if xi)


def knapsack_feasible(w, cap: int, x) -> bool:
    return knapsack_weight(w, x[:len(w)]) <= cap


def slack_bits(cap: int) -> List[int]:
    """Coefficients 1, 2, 4, ..., with the last one cut so they sum to exactly cap: every slack value 0..cap is
    reachable and none above it."""
    out, total, b = [], 0, 1
    while total + b <= cap:
        out.append(b)
        total += b
        b *= 2
    if total < cap:
        out.append(cap - total)
    return out


def knapsack_penalty_energy(w, v, cap: int, lam: float, x) -> float:
    """-value + lam * (sum w x + sum b s - cap)^2, with the slack bits s after the item bits in x."""
    bits = slack_bits(cap)
    n = len(w)
    load = knapsack_weight(w, x[:n]) + sum(b for b, s in zip(bits, x[n:]) if s)
    return -knapsack_value(v, x[:n]) + lam * (load - cap) ** 2


# ---- QUBO builders --------------------------------------------------------------------------------

def _add(Q: Dict[Tuple[int, int], float], i: int, j: int, val: float) -> None:
    key = (i, j) if i < j else (j, i)
    Q[key] = Q.get(key, 0.0) + val


def _square_penalty(Q, lin, terms: List[Tuple[int, float]], rhs: float, lam: float) -> float:
    """Add lam * (sum a_t y_t - rhs)^2 to (Q, lin), using y*y = y; return the constant lam * rhs^2."""
    for t, (i, a) in enumerate(terms):
        lin[i] += lam * (a * a - 2 * rhs * a)
        for k, b in terms[t + 1:]:
            _add(Q, i, k, 2 * lam * a * b)
    return lam * rhs * rhs


def assignment_penalty_model(c, lam: float):
    """The textbook penalty QUBO for the assignment problem: n^2 variables, no rows."""
    n = len(c)
    Q: Dict[Tuple[int, int], float] = {}
    lin = [float(c[i][j]) for i in range(n) for j in range(n)]
    offset = 0.0
    for i in range(n):
        offset += _square_penalty(Q, lin, [(i * n + j, 1.0) for j in range(n)], 1.0, lam)
    for j in range(n):
        offset += _square_penalty(Q, lin, [(i * n + j, 1.0) for i in range(n)], 1.0, lam)
    return fourcolors.qubo.to_model(Q, lin, n=n * n, offset=offset)


def assignment_exact_model(c):
    """The same problem with exact rows: the objective is the cost alone, and 2n rows say 'exactly one'."""
    n = len(c)
    m = fourcolors.qubo.to_model({}, [c[i][j] for i in range(n) for j in range(n)], n=n * n)
    for i in range(n):
        m.add_constraint({i * n + j: 1 for j in range(n)}, lo=1, hi=1)       # worker i does one job
    for j in range(n):
        m.add_constraint({i * n + j: 1 for i in range(n)}, lo=1, hi=1)       # job j gets one worker
    return m


def knapsack_penalty_model(w, v, cap: int, lam: float):
    """min -value + lam * (load + slack - cap)^2: n item bits plus len(slack_bits(cap)) slack bits."""
    n = len(w)
    bits = slack_bits(cap)
    lin = [-float(vi) for vi in v] + [0.0] * len(bits)
    Q: Dict[Tuple[int, int], float] = {}
    terms = [(i, float(wi)) for i, wi in enumerate(w)] + [(n + k, float(b)) for k, b in enumerate(bits)]
    offset = _square_penalty(Q, lin, terms, float(cap), lam)
    return fourcolors.qubo.to_model(Q, lin, n=n + len(bits), offset=offset)


def knapsack_exact_model(w, v, cap: int):
    """min -value with one exact row sum w x <= cap: n variables, no slack."""
    m = fourcolors.qubo.to_model({}, [-vi for vi in v], n=len(w))
    m.add_constraint({i: wi for i, wi in enumerate(w)}, hi=cap)
    return m


# ---- exact references -----------------------------------------------------------------------------

def assignment_optimum(c) -> int:
    """The minimum cost, by the Hungarian method (scipy)."""
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(c)
    return int(sum(c[i][j] for i, j in zip(rows, cols)))


def knapsack_optimum(w, v, cap: int) -> int:
    """The maximum value, by dynamic programming over capacity (exact for integer weights)."""
    best = [0] * (cap + 1)
    for wi, vi in zip(w, v):
        for r in range(cap, wi - 1, -1):
            best[r] = max(best[r], best[r - wi] + vi)
    return best[cap]


def assignment_brute_force(c) -> int:
    """Every permutation; for tests at n <= 7."""
    n = len(c)
    return min(sum(c[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))


def knapsack_brute_force(w, v, cap: int) -> int:
    """Every subset; for tests at n <= 16."""
    n = len(w)
    best = 0
    for mask in range(1 << n):
        x = [(mask >> i) & 1 for i in range(n)]
        if knapsack_weight(w, x) <= cap:
            best = max(best, knapsack_value(v, x))
    return best


# ---- one place that turns a result into a row -----------------------------------------------------

def judge(problem: str, inst, x, *, lam: float | None = None) -> dict:
    """Plain-Python verdict on a returned 0/1 vector (or None): feasibility, original objective, penalty energy."""
    if x is None:
        return {"feasible": False, "objective": None, "penalty_energy_python": None}
    if problem == "assignment":
        c = inst
        out = {"feasible": assignment_feasible(len(c), x), "objective": assignment_cost(c, x)}
        if lam is not None:
            out["penalty_energy_python"] = assignment_penalty_energy(c, lam, x)
        return out
    w, v, cap = inst
    out = {"feasible": knapsack_feasible(w, cap, x), "objective": knapsack_value(v, x[:len(w)]),
           "weight": knapsack_weight(w, x[:len(w)]), "capacity": cap}
    if lam is not None:
        out["penalty_energy_python"] = knapsack_penalty_energy(w, v, cap, lam, x)
    return out


def instance(problem: str, n: int, seed: int):
    return assignment_instance(n, seed) if problem == "assignment" else knapsack_instance(n, seed)


def optimum(problem: str, inst) -> int:
    """Assignment: minimum cost. Knapsack: maximum value."""
    return assignment_optimum(inst) if problem == "assignment" else knapsack_optimum(*inst)


def lam_scale(problem: str, inst) -> float:
    """The unit lam is measured in: the largest single cost (assignment) or value (knapsack)."""
    return float(max(max(r) for r in inst)) if problem == "assignment" else float(max(inst[1]))


def gap(problem: str, obj, opt) -> float | None:
    """Relative gap to the optimum, >= 0 for a feasible answer (cost above, or value below, the optimum)."""
    if obj is None:
        return None
    return (obj - opt) / opt if problem == "assignment" else (opt - obj) / opt
