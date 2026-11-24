"""The open-source rival arm [OD-2]: dwave-samplers simulated annealing (formerly neal) on the PENALTY form.

An annealer takes a plain QUBO, so the one-hot rows become the paper's penalty lambda * (1 - sum_j q[i, j])^2 per
car. `safe_lambda` picks a lambda no car can beat by taking no route or two: it exceeds the largest increase in
congestion that adding any single candidate route can cause. One anneal with its sweep count calibrated so it ends
at the budget ("fill"). Wall times are comparable only on booked, timed cores (see the post).
"""
from __future__ import annotations

import time
from collections import defaultdict

import traffic

PROBE_SWEEPS = 20


def safe_lambda(routes) -> int:
    """1 + max over candidate routes of sum over its segments of (2 x candidates on that segment + 1)."""
    cand = defaultdict(int)
    for rs in routes:
        for r in rs:
            for e in r:
                cand[e] += 1
    return 1 + max((sum(2 * cand[e] + 1 for e in r) for rs in routes for r in rs), default=0)


def penalty_bqm(routes, lam: int):
    import dimod

    linear, quad = traffic.qubo_terms(routes)
    bqm = dimod.BQM(dimod.BINARY)
    for k, v in enumerate(linear):
        bqm.add_variable(k, float(v))
    for (a, b), v in quad.items():
        bqm.add_interaction(a, b, float(v))
    k = 0
    for rs in routes:                             # lam * (1 - sum q)^2 = lam - lam * sum q + 2 lam sum_{a<b} q_a q_b
        ks = list(range(k, k + len(rs)))
        bqm.offset += lam
        for a in ks:
            bqm.add_linear(a, -float(lam))
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                bqm.add_interaction(ks[x], ks[y], 2.0 * lam)
        k += len(rs)
    return bqm


def calibrated_sweeps(seconds_per_sweep: float, budget_s: float, *, floor: int = 1) -> int:
    if seconds_per_sweep <= 0:
        return floor
    return max(floor, int(budget_s / seconds_per_sweep))


def anneal(routes, budget_s: float, seed: int):
    """Best 0/1 sample and record fields of one calibrated anneal on the penalty form."""
    import numpy as np
    from dwave.samplers import SimulatedAnnealingSampler

    lam = safe_lambda(routes)
    bqm = penalty_bqm(routes, lam)
    smp = SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    smp.sample(bqm, num_reads=1, num_sweeps=PROBE_SWEEPS, seed=seed)
    per = (time.perf_counter() - t0) / PROBE_SWEEPS
    sweeps = calibrated_sweeps(per, budget_s)
    t0 = time.perf_counter()
    ss = smp.sample(bqm, num_reads=1, num_sweeps=sweeps, seed=seed)
    wall = time.perf_counter() - t0
    rec = ss.record[int(np.argmin(ss.record.energy))]
    x = [0] * len(bqm.variables)
    for k, v in enumerate(ss.variables):
        x[int(v)] = int(rec.sample[k])
    return x, {"lambda": lam, "num_sweeps": sweeps, "rival_wall_s": round(wall, 3)}
