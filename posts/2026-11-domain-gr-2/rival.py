"""The open-source rival arm [OD-2]: dwave-samplers simulated annealing (formerly neal) on the penalty QUBO.

One anneal with its sweep count calibrated so the anneal ends at the budget ("fill"): a short probe anneal measures
seconds per sweep on this instance, and the real anneal gets budget / (seconds per sweep) sweeps. Wall times are only
comparable on booked, timed cores (see the post); an untimed run labels them untimed.
"""
from __future__ import annotations

import time

PROBE_SWEEPS = 20


def bqm_of(n: int, edges, P: int = 2):
    import dimod
    import numpy as np

    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return dimod.BQM.from_numpy_vectors(-np.ones(n), (e[:, 0], e[:, 1], np.full(len(e), float(P))), 0.0,
                                        dimod.BINARY)


def calibrated_sweeps(seconds_per_sweep: float, budget_s: float, *, floor: int = 1) -> int:
    """Sweeps that fill `budget_s` at the measured rate."""
    if seconds_per_sweep <= 0:
        return floor
    return max(floor, int(budget_s / seconds_per_sweep))


def anneal(n: int, edges, budget_s: float, seed: int, P: int = 2):
    """Best sample (list of 0/1) and the record fields of one calibrated anneal."""
    import numpy as np
    from dwave.samplers import SimulatedAnnealingSampler

    bqm = bqm_of(n, edges, P)
    smp = SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    smp.sample(bqm, num_reads=1, num_sweeps=PROBE_SWEEPS, seed=seed)
    per_sweep = (time.perf_counter() - t0) / PROBE_SWEEPS
    sweeps = calibrated_sweeps(per_sweep, budget_s)
    t0 = time.perf_counter()
    ss = smp.sample(bqm, num_reads=1, num_sweeps=sweeps, seed=seed)
    wall = time.perf_counter() - t0
    rec = ss.record[int(np.argmin(ss.record.energy))]
    order = list(ss.variables)
    x = [0] * n
    for k, v in enumerate(order):
        x[int(v)] = int(rec.sample[k])
    return x, {"num_sweeps": sweeps, "rival_wall_s": round(wall, 3), "probe_s_per_sweep": per_sweep}
