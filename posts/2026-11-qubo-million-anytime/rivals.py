"""The open-source rival settings of the anytime study, on top of the comparison post's wrappers.

From `qubo-solver-showdown/solvers.py` (unchanged): mqlib-hh, mqlib-burer, neal, neal-fill, openjij, openjij-fill.
Added here:
    tabu      dwave-samplers TabuSampler, one read, its timeout set to fill the budget
    sb-fill   ballistic simulated bifurcation (bSB, Goto et al., Sci. Adv. 7, eabe7953, 2021) on CPU, our own
              sparse numpy implementation of the published update rule (below), its step count calibrated to
              end at the budget. The open-source simulated-bifurcation package holds the coupling matrix dense,
              which at a million variables would be 10^12 entries, so it cannot run this study's sizes.

Clock rule for every rival: the wall time counts from the start of building the solver's input from the instance
arrays (`total_s` = `prep_s` + `wall_s`), so slow input paths count, as they do for a user. Calibration pilots for
the "-fill" settings run before the clock starts and are recorded.

`streams` > 1 runs that many independent seeds of one setting in parallel processes (one core each) and keeps the
best: the way to give a single-threaded rival the same core count as the API job used.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.append(str(next(iter(sorted(__import__("pathlib").Path(__file__).resolve().parents[1].glob("*-qubo-solver-showdown"))))))

import sys as _sys                           # noqa: E402
from pathlib import Path as _Path             # noqa: E402
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import solvers as S            # noqa: E402

SHOWDOWN = ["mqlib-hh", "mqlib-burer", "neal", "neal-fill", "openjij", "openjij-fill"]
EXTRA = ["tabu", "sb-fill"]
SETTINGS = SHOWDOWN + EXTRA


# ---- Tabu -----------------------------------------------------------------------------------------

def run_tabu(inst, T, seed):
    from dwave.samplers import TabuSampler

    t0 = time.perf_counter()
    bqm = S.bqm_of(inst)
    prep = time.perf_counter() - t0
    left_ms = max(1, int(1000 * (0.95 * T - prep)))
    t0 = time.perf_counter()
    ss = TabuSampler().sample(bqm, num_reads=1, timeout=left_ms, seed=seed)
    wall = time.perf_counter() - t0
    return {"s": S.best_sample(ss, inst["n"]), "wall_s": wall, "prep_s": prep,
            "reported_energy": float(np.min(ss.record.energy)), "tabu_timeout_ms": left_ms}


# ---- ballistic simulated bifurcation, sparse, CPU -------------------------------------------------
#
# Ising form of Goto et al. 2021: minimize H(s) = -1/2 s'Js. Our instances are H(s) = sum_k c_k s_i s_j, so
# J_ij = J_ji = -c_k. bSB with x, y in R^n, a0 = 1, a(t) rising linearly from 0 to a0:
#     y <- y + dt * ( -(a0 - a(t)) x + c0 J x )
#     x <- x + dt * a0 * y
#     where |x_i| > 1: x_i <- sign(x_i), y_i <- 0
# and the answer is s = sign(x). c0 = 0.5 / (sigma_J sqrt(n)) with sigma_J the root mean square of J's entries
# over all n^2 positions, as in the paper; dt = 1.25.

def sb_matrix(inst):
    import scipy.sparse as sp

    n = inst["n"]
    i, j, c = inst["i"], inst["j"], -np.asarray(inst["c"], dtype=np.float64)
    J = sp.coo_matrix((np.concatenate([c, c]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(n, n))
    return J.tocsr()


def sb_c0(J, n: int) -> float:
    rms = np.sqrt((J.data ** 2).sum() / (n * n)) if J.nnz else 1.0
    return 0.5 / (rms * np.sqrt(n))


def bsb(J, steps: int, seed: int, *, dt: float = 1.25, a0: float = 1.0, c0=None):
    """One bSB run of `steps` steps; returns the spins sign(x) (+1 where x == 0)."""
    n = J.shape[0]
    c0 = sb_c0(J, n) if c0 is None else c0
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.1, 0.1, n)
    y = rng.uniform(-0.1, 0.1, n)
    for k in range(steps):
        a = a0 * k / max(1, steps - 1)
        y += dt * (-(a0 - a) * x + c0 * (J @ x))
        x += dt * a0 * y
        out = np.abs(x) > 1
        if out.any():
            x[out] = np.sign(x[out])
            y[out] = 0.0
    return np.where(x >= 0, 1, -1).astype(np.int64)


def calibrate_sb(J, T, *, prep_s: float, seed: int = 10 ** 6) -> dict:
    """Seconds per step from pilots that double until one takes 5% of the budget (10-250 ms), outside the clock."""
    target = min(0.25, max(0.01, 0.05 * T))
    k, pilots = 4, []
    while True:
        t0 = time.perf_counter()
        bsb(J, k, seed)
        pilots.append((k, time.perf_counter() - t0))
        if pilots[-1][1] >= target or k >= 1 << 22:
            break
        k *= 2
    per = pilots[-1][1] / pilots[-1][0]
    steps = int(max(10, 0.95 * (T - prep_s) / per))
    return {"per_step_s": per, "steps": steps, "pilots": pilots, "budget_reachable": prep_s + 10 * per < T}


def run_sb(inst, T, seed, *, calibration=None):
    t0 = time.perf_counter()
    J = sb_matrix(inst)
    prep = time.perf_counter() - t0
    cal = calibration or calibrate_sb(J, T, prep_s=prep)
    t0 = time.perf_counter()
    s = bsb(J, cal["steps"], seed)
    wall = time.perf_counter() - t0
    return {"s": s, "wall_s": wall, "prep_s": prep, "reported_energy": None, "sb_steps": cal["steps"],
            "calibration": cal}


# ---- one entry point ------------------------------------------------------------------------------

def run_one(setting: str, inst, T: float, seed: int, *, mqlib=None, calibration=None) -> dict:
    """One rival run; the clock is prep + solve (see the module docstring)."""
    if setting == "tabu":
        out = run_tabu(inst, T, seed)
    elif setting == "sb-fill":
        out = run_sb(inst, T, seed, calibration=calibration)
    elif setting.startswith("mqlib"):
        if not mqlib:
            raise RuntimeError("no MQLib executable: pass --mqlib or set MQLIB_BIN")
        out = S.run_mqlib(inst, T, seed, binary=mqlib, hh=setting == "mqlib-hh")
    elif setting in ("neal", "neal-fill", "openjij", "openjij-fill"):
        t0 = time.perf_counter()
        bqm = S.bqm_of(inst)
        prep = time.perf_counter() - t0
        out = S.run_annealer(inst, T, seed, name=setting, bqm=bqm, calibration=calibration)
        out["prep_s"] = prep
    else:
        raise ValueError(f"unknown rival setting {setting!r}")
    out["total_s"] = float(out.get("prep_s") or 0.0) + float(out["wall_s"])
    return out


def calibrate(setting: str, inst, T: float):
    """The pilot calibration of a "-fill" setting for one (instance, budget), or None for the others."""
    if setting == "sb-fill":
        t0 = time.perf_counter()
        J = sb_matrix(inst)
        return calibrate_sb(J, T, prep_s=time.perf_counter() - t0)
    if setting in ("neal-fill", "openjij-fill"):
        return S.calibrate(setting, S.bqm_of(inst), T, seed=0)
    return None


def _worker(args):
    setting, inst, T, seed, mqlib, calibration, cpu = args
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    return run_one(setting, inst, T, seed, mqlib=mqlib, calibration=calibration)


def run_streams(setting: str, inst, T: float, seed: int, *, streams: int = 1, cpus=None, mqlib=None,
                calibration=None, energy=None) -> dict:
    """`streams` independent runs (seeds seed*1000 + k) in parallel, one core each; the lowest-energy one is kept.

    `energy(s)` scores each stream's answer with the study's one evaluator. With streams == 1 this is run_one."""
    if streams <= 1:
        return run_one(setting, inst, T, seed, mqlib=mqlib, calibration=calibration)
    cpus = list(cpus or [None] * streams)[:streams]
    jobs = [(setting, inst, T, seed * 1000 + k, mqlib, calibration, cpus[k] if k < len(cpus) else None)
            for k in range(streams)]
    with ProcessPoolExecutor(max_workers=streams) as ex:
        t0 = time.perf_counter()
        outs = list(ex.map(_worker, jobs))
        span = time.perf_counter() - t0
    scored = [(energy(o["s"]) if energy is not None else 0, k) for k, o in enumerate(outs)]
    best = outs[min(scored)[1]]
    best = dict(best, streams=streams, stream_total_s=[o["total_s"] for o in outs], span_s=span,
                total_s=max(o["total_s"] for o in outs))
    return best
