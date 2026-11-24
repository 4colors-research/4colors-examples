"""One wrapper per solver and setting. Each returns a dict with the answer `s` (+-1 spins in variable
order), `wall_s` (the solver call only), `prep_s` (building the solver's input, outside the clock) and
details. Nothing here scores an answer: run.py re-evaluates every one with instances.energy.

Settings:
    4colors          the 4colors Optimizer for QUBO (the live release) through the public API (upload once, solve by id)
    mqlib-hh         MQLib's hyper-heuristic (-hh), run with cwd = the MQLib folder (see pitfalls.md)
    mqlib-burer      MQLib's BURER2002 heuristic (-h BURER2002)
    neal             D-Wave neal simulated annealing, library defaults (1000 sweeps, 1 read)
    neal-fill        neal, one anneal with its sweep count calibrated to end at the budget
    openjij          OpenJij SASampler, library defaults
    openjij-fill     OpenJij, calibrated the same way
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import instances as I

SETTINGS = ["4colors", "mqlib-hh", "mqlib-burer", "neal", "neal-fill", "openjij", "openjij-fill"]
RIVALS = SETTINGS[1:]


# ---- 4colors, through the API ---------------------------------------------------------------------

def upload_4colors(inst, client):
    import scipy.sparse as sp
    import fourcolors.qubo

    n = inst["n"]
    model = fourcolors.qubo.to_model(sp.coo_matrix((inst["c"], (inst["i"], inst["j"])), shape=(n, n)), vartype="SPIN")
    t0 = time.perf_counter()
    mid = fourcolors.qubo.upload(model, client=client)
    return mid, time.perf_counter() - t0


def _client_version():
    try:
        import fourcolors
    except ImportError:
        return None
    return getattr(fourcolors, "__version__", None)


def run_4colors(inst, T, seed, *, client, input_id):
    import fourcolors.qubo

    t0 = time.perf_counter()
    r = fourcolors.qubo.solve(input_id=input_id, time_limit_s=T, seed=seed, client=client)
    wall = time.perf_counter() - t0
    job = getattr(r, "job", None)
    info = {k: (getattr(job, k, None) if job is not None else None) for k in ("id", "billed_s", "run_s", "wait_s")}
    traj = getattr(r, "trajectory", None) or []
    return {"s": None if r.solution is None else np.asarray(r.solution, dtype=np.int64), "wall_s": wall,
            "prep_s": 0.0, "reported_energy": r.objective, "status": r.status,
            "first_answer_s": traj[0].get("t_search_s") if traj else None,
            "engine_version": (getattr(r, "engine", None) or {}).get("version"), "client_version": _client_version(),
            **{"job_" + k: v for k, v in info.items()}}


# ---- MQLib ----------------------------------------------------------------------------------------

class MQLibTrap(RuntimeError):
    """MQLib answered something that cannot be its real answer (see pitfalls.md)."""


HEAD = re.compile(r'^(?P<limit>[^,]*),(?P<heur>[^,]*),"(?P<inst>.*)",(?P<obj>[^,]+),(?P<runtime>[^,]+),\[(?P<hist>.*)\]$')


def parse_mqlib(stdout: str) -> dict:
    """The first line (limit, heuristic, instance, best objective, runtime, [objective:time;...]) and the
    line after "Solution:" (+-1 per node for a Max-Cut file)."""
    lines = [ln.strip() for ln in stdout.splitlines()]
    head = next(ln for ln in lines if ln)
    m = HEAD.match(head)
    if not m:
        raise ValueError(f"unexpected MQLib output: {head[:200]!r}")
    hist = [(float(o), float(t)) for o, t in (h.split(":") for h in m["hist"].split(";") if h)]
    k = lines.index("Solution:")
    sol = next(ln for ln in lines[k + 1:] if ln)
    s = np.array([int(float(x)) for x in sol.split()], dtype=np.int64)
    return {"heuristic": m["heur"], "objective": float(m["obj"]), "runtime_s": float(m["runtime"]),
            "history": hist, "s": s}


def mqlib_root(binary) -> Path:
    """MQLib's own folder (the parent of bin/): -hh reads hhdata/ by a relative path from the cwd."""
    return Path(binary).resolve().parent.parent


def check_mqlib(parsed: dict, inst: dict, hh: bool) -> None:
    """Catch the silent failure: -hh outside the MQLib folder picks no heuristic ("HH_") and returns an
    all-equal answer with objective 0. Also refuse an answer whose cut is not the one MQLib printed."""
    s = parsed["s"]
    if len(s) != inst["n"] or not np.isin(s, (-1, 1)).all():
        raise MQLibTrap(f"MQLib returned {len(s)} values for {inst['n']} nodes")
    if hh and parsed["heuristic"] in ("HH_", ""):
        raise MQLibTrap("the hyper-heuristic chose no heuristic (run it with cwd = the MQLib folder)")
    cut = (int(np.sum(inst["c"])) - I.energy(inst, s)) // 2      # the cut MQLib sees, weights c
    if abs(cut - parsed["objective"]) > 1e-6 * max(1, abs(cut)):
        raise MQLibTrap(f"MQLib printed objective {parsed['objective']}, its answer cuts {cut}")


def run_mqlib(inst, T, seed, *, binary, hh: bool, workdir=None):
    root = mqlib_root(binary)
    if hh and not (root / "hhdata").is_dir():
        raise MQLibTrap(f"{root} has no hhdata/ folder; -hh would silently return an all-zero answer")
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        path = Path(tmp) / f"{inst['name']}.txt"
        t0 = time.perf_counter()
        I.write_maxcut_file(inst, path)
        prep = time.perf_counter() - t0
        heur = ["-hh"] if hh else ["-h", "BURER2002"]
        cmd = [str(Path(binary).resolve()), "-fM", str(path), *heur, "-r", f"{T:g}", "-ps", "-s", str(seed)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"MQLib exited {p.returncode}: {p.stderr[-500:]}")
    parsed = parse_mqlib(p.stdout)
    check_mqlib(parsed, inst, hh)
    # MQLib reports a cut; for H: H = W - 2 cut with W = Σ c
    rep_h = int(round(int(np.sum(inst["c"])) - 2 * parsed["objective"]))
    return {"s": parsed["s"], "wall_s": wall, "prep_s": prep, "reported_energy": rep_h,
            "heuristic": parsed["heuristic"], "solver_runtime_s": parsed["runtime_s"],
            "first_answer_s": parsed["history"][0][1] if parsed["history"] else None,
            "improvements": len(parsed["history"])}


# ---- annealers ------------------------------------------------------------------------------------

def bqm_of(inst):
    import dimod

    return dimod.BinaryQuadraticModel.from_numpy_vectors(np.zeros(inst["n"]), (inst["i"], inst["j"], inst["c"]),
                                                         0.0, "SPIN")


def sampler(name: str):
    if name.startswith("neal"):
        from dwave.samplers import SimulatedAnnealingSampler
        return SimulatedAnnealingSampler()
    import openjij
    return openjij.SASampler()


def best_sample(ss, n: int) -> np.ndarray:
    """The lowest-energy sample as +-1 spins in variable order 0..n-1."""
    rec = ss.record
    k = int(np.argmin(rec.energy))
    cols = np.asarray(list(ss.variables), dtype=np.int64)
    s = np.ones(n, dtype=np.int64)               # a variable the sampler never saw cannot occur here
    s[cols] = rec.sample[k]
    return s


def calibrate(name, bqm, T, *, seed, start: int = 8, max_sweeps: int = 1 << 22) -> dict:
    """Seconds per sweep and fixed overhead from short pilot anneals (outside the timed run), then the
    sweep count that should end one anneal at about 95% of the budget.

    The pilot doubles its sweeps until one pilot takes a measurable time (5% of the budget, between 10 ms
    and 250 ms), so timer noise on a small instance cannot make the per-sweep cost look near zero (which
    would ask for billions of sweeps). The per-sweep cost is the slope of the last two pilots, but never
    below half their average cost per sweep; with a single pilot it is that pilot's average (overhead
    included, so the anneal ends early rather than late)."""
    smp = sampler(name)
    target = min(0.25, max(0.01, 0.05 * T))
    pilots = []
    k = start
    while True:
        t0 = time.perf_counter()
        smp.sample(bqm, num_reads=1, num_sweeps=k, seed=seed + 10 ** 6)
        pilots.append((k, time.perf_counter() - t0))
        if pilots[-1][1] >= target or k >= max_sweeps:
            break
        k *= 2
    k2, t2 = pilots[-1]
    if len(pilots) >= 2:
        k1, t1 = pilots[-2]
        per = max((t2 - t1) / (k2 - k1), 0.5 * t2 / k2)
    else:
        per = t2 / k2
    over = max(0.0, t2 - per * k2)
    sweeps = int(min(max_sweeps, max(10, 0.95 * (T - over) / per)))
    return {"per_sweep_s": per, "overhead_s": over, "num_sweeps": sweeps, "pilots": pilots,
            "budget_reachable": over + 10 * per < T}


def run_annealer(inst, T, seed, *, name: str, bqm=None, calibration=None):
    t0 = time.perf_counter()
    bqm = bqm if bqm is not None else bqm_of(inst)
    prep = time.perf_counter() - t0
    smp = sampler(name)
    kw = {"seed": seed}
    if name.endswith("-fill"):
        cal = calibration or calibrate(name, bqm, T, seed=seed)
        kw.update(num_reads=1, num_sweeps=cal["num_sweeps"])
    else:
        cal = None                                   # library defaults: no num_reads, no num_sweeps
    t0 = time.perf_counter()
    ss = smp.sample(bqm, **kw)
    wall = time.perf_counter() - t0
    s = best_sample(ss, inst["n"])
    return {"s": s, "wall_s": wall, "prep_s": prep, "reported_energy": float(np.min(ss.record.energy)),
            "num_sweeps": kw.get("num_sweeps"), "calibration": cal}
