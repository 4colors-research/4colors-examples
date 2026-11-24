"""The anytime study: million-variable models, the 4colors API against open-source rivals at equal budgets.

    python run.py --out results/try --instance ea3d:L=20,seed=1 --arms api neal-fill --budgets 1 3 --seeds 1
    python run.py --out $RUN/out --instance ea3d:L=100,seed=1 ... --arms api neal-fill tabu sb-fill \
        --budgets 1 3 10 30 60 --seeds 1 2 3 4 5 6 7 8 9 10 --cpus 4 5 --streams 2

Arms:
    api        the 4colors Optimizer for QUBO through the public client: each instance is uploaded once, every job
               names it by id (`input_id`), one job at a time. Rows keep the service's version, search time,
               billed seconds and whatever it reports about its threads, plus the client wall time from submit to
               the final answer and to the first answer received.
    <rival>    one of rivals.SETTINGS, run in this process (MQLib in a child), pinned to --cpus, with --streams
               parallel seeds when the API job used more than one thread (read it from a smoke run first).

The cells are interleaved: for every (instance, budget, seed) the arms run back to back, in an order drawn from
--order-seed, so drift on the machine hits every arm alike. Every answer is re-scored here with one evaluator
(families.score). `clock_s` is the arm's clock: the client wall time from submit to answer for the API (the
model is already uploaded), prep + solve for a rival. Rows go to <out>/results.jsonl: kind = instance | upload | solve | error.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy loads: one thread per rival stream

import argparse                              # noqa: E402
import platform                              # noqa: E402
import random                                # noqa: E402
import time                                  # noqa: E402
from pathlib import Path                     # noqa: E402

import numpy as np                           # noqa: E402

import sys as _sys                           # noqa: E402
from pathlib import Path as _Path             # noqa: E402
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import examplekit as bk                         # noqa: E402
import families as F                         # noqa: E402
import rivals as R                           # noqa: E402

ARMS = ["api"] + R.SETTINGS
OVERRUN = 1.10                               # a wall time above 110% of the budget is flagged "overran"


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--instance", nargs="+", required=True, help="family specs (see families.py)")
    p.add_argument("--arms", nargs="+", default=["api"], choices=ARMS)
    p.add_argument("--budgets", type=float, nargs="+", default=[1, 3, 10, 30, 60])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--cpus", type=int, nargs="*", default=None, help="cores for the rivals (one per stream)")
    p.add_argument("--streams", type=int, default=1, help="parallel rival seeds per run (match the API's threads)")
    p.add_argument("--mqlib", default=os.environ.get("MQLIB_BIN"), help="path to MQLib's bin/MQLib")
    p.add_argument("--order-seed", type=int, default=0, help="seed of the arm order inside each cell")
    p.add_argument("--inline", action="store_true", help="send the model with each job instead of uploading once")
    return p.parse_args(argv)


def api_model(inst):
    import scipy.sparse as sp
    import fourcolors.qubo

    n = inst["n"]
    return fourcolors.qubo.to_model(sp.coo_matrix((inst["c"], (inst["i"], inst["j"])), shape=(n, n)),
                                    vartype="SPIN")


def threads_reported(r):
    """What the service says about the threads a job used, wherever it says it (None if it does not)."""
    for d in (getattr(r, "timings", None) or {}, getattr(r, "engine", None) or {}):
        for k in ("threads", "num_threads", "cores"):
            if k in d:
                return d[k]
    return None


def run_api(model_or_none, input_id, T, seed, client):
    import fourcolors.qubo

    first = []
    t0 = time.perf_counter()

    def seen(best):
        if not first and best.solution is not None:
            first.append(time.perf_counter() - t0)

    if input_id is not None:
        r = fourcolors.qubo.solve(input_id=input_id, time_limit_s=T, seed=seed, client=client, on_incumbent=seen)
    else:
        r = fourcolors.qubo.solve(model_or_none, time_limit_s=T, seed=seed, client=client, on_incumbent=seen)
    wall = time.perf_counter() - t0
    timings = dict(getattr(r, "timings", None) or {})
    traj = getattr(r, "trajectory", None) or []
    return {"s": None if r.solution is None else np.asarray(r.solution, dtype=np.int64),
            "reported_energy": r.objective, "client_wall_s": wall,
            "client_first_answer_s": first[0] if first else None,
            "search_first_answer_s": traj[0].get("t_search_s") if traj else None,
            "search_s": timings.get("search_s", timings.get("run_s")), "timings": timings,
            "threads": threads_reported(r), **bk.result_fields(r)}


def main(argv=None, client=None):
    a = parse(argv)
    res = bk.Results(Path(a.out) / "results.jsonl")
    order_rng = random.Random(a.order_seed)
    host = {"machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(),
            "rival_cpus": a.cpus, "streams": a.streams}
    if a.cpus:
        os.sched_setaffinity(0, set(a.cpus))
    for spec in a.instance:
        t0 = time.perf_counter()
        inst = F.build(spec)
        base = dict(instance=inst["name"], family=inst["family"], sha256=inst["sha256"], n=inst["n"],
                    m=int(len(inst["i"])), reference=inst["reference"], reference_kind=inst["reference_kind"],
                    spec=spec)
        res.add(kind="instance", **base, build_s=round(time.perf_counter() - t0, 3),
                reference_alt=inst.get("reference_alt"), reference_alt_kind=inst.get("reference_alt_kind"))
        print(f"{inst['name']}: n={inst['n']:,}, m={len(inst['i']):,}, sha256 {inst['sha256'][:16]}")
        input_id, model, cals = None, None, {}
        if "api" in a.arms:
            client = client if client is not None else bk.client()
            model = api_model(inst)
            if not a.inline:
                import fourcolors.qubo
                t0 = time.perf_counter()
                input_id = fourcolors.qubo.upload(model, client=client)
                res.add(kind="upload", **base, input_id=input_id, upload_wall_s=round(time.perf_counter() - t0, 3))
        for T in a.budgets:
            for setting in a.arms:                       # calibration pilots: before any clock, recorded
                if setting != "api" and setting.endswith("-fill") and (setting, T) not in cals:
                    try:
                        cals[(setting, T)] = R.calibrate(setting, inst, T)
                    except Exception as exc:
                        res.add(kind="error", **base, arm="rival", setting=setting, budget_s=T, stage="calibrate",
                                error=f"{type(exc).__name__}: {exc}")
            for seed in a.seeds:
                arms = list(a.arms)
                order_rng.shuffle(arms)
                for setting in arms:
                    row = dict(base, kind="solve", arm="api" if setting == "api" else "rival", setting=setting,
                               budget_s=T, seed=seed, order=arms, **host)
                    try:
                        if setting == "api":
                            out = run_api(model, input_id, T, seed, client)
                            wall = out["client_wall_s"]
                        else:
                            out = R.run_streams(setting, inst, T, seed, streams=a.streams, cpus=a.cpus,
                                                mqlib=a.mqlib, calibration=cals.get((setting, T)),
                                                energy=lambda s: F.I.energy(inst, s))
                            wall = out["total_s"]
                    except Exception as exc:                 # recorded, never dropped
                        res.add(**dict(row, kind="error", error=f"{type(exc).__name__}: {exc}"))
                        print(f"  {setting} T={T:g} seed={seed}: ERROR {exc}")
                        continue
                    s = out.pop("s")
                    if s is not None:
                        sc = F.score(inst, s)
                        rep = out.get("reported_energy")
                        row.update(sc, agrees=None if rep is None else bool(abs(rep - sc["energy"]) < 1e-6 * max(
                            1, abs(sc["energy"]))))
                    row.update(clock_s=wall, overran=bool(wall > OVERRUN * T))
                    row.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()
                                if k != "clock_s"})
                    res.add(**row)
                    g = row.get("gap")
                    print(f"  {setting:12s} T={T:g} seed={seed}: gap={'-' if g is None else f'{100 * g:.3f}%'} "
                          f"clock={wall:.2f}s{' OVERRAN' if row['overran'] else ''}")
    return res.rows()


if __name__ == "__main__":
    main()
