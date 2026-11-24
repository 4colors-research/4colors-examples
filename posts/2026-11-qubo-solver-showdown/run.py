"""Run 4colors (through the API) and local solvers on the same instances and budgets; re-score everything.

    python run.py --out results/try --instance ea3d:L=20,seed=1 regular:n=10000,d=3,seed=1 \
        --budgets 1 3 10 --seeds 0 1 2 --mqlib ~/MQLib/bin/MQLib --cpu 3

The local solvers run in this process (annealers) or in a child process (MQLib), pinned to one core
(--cpu) with one thread each. 4colors runs on the service, so its wall time includes the network and any
queueing: the rows carry the job's own run and billed seconds next to it. This is the reader's version:
the local solvers run on YOUR machine, the 4colors solver on the service's, so it is not a same-hardware
comparison (see README).

One row per (instance, solver setting, budget, seed) in <out>/results.jsonl:
    solver, setting, instance, kind, n, m, seed, budget_s, wall_s, prep_s, objective (re-evaluated, native:
    cut or energy), energy (H, re-evaluated), reported_energy, agrees, reference, reference_kind, gap,
    cpu (the affinity the solver ran with), and solver details.
A solver that fails gives a row with `error` instead of an answer; it is never dropped.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # before numpy and the annealers load: one thread each

import argparse
import platform
import time
from pathlib import Path

import numpy as np

import examplekit as bk
import instances as I
import solvers as S


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--instance", nargs="+", required=True, help="instance specs (see instances.py)")
    p.add_argument("--budgets", type=float, nargs="+", default=[1, 3, 10])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--solvers", nargs="+", default=S.SETTINGS, choices=S.SETTINGS)
    p.add_argument("--mqlib", default=os.environ.get("MQLIB_BIN"), help="path to the MQLib executable (bin/MQLib)")
    p.add_argument("--cpu", type=int, default=None, help="pin this process (and MQLib) to one core")
    p.add_argument("--max-annealer-n", type=int, default=2_000_000,
                   help="skip the Python annealers above this many variables (memory); recorded as skipped")
    return p.parse_args(argv)


def main(argv=None, client=None):
    a = parse(argv)
    if a.cpu is not None:
        os.sched_setaffinity(0, {a.cpu})
    cpu = sorted(os.sched_getaffinity(0))
    res = bk.Results(Path(a.out) / "results.jsonl")
    machine = {"machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version()}
    for spec in a.instance:
        inst = I.build(spec)
        base = dict(instance=inst["name"], kind=inst["kind"], n=inst["n"], m=int(len(inst["i"])),
                    reference=inst["reference"], reference_kind=inst["reference_kind"], spec=spec)
        print(f"{inst['name']}: n={inst['n']:,}, m={len(inst['i']):,}")
        input_id = bqm = None
        cals = {}
        for setting in a.solvers:
            if setting.startswith(("neal", "openjij")) and inst["n"] > a.max_annealer_n:
                res.add(**base, solver=setting.split("-")[0], setting=setting, skipped="n above --max-annealer-n")
                continue
            for T in a.budgets:
                for seed in a.seeds:
                    row = dict(base, solver=setting.split("-")[0], setting=setting, budget_s=T, seed=seed,
                               cpu=cpu if setting != "4colors" else "service", **machine)
                    try:
                        if setting == "4colors":
                            if input_id is None:
                                client = client if client is not None else bk.client()
                                input_id, up = S.upload_4colors(inst, client)
                                res.add(**base, event="upload", input_id=input_id, upload_wall_s=round(up, 3))
                            out = S.run_4colors(inst, T, seed, client=client, input_id=input_id)
                        elif setting.startswith("mqlib"):
                            if not a.mqlib:
                                raise RuntimeError("no MQLib executable: pass --mqlib or set MQLIB_BIN")
                            out = S.run_mqlib(inst, T, seed, binary=a.mqlib, hh=setting == "mqlib-hh")
                        else:
                            if bqm is None:
                                t0 = time.perf_counter(); bqm = S.bqm_of(inst); bqm_s = time.perf_counter() - t0
                            if setting.endswith("-fill") and (setting, T) not in cals:
                                cals[(setting, T)] = S.calibrate(setting, bqm, T, seed=0)   # outside the clock
                            out = S.run_annealer(inst, T, seed, name=setting, bqm=bqm, calibration=cals.get((setting, T)))
                            out["prep_s"] = bqm_s
                    except Exception as exc:                         # recorded, never dropped
                        res.add(**row, error=f"{type(exc).__name__}: {exc}")
                        print(f"  {setting} T={T:g} seed={seed}: ERROR {exc}")
                        continue
                    s = out.pop("s")
                    if s is not None:
                        h = I.energy(inst, s)
                        obj = I.objective(inst, s)
                        rep = out.get("reported_energy")
                        row.update(energy=h, objective=obj, gap=I.gap(inst, obj),
                                   agrees=None if rep is None else bool(abs(rep - h) < 1e-6 * max(1, abs(h))))
                    row.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
                    res.add(**row)
                    g = row.get("gap")
                    print(f"  {setting} T={T:g} seed={seed}: objective={row.get('objective')} "
                          f"gap={'-' if g is None else f'{100 * g:.3f}%'} wall={row['wall_s']:.2f}s")
    return res.rows()


if __name__ == "__main__":
    main()
