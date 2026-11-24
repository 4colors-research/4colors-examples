"""The open-source rival arms, run locally on pinned cores of the service's CPU class, on the same generated
instances (rebuilt from the same family, size and seed; the sha256 must match the 4colors rows).

    python rivals.py --out runs/dense-rivals --family reg --degrees 4 16 64 256 --instance-seeds 101 102 103 \
        --budgets 1 3 10 60 300 --solver-seeds 1 2 3 --settings neal neal-fill openjij openjij-fill tabu \
        mqlib-burer mqlib-hh --mqlib /path/to/MQLib/bin/MQLib

Settings reuse the reader harness of the showdown post (../qubo-solver-showdown/solvers.py: library defaults and
budget-filling calibration for the annealers, MQLib with its silent-failure checks). `tabu` is dwave-samplers'
TabuSampler with its timeout set to the budget. `sb` (CPU simulated bifurcation) is listed in the plan but has no
harness here yet [CHECK: package, version pin and license before the prereg]; asking for it stops the run.
Every answer is re-scored with dense.energy; a rival whose wall time ran past 1.1 x the budget is flagged
`over_budget` and is not a same-time cell.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(next(iter(sorted(__import__("pathlib").Path(__file__).resolve().parents[1].glob("*-qubo-solver-showdown"))))))   # its solvers.py and instances.py

import numpy as np

import examplekit as bk
import dense as D

SETTINGS = ["mqlib-hh", "mqlib-burer", "neal", "neal-fill", "openjij", "openjij-fill", "tabu", "sb"]
OVER = 1.10


def bqm_of(inst):
    import dimod

    lin = np.zeros(inst["n"]) if inst["linear"] is None else np.asarray(inst["linear"], dtype=float)
    return dimod.BinaryQuadraticModel.from_numpy_vectors(lin, (inst["i"], inst["j"], inst["c"]), 0.0,
                                                         inst["vartype"])


def to_vartype(inst, s):
    """Samplers on a BINARY model return 0/1; MQLib on Max-Cut returns +-1 (reg family only)."""
    s = np.asarray(s, dtype=np.int64)
    return s if inst["vartype"] == "SPIN" else (s > 0).astype(np.int64)


def run_setting(name, inst, T, seed, *, bqm=None, mqlib=None):
    import solvers as S

    if name.startswith("mqlib"):
        if inst["family"] != "reg":
            raise ValueError("MQLib runs on the Max-Cut (reg) family here; the full family's QUBO-file path "
                             "is not wired [CHECK]")
        return S.run_mqlib(inst, T, seed, binary=mqlib, hh=name == "mqlib-hh")
    if name in ("neal", "neal-fill", "openjij", "openjij-fill"):
        return S.run_annealer(inst, T, seed, name=name, bqm=bqm)
    if name == "tabu":
        from dwave.samplers import TabuSampler

        t0 = time.perf_counter()
        ss = TabuSampler().sample(bqm, num_reads=1, timeout=int(1000 * T), seed=seed)
        wall = time.perf_counter() - t0
        return {"s": S.best_sample(ss, inst["n"]) if inst["vartype"] == "SPIN" else _best_binary(ss, inst["n"]),
                "wall_s": wall, "prep_s": 0.0, "reported_energy": float(np.min(ss.record.energy))}
    raise NotImplementedError(f"{name}: no harness yet [CHECK]")


def _best_binary(ss, n):
    rec = ss.record
    k = int(np.argmin(rec.energy))
    x = np.zeros(n, dtype=np.int64)
    x[np.asarray(list(ss.variables), dtype=np.int64)] = rec.sample[k]
    return x


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--family", choices=["reg", "full"], required=True)
    p.add_argument("--n", type=int, default=D.POST_REG["n"])
    p.add_argument("--degrees", type=int, nargs="+", default=D.POST_REG["degrees"])
    p.add_argument("--sizes", type=int, nargs="+", default=D.POST_FULL["sizes"])
    p.add_argument("--instance-seeds", type=int, nargs="+", default=[101, 102, 103])
    p.add_argument("--budgets", type=float, nargs="+", default=[1, 3, 10, 60, 300])
    p.add_argument("--solver-seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--settings", nargs="+", default=SETTINGS[:-1], choices=SETTINGS)
    p.add_argument("--mqlib", default=None, help="the MQLib binary (needed for mqlib-* settings)")
    a = p.parse_args(argv)
    if "sb" in a.settings:
        raise SystemExit("sb has no harness yet [CHECK]; add it before the prereg")
    res = bk.Results(Path(a.out) / "results.jsonl")
    todo = [("reg", a.n, d) for d in a.degrees] if a.family == "reg" else [("full", n, None) for n in a.sizes]
    for family, n, d in todo:
        for iseed in a.instance_seeds:
            inst = D.build(family, n, iseed, d)
            common = dict(instance=inst["name"], family=family, n=n, degree=inst["degree"], instance_seed=iseed,
                          couplings=len(inst["i"]), instance_sha256=D.sha256(inst))
            bqm = bqm_of(inst) if any(not s.startswith("mqlib") for s in a.settings) else None
            for T in a.budgets:
                for name in a.settings:
                    for seed in a.solver_seeds:
                        try:
                            out = run_setting(name, inst, T, seed, bqm=bqm, mqlib=a.mqlib)
                        except Exception as exc:              # recorded, never dropped
                            res.add(event="error", arm=name, **common, budget_s=T, solver_seed=seed,
                                    error=f"{type(exc).__name__}: {exc}"[:500])
                            continue
                        x = to_vartype(inst, out["s"])
                        e = D.energy(inst, x)
                        wall = out["wall_s"] + out.get("prep_s", 0.0)
                        res.add(event="solve", arm=name, **common, budget_s=T, solver_seed=seed, energy=e,
                                reported_energy=out.get("reported_energy"), wall_s=round(wall, 3),
                                over_budget=wall > OVER * T, num_sweeps=out.get("num_sweeps"))
                        print(f"  {inst['name']} {name} T={T:g}s seed={seed}: energy={e}, wall {wall:.1f}s")
    return res.rows()


if __name__ == "__main__":
    main()
