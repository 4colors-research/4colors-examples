"""Open-source settings on the SAME penalty QUBOs (open-source QUBO solvers take no rows), run locally on pinned
cores of the service's CPU class, scored with structure.score.

    python rivals.py --out runs/structure-rivals --problems knapsack numpart --budgets 1 10 60 --seeds 1 2 3 \
        --settings neal neal-fill openjij openjij-fill tabu mqlib-burer mqlib-hh --mqlib /path/to/MQLib/bin/MQLib

Settings reuse the showdown post's reader harness (../qubo-solver-showdown/solvers.py). MQLib runs on number
partitioning in its Max-Cut form; the knapsack's QUBO-file path is not wired [CHECK]. CPU simulated bifurcation
has no harness yet [CHECK: package, version and license]; asking for it stops the run. A rival whose wall time ran
past 1.1 x the budget is flagged `over_budget` and is not a same-time cell.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(next(iter(sorted(__import__("pathlib").Path(__file__).resolve().parents[1].glob("*-qubo-solver-showdown"))))))

import numpy as np

import examplekit as bk
import structure as S

SETTINGS = ["mqlib-hh", "mqlib-burer", "neal", "neal-fill", "openjij", "openjij-fill", "tabu", "sb"]
OVER = 1.10


def penalty_arrays(inst):
    """(vartype, n, i, j, c, linear, offset) of the penalty QUBO, integers."""
    if inst["problem"] == "knapsack":
        Q, lin, off, n, _ = S.knapsack_penalty_qubo(inst)
        vt = "BINARY"
    else:
        Q, off = S.numpart_ising(inst)
        n, lin, vt = inst["n"], [0] * inst["n"], "SPIN"
    keys = sorted(Q)
    i = np.array([k[0] for k in keys], dtype=np.int64)
    j = np.array([k[1] for k in keys], dtype=np.int64)
    c = np.array([Q[k] for k in keys], dtype=np.int64)
    return vt, n, i, j, c, np.asarray(lin, dtype=np.int64), int(off)


def bqm_of(arr):
    import dimod

    vt, n, i, j, c, lin, off = arr
    return dimod.BinaryQuadraticModel.from_numpy_vectors(lin.astype(float), (i, j, c.astype(float)), float(off), vt)


def best_of(ss, n, vt):
    rec = ss.record
    k = int(np.argmin(rec.energy))
    x = np.ones(n, dtype=np.int64) if vt == "SPIN" else np.zeros(n, dtype=np.int64)
    x[np.asarray(list(ss.variables), dtype=np.int64)] = rec.sample[k]
    return x


def run_setting(name, inst, arr, T, seed, *, bqm, mqlib=None):
    import solvers as SV

    vt, n, i, j, c, lin, off = arr
    if name.startswith("mqlib"):
        if inst["problem"] != "numpart":
            raise ValueError("MQLib is wired for number partitioning (Max-Cut form) only [CHECK]")
        mc = {"name": inst["name"], "kind": "maxcut", "n": n, "i": i, "j": j, "c": c, "sense": "max",
              "total_weight": int(c.sum())}
        return SV.run_mqlib(mc, T, seed, binary=mqlib, hh=name == "mqlib-hh")
    if name in ("neal", "neal-fill", "openjij", "openjij-fill"):
        out = SV.run_annealer({"n": n}, T, seed, name=name, bqm=bqm)
        out["s"] = best_of_vector(out["s"], vt)
        return out
    if name == "tabu":
        from dwave.samplers import TabuSampler

        t0 = time.perf_counter()
        ss = TabuSampler().sample(bqm, num_reads=1, timeout=max(1, int(1000 * T)), seed=seed)
        return {"s": best_of(ss, n, vt), "wall_s": time.perf_counter() - t0, "prep_s": 0.0,
                "reported_energy": float(np.min(ss.record.energy))}
    raise NotImplementedError(f"{name}: no harness yet [CHECK]")


def best_of_vector(s, vt):
    """The showdown harness fills unseen variables with +1; on a BINARY model its samples are 0/1 already."""
    s = np.asarray(s, dtype=np.int64)
    return s if vt == "SPIN" else (s > 0).astype(np.int64)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--problems", nargs="+", default=["knapsack", "numpart"], choices=["knapsack", "numpart"])
    p.add_argument("--sizes-knapsack", type=int, nargs="+", default=[100, 200, 500, 1000])
    p.add_argument("--sizes-numpart", type=int, nargs="+", default=[50, 100, 200, 400])
    p.add_argument("--instance-seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--budgets", type=float, nargs="+", default=[1, 10, 60])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--settings", nargs="+", default=SETTINGS[:-1], choices=SETTINGS)
    p.add_argument("--mqlib", default=None)
    a = p.parse_args(argv)
    if "sb" in a.settings:
        raise SystemExit("sb has no harness yet [CHECK]; add it before the prereg")
    res = bk.Results(Path(a.out) / "results.jsonl")
    for problem in a.problems:
        for n in (a.sizes_knapsack if problem == "knapsack" else a.sizes_numpart):
            for iseed in a.instance_seeds:
                inst = S.build(problem, n, iseed)
                ref = S.reference(inst)
                arr = penalty_arrays(inst)
                bqm = bqm_of(arr)
                for T in a.budgets:
                    for name in a.settings:
                        if name.startswith("mqlib") and problem != "numpart":
                            continue
                        for seed in a.seeds:
                            common = dict(kind="solve", arm=name, problem=problem, instance=inst["name"], n=n,
                                          instance_seed=iseed, n_vars=arr[1], budget_s=T, solver_seed=seed)
                            try:
                                o = run_setting(name, inst, arr, T, seed, bqm=bqm, mqlib=a.mqlib)
                            except Exception as exc:          # recorded, never dropped
                                res.add(**common, error=f"{type(exc).__name__}: {exc}"[:500])
                                continue
                            wall = o["wall_s"] + o.get("prep_s", 0.0)
                            res.add(**common, **S.score(inst, ref, o["s"].tolist()), wall_s=round(wall, 3),
                                    over_budget=wall > OVER * T, reported_energy=o.get("reported_energy"))
    return res.rows()


if __name__ == "__main__":
    main()
