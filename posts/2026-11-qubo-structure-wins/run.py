"""The 4colors arms: send each knapsack and number-partitioning instance as the textbook penalty QUBO (`penalty`), and
the knapsack also with one explicit row (`rows`), solve at every budget and seed, and score every answer with
structure.score (never the service's number alone).

    python run.py --out runs/structure --problems knapsack numpart --sizes-knapsack 100 200 500 1000 \
        --sizes-numpart 50 100 200 400 --instance-seeds 1 2 3 --budgets 1 10 60 --seeds 1 2 3

Constraint detection (QUBO 1.0.7) is a service feature: the script sends the same penalty QUBO a customer would and
records whatever the release reports about it. Which result field carries the detection report is not public yet
[CHECK at 1.0.7]; until then every result attribute outside the known set is stored under `extra_fields`, so the
record keeps it whatever its name.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors
import fourcolors.qubo

import examplekit as bk
import structure as S

KNOWN = {"status", "objective", "solution", "labels", "vartype", "reason", "engine", "job", "trajectory",
         "constraints", "billed_s"}


def extra_fields(r) -> dict:
    """Every public, JSON-able attribute of the result that examplekit.result_fields does not already record."""
    names = getattr(r, "__dataclass_fields__", None) or getattr(r, "__dict__", {})
    out = {}
    for k in names:
        if k.startswith("_") or k in KNOWN:
            continue
        v = getattr(r, k, None)
        if v is None or isinstance(v, (str, int, float, bool, list, dict)):
            out[k] = v
    return out


def models(inst):
    if inst["problem"] == "knapsack":
        return {"penalty": S.knapsack_penalty_model(inst), "rows": S.knapsack_rows_model(inst)}
    return {"penalty": S.numpart_model(inst)}


def main(argv=None):
    p = bk.base_parser(__doc__, time_limit=10.0)
    p.add_argument("--problems", nargs="+", default=["knapsack", "numpart"], choices=["knapsack", "numpart"])
    p.add_argument("--sizes-knapsack", type=int, nargs="+", default=[100, 200, 500, 1000])
    p.add_argument("--sizes-numpart", type=int, nargs="+", default=[50, 100, 200, 400])
    p.add_argument("--instance-seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--budgets", type=float, nargs="+", default=None, help="overrides --time-limit")
    p.add_argument("--arms", nargs="+", default=["penalty", "rows"], choices=["penalty", "rows"])
    a = p.parse_args(argv)
    budgets = a.budgets or [a.time_limit]
    out = bk.Results(Path(a.out) / "results.jsonl")
    client = bk.client()
    for problem in a.problems:
        sizes = a.sizes_knapsack if problem == "knapsack" else a.sizes_numpart
        for n in sizes:
            for iseed in a.instance_seeds:
                inst = S.build(problem, n, iseed)
                ref = S.reference(inst)
                out.add(kind="reference", problem=problem, instance=inst["name"], n=n, instance_seed=iseed, **ref)
                for arm, m in models(inst).items():
                    if arm not in a.arms:
                        continue
                    for T in budgets:
                        for seed in a.seeds:
                            common = dict(kind="solve", arm=f"4colors-{arm}", problem=problem, instance=inst["name"],
                                          n=n, instance_seed=iseed, n_vars=m.n, n_rows=m.m, budget_s=T,
                                          solver_seed=seed)
                            t0 = time.perf_counter()
                            try:
                                r = fourcolors.qubo.solve(m, time_limit_s=T, seed=seed, client=client)
                            except fourcolors.errors.FourColorsError as exc:
                                out.add(**common, error_code=getattr(exc, "code", None), error=str(exc)[:500])
                                continue
                            wall = time.perf_counter() - t0
                            sc = S.score(inst, ref, r.solution)
                            row = out.add(**common, wall_s=round(wall, 3), **sc,
                                          model_energy=None if r.solution is None else m.energy(r.solution),
                                          **bk.result_fields(r), extra_fields=extra_fields(r))
                            print(f"  {inst['name']} {arm:7s} T={T:g}s seed={seed}: "
                                  + (f"gap={row.get('gap')}" if problem == "knapsack" else f"diff={row.get('diff')}"))
    return out.rows()


if __name__ == "__main__":
    main()
