"""Script A: the penalty-weight sweep every QUBO tutorial asks you to do by hand.

For each problem, each lam (a multiple of the largest cost or value) and each seed, solve the penalty QUBO with the
same time limit, then judge the answer in plain Python: is it feasible, and what is its original objective?

    python sweep_penalty.py --out runs/penalty --time-limit 5 --seeds 1 2 3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors.qubo

import examplekit
import constrained_problems as cp

FACTORS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]


def builders(problem, inst, lam):
    if problem == "assignment":
        return cp.assignment_penalty_model(inst, lam)
    return cp.knapsack_penalty_model(*inst, lam)


def main(argv=None):
    p = examplekit.base_parser(__doc__)
    p.add_argument("--problems", nargs="+", default=["assignment", "knapsack"], choices=["assignment", "knapsack"])
    p.add_argument("--assignment-n", type=int, default=12, help="workers = jobs (default %(default)s)")
    p.add_argument("--knapsack-n", type=int, default=40, help="items (default %(default)s)")
    p.add_argument("--instance-seed", type=int, default=7)
    p.add_argument("--factors", type=float, nargs="+", default=FACTORS, help="lam = factor x largest cost/value")
    a = p.parse_args(argv)

    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()
    for problem in a.problems:
        n = a.assignment_n if problem == "assignment" else a.knapsack_n
        inst = cp.instance(problem, n, a.instance_seed)
        opt = cp.optimum(problem, inst)
        out.add(kind="reference", problem=problem, size=n, instance_seed=a.instance_seed, optimum=opt,
                method="hungarian" if problem == "assignment" else "dynamic programming")
        unit = cp.lam_scale(problem, inst)
        for factor in a.factors:
            lam = factor * unit
            m = builders(problem, inst, lam)
            for seed in a.seeds:
                r = fourcolors.qubo.solve(m, time_limit_s=a.time_limit, seed=seed, client=client)
                x = r.solution
                verdict = cp.judge(problem, inst, x, lam=lam)
                row = out.add(kind="penalty", problem=problem, size=n, instance_seed=a.instance_seed,
                              n_vars=m.n, n_quad=m.nnz, n_rows=m.m, lam_factor=factor, lam=lam, seed=seed,
                              time_limit_s=a.time_limit, optimum=opt,
                              gap=cp.gap(problem, verdict["objective"], opt) if verdict["feasible"] else None,
                              model_energy_python=None if x is None else m.energy(x),
                              **verdict, **examplekit.result_fields(r))
                print(f"{problem:10s} lam={factor:<6g} seed={seed} feasible={row['feasible']} "
                      f"objective={row['objective']} (optimum {opt})")


if __name__ == "__main__":
    main()
