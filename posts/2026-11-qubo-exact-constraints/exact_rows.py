"""Script B: the same problems with exact rows instead of penalties. No lam, no slack bits.

The objective is the cost (or minus the value) alone; each constraint is one `add_constraint` row. The answer is
checked three ways: by the service (`result.constraints`), by the client (`model.feasible`), and in plain Python.

    python exact_rows.py --out runs/exact --time-limit 5 --seeds 1 2 3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors.qubo

import examplekit
import constrained_problems as cp


def main(argv=None):
    p = examplekit.base_parser(__doc__)
    p.add_argument("--problems", nargs="+", default=["assignment", "knapsack"], choices=["assignment", "knapsack"])
    p.add_argument("--assignment-n", type=int, default=12)
    p.add_argument("--knapsack-n", type=int, default=40)
    p.add_argument("--instance-seed", type=int, default=7)
    a = p.parse_args(argv)

    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()
    for problem in a.problems:
        n = a.assignment_n if problem == "assignment" else a.knapsack_n
        inst = cp.instance(problem, n, a.instance_seed)
        opt = cp.optimum(problem, inst)
        m = cp.assignment_exact_model(inst) if problem == "assignment" else cp.knapsack_exact_model(*inst)
        for seed in a.seeds:
            r = fourcolors.qubo.solve(m, time_limit_s=a.time_limit, seed=seed, client=client)
            x = r.solution
            verdict = cp.judge(problem, inst, x)
            service_ok = (r.constraints or {}).get("feasible")
            client_ok = None if x is None else m.feasible(x)
            row = out.add(kind="exact", problem=problem, size=n, instance_seed=a.instance_seed, n_vars=m.n,
                          n_quad=m.nnz, n_rows=m.m, seed=seed, time_limit_s=a.time_limit, optimum=opt,
                          gap=cp.gap(problem, verdict["objective"], opt) if verdict["feasible"] else None,
                          model_feasible=client_ok, service_feasible=service_ok,
                          checks_agree=(service_ok == client_ok == verdict["feasible"]) if x is not None else None,
                          model_energy_python=None if x is None else m.energy(x),
                          **verdict, **examplekit.result_fields(r))
            print(f"{problem:10s} exact rows seed={seed} feasible: service={service_ok} client={client_ok} "
                  f"python={row['feasible']} objective={row['objective']} (optimum {opt})")


if __name__ == "__main__":
    main()
