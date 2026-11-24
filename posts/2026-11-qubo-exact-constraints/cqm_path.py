"""Script C: the same two problems as dimod ConstrainedQuadraticModels, through FourColorsSampler.

If your code already builds a CQM with dimod, you do not rewrite it: `FourColorsSampler().sample(cqm, ...)` takes a
CQM whose variables are all BINARY and whose constraints are linear and hard, and sends it with the constraints as
exact rows. The answer is checked three ways, as in script B: the service's exact check (`info["constraints"]`, the
authoritative verdict), dimod's own float check (`sampleset.first.is_feasible`, with a tolerance), and plain Python.

    python cqm_path.py --out runs/cqm --time-limit 1 --seeds 1 2 3

Needs `pip install "fourcolors[ocean]"` (dimod).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import examplekit
import constrained_problems as cp


def knapsack_cqm(w, v, cap):
    """max sum v x  s.t.  sum w x <= cap, written the dimod way (dimod minimizes, so the objective is -value)."""
    import dimod

    x = [dimod.Binary(f"x{i}") for i in range(len(w))]
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(-dimod.quicksum(vi * xi for vi, xi in zip(v, x)))
    cqm.add_constraint(dimod.quicksum(wi * xi for wi, xi in zip(w, x)) <= cap, label="capacity")
    return cqm


def assignment_cqm(c):
    """min sum c x  s.t.  every worker one job, every job one worker. Variable x_i_j = worker i does job j."""
    import dimod

    n = len(c)
    x = {(i, j): dimod.Binary(f"x_{i}_{j}") for i in range(n) for j in range(n)}
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(dimod.quicksum(c[i][j] * x[i, j] for i in range(n) for j in range(n)))
    for i in range(n):
        cqm.add_constraint(dimod.quicksum(x[i, j] for j in range(n)) == 1, label=f"worker_{i}")
    for j in range(n):
        cqm.add_constraint(dimod.quicksum(x[i, j] for i in range(n)) == 1, label=f"job_{j}")
    return cqm


def vector(problem, n, sample):
    """The sample (a {label: 0/1} mapping) as the 0/1 vector the plain-Python judges read."""
    if problem == "assignment":
        return [int(sample[f"x_{i}_{j}"]) for i in range(n) for j in range(n)]
    return [int(sample[f"x{i}"]) for i in range(n)]


def cqm_for(problem, inst):
    return assignment_cqm(inst) if problem == "assignment" else knapsack_cqm(*inst)


def main(argv=None):
    p = examplekit.base_parser(__doc__, time_limit=1.0)
    p.add_argument("--problems", nargs="+", default=["assignment", "knapsack"], choices=["assignment", "knapsack"])
    p.add_argument("--assignment-n", type=int, default=12)
    p.add_argument("--knapsack-n", type=int, default=40)
    p.add_argument("--instance-seed", type=int, default=7)
    a = p.parse_args(argv)

    from fourcolors import FourColorsSampler

    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()
    sampler = FourColorsSampler(client=client) if client is not None else FourColorsSampler()
    for problem in a.problems:
        n = a.assignment_n if problem == "assignment" else a.knapsack_n
        inst = cp.instance(problem, n, a.instance_seed)
        opt = cp.optimum(problem, inst)
        cqm = cqm_for(problem, inst)
        for seed in a.seeds:
            try:
                ss = sampler.sample(cqm, time_limit_s=a.time_limit, seed=seed)
            except RuntimeError as err:                    # no answer meeting every constraint in time
                out.add(kind="cqm", problem=problem, size=n, instance_seed=a.instance_seed, seed=seed,
                        time_limit_s=a.time_limit, optimum=opt, feasible=False, objective=None, gap=None,
                        error=str(err)[:200], n_constraints=len(cqm.constraints))
                print(f"{problem:10s} CQM seed={seed}: {err}")
                continue
            first = ss.first
            x = vector(problem, n, first.sample)
            verdict = cp.judge(problem, inst, x)
            service_ok = (ss.info.get("constraints") or {}).get("feasible")
            dimod_ok = bool(first.is_feasible)
            row = out.add(kind="cqm", problem=problem, size=n, instance_seed=a.instance_seed, seed=seed,
                          time_limit_s=a.time_limit, optimum=opt, n_constraints=len(cqm.constraints),
                          gap=cp.gap(problem, verdict["objective"], opt) if verdict["feasible"] else None,
                          service_feasible=service_ok, dimod_feasible=dimod_ok,
                          checks_agree=service_ok == dimod_ok == verdict["feasible"],
                          dimod_energy=float(first.energy), job_id=ss.info.get("job_id"),
                          status=ss.info.get("status"), engine=ss.info.get("engine"),
                          engine_version=(ss.info.get("engine") or {}).get("version"),
                          client_version=examplekit.client_version(), service_constraints=ss.info.get("constraints"),
                          **verdict)
            print(f"{problem:10s} CQM seed={seed} feasible: service={service_ok} dimod={dimod_ok} "
                  f"python={row['feasible']} objective={row['objective']} (optimum {opt})")


if __name__ == "__main__":
    main()
