"""Solve the ten graph problems through the API, decode each answer and check it in plain Python.

    python run.py --out runs/ten --time-limit 1 --seeds 1
    python run.py --out runs/ten --problems max_cut tsp --size small

Graphs come from networkx generators with a fixed --graph-seed, so every reader builds the same instances; the solver
seed is --seeds. Each row records the problem, its size, the decoded answer, `valid` and `score` from plain Python,
the client's exact row check (`model_feasible`) and the service's (`service_feasible`), and the service's fields.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors
import fourcolors.qubo

import examplekit
import graphs


def jsonable(answer):
    """Sets become sorted lists so the row stays readable JSON."""
    if isinstance(answer, (set, frozenset)):
        return sorted(answer, key=str)
    return answer


def main(argv=None):
    p = examplekit.base_parser(__doc__, time_limit=1.0, seeds=(1,))
    p.add_argument("--problems", nargs="+", default=list(graphs.BUILDERS), choices=list(graphs.BUILDERS))
    p.add_argument("--size", choices=["default", "small"], default="default")
    p.add_argument("--graph-seed", type=int, default=7, help="seed of the networkx generators (default 7)")
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()

    for name, G, kw in graphs.instances(a.size, a.graph_seed):
        if name not in a.problems:
            continue
        prob = graphs.build(name, G, **kw)
        m = prob.model
        common = dict(kind="solve", problem=name, goal=prob.goal, n_nodes=G.number_of_nodes(),
                      n_edges=G.number_of_edges(), n_vars=m.n, n_quad=m.nnz, n_rows=m.m, sense=m.sense,
                      graph_seed=a.graph_seed, size=a.size, time_limit_s=a.time_limit, **prob.info)
        for seed in a.seeds:
            try:
                r = fourcolors.qubo.solve(m, time_limit_s=a.time_limit, seed=seed, client=client)
            except fourcolors.errors.FourColorsError as exc:            # a refusal is recorded, not hidden
                out.add(**common, seed=seed, error_code=getattr(exc, "code", None), error=str(exc))
                print(f"{name:22s} seed={seed} refused: {exc}")
                continue
            x = r.solution
            answer = None if x is None else prob.decode(list(x))
            verdict = {"valid": False, "score": None} if answer is None else prob.check(answer)
            row = out.add(**common, seed=seed, answer=jsonable(answer), **verdict,
                          model_feasible=None if x is None else m.feasible(x),
                          service_feasible=(r.constraints or {}).get("feasible") if m.m else None,
                          energy_python=None if x is None else m.energy(x), **examplekit.result_fields(r))
            print(f"{name:22s} seed={seed} valid={row['valid']} score={row['score']} ({prob.goal}) "
                  f"reason={row['reason']}")
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
