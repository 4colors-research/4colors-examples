"""TSP as a QUBO, the process: the n^2 one-hot encoding with exact rows and with penalties, decoded and checked.

    python solve_tsp.py --out runs/tsp --sizes 5 6 8 10 12 15 20 30 --time-limit 5 --seeds 1 2 3

For each size: one random Euclidean instance, the exact optimum by Held-Karp for n <= --exact-max (a labeled
reference; no reference above it), then one job per (arm, seed). Arms: `exact` (2n exact rows) and `penalty` at each
--lam factor (times the longest distance). Every answer is decoded and its tour length recomputed here; an answer
that is not a permutation counts as "not a tour". `size` rows give the model size by n, to --size-table-max, beside
what a routing solver reads for the same problem (one distance per city pair), with no API call.

A job the service refuses or fails (JobFailed, InvalidRequest: for example a model over the plan's limits) is
recorded as a `solve` row with status "refused" and its code, message and limit (circuitkit.refusal_fields, the
pattern circuitkit.compile_4colors uses), never dropped, and the run goes on. Errors that are not about this model
(no key, no credit, rate limits, network) are raised, not recorded.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors.qubo
from fourcolors.errors import InvalidRequest, JobFailed

import examplekit
import circuitkit
import tsp


def solve_or_refusal(m, *, time_limit_s, seed, client):
    """(result, None), or (None, refusal row fields) when the service refused or failed this job."""
    try:
        return fourcolors.qubo.solve(m, time_limit_s=time_limit_s, seed=seed, client=client), None
    except (JobFailed, InvalidRequest) as err:
        return None, circuitkit.refusal_fields(err)


def main(argv=None):
    p = examplekit.base_parser(__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[5, 6, 8, 10, 12, 15, 20, 30])
    p.add_argument("--lam", type=float, nargs="+", default=[0.5, 1, 2, 5], help="lam factors x longest distance")
    p.add_argument("--exact-max", type=int, default=12, help="largest n solved exactly by Held-Karp")
    p.add_argument("--instance-seed", type=int, default=2026)
    p.add_argument("--size-table-max", type=int, default=200)
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()

    for n in [k for k in (5, 10, 20, 50, 100, 200, 500, 1000) if k <= a.size_table_max]:
        for penalty in (False, True):
            out.add(kind="size", n=n, model="penalty" if penalty else "exact", routing_input=tsp.routing_input_size(n),
                    **tsp.model_size(n, penalty))

    for n in a.sizes:
        pts, d = tsp.random_instance(n, a.instance_seed + n)
        ref, method = None, None
        if n <= a.exact_max:
            ref, ref_tour = tsp.held_karp(d)
            method = "held-karp (exact optimum)"
            out.add(kind="reference", n=n, instance_seed=a.instance_seed + n, points=pts, length=ref, tour=ref_tour,
                    method=method)
        else:
            out.add(kind="instance", n=n, instance_seed=a.instance_seed + n, points=pts)
        dmax = max(max(row) for row in d)
        arms = [("exact", None)] + [("penalty", f) for f in a.lam]
        for arm, factor in arms:
            m = tsp.exact_model(d) if arm == "exact" else tsp.penalty_model(d, factor * dmax)
            for seed in a.seeds:
                common = dict(kind="solve", n=n, arm=arm, lam_factor=factor,
                              lam=None if factor is None else factor * dmax, seed=seed, time_limit_s=a.time_limit,
                              n_vars=m.n, n_quad=m.nnz, n_rows=m.m, reference=ref, reference_method=method)
                r, refused = solve_or_refusal(m, time_limit_s=a.time_limit, seed=seed, client=client)
                if refused is not None:                     # a refusal is a row, never dropped
                    out.add(**common, feasible=None, tour=None, length=None, gap=None, **refused,
                            client_version=examplekit.client_version())
                    print(f"n={n:<3d} {arm:7s} lam={factor} seed={seed} refused: {refused['refusal_code']} "
                          f"{refused['refusal_message']}")
                    continue
                tour = None if r.solution is None else tsp.decode(n, r.solution)
                length = None if tour is None else tsp.tour_length(d, tour)
                row = out.add(**common, feasible=tour is not None, tour=tour, length=length,
                              gap=None if length is None or ref is None else (length - ref) / ref,
                              model_feasible=None if r.solution is None else m.feasible(r.solution),
                              energy_python=None if r.solution is None else m.energy(r.solution),
                              **examplekit.result_fields(r))
                print(f"n={n:<3d} {arm:7s} lam={factor} seed={seed} tour={row['feasible']} "
                      f"length={length} (optimum {ref if ref is not None else 'not computed'})")


if __name__ == "__main__":
    main()
