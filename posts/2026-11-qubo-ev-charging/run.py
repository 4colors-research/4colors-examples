"""EV smart charging as MWIS: solve with exact rows through the API, check every schedule, compare with references.

    python run.py --out results/ev                                   # 100 to 100,000 tasks, 1 and 10 s, seeds 1-3
    python run.py --out results/ev --sizes 100 --time-limits 1 --seeds 1
    python run.py --out results/ev --rival neal --skip-4colors       # the local open-source rival only (timed run)
    python run.py --out results/ev --site-kw 60 --sizes 100,1000     # with a site power cap row per slot

Rows:
* "instance": tasks, vehicles, chargers, clique rows, conflict edges (the MWIS graph's size), seed;
* "reference": HiGHS (scipy.optimize.milp) on the same rows, with a time limit: optimum or best and upper bound;
* "solve": the 4colors Optimizer for QUBO (`fourcolors.qubo.solve`), the schedule decoded and checked in plain
  Python (`charging.check`), the client's row check and the service's fields;
* "rival" (with --rival neal): dwave-samplers' simulated annealing on the penalty form, as given and budget-filling
  (reads until the same seconds are spent, best kept), decoded and checked the same way.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors
import fourcolors.qubo

import examplekit
import charging


def neal_runs(inst, time_limit_s: float, seed: int):
    """Two rows: neal with its defaults (one read), and reads repeated until `time_limit_s` (best kept)."""
    from dwave.samplers import SimulatedAnnealingSampler

    lin, quad, P = charging.penalty_bqm(inst)
    Q = {**{(i, i): v for i, v in lin.items()}, **quad}
    sampler = SimulatedAnnealingSampler()
    out = []
    for mode in ("as_given", "budget_filling"):
        t0, best, reads, k = time.perf_counter(), None, 0, 0
        while True:
            res = sampler.sample_qubo(Q, num_reads=1, seed=1000 * seed + k)
            rec = res.first
            reads += 1; k += 1
            if best is None or rec.energy < best.energy:
                best = rec
            if mode == "as_given" or time.perf_counter() - t0 >= time_limit_s:
                break
        x = [int(best.sample[i]) for i in range(len(inst.tasks))]
        out.append({"mode": mode, "x": x, "reads": reads, "wall_s": time.perf_counter() - t0, "penalty": P,
                    "energy": float(best.energy)})
    return out


def main(argv=None):
    p = examplekit.base_parser(__doc__, time_limit=1.0, seeds=(1, 2, 3))
    p.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[100, 1000, 10000, 100000],
                   help="tasks per instance (MWIS nodes)")
    p.add_argument("--instances", type=int, nargs="+", default=[1, 2], help="instance seeds")
    p.add_argument("--time-limits", type=lambda s: [float(x) for x in s.split(",")], default=[1.0, 10.0])
    p.add_argument("--site-kw", type=int, default=None, help="site power cap per slot (kW); default none")
    p.add_argument("--reference-time", type=float, default=60.0, help="HiGHS time limit per instance")
    p.add_argument("--skip-reference", action="store_true")
    p.add_argument("--rival", choices=["none", "neal"], default="none")
    p.add_argument("--skip-4colors", action="store_true")
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    out.add(kind="meta", argv=sys.argv[1:], args=vars(a), fourcolors=examplekit.client_version())
    client = None if a.skip_4colors else examplekit.client()

    for size in a.sizes:
        for iseed in a.instances:
            inst = charging.for_tasks(size, seed=iseed, site_kw=a.site_kw)
            m = charging.model(inst)
            common = dict(n_tasks=len(inst.tasks), instance_seed=iseed, size=size, site_kw=a.site_kw)
            out.add(kind="instance", **common, vehicles=len(inst.vehicles), chargers=len(inst.chargers_kw),
                    depot=({"chargers_kw": inst.chargers_kw, "vehicles": inst.vehicles} if size <= 200 else None),
                    n_rows=m.m, conflict_edges=len(charging.conflict_edges(inst)))
            if not a.skip_reference:
                t0 = time.perf_counter()
                ref = charging.milp_reference(inst, a.reference_time)
                ref_x = ref.pop("reference_x")
                chk = charging.check(inst, charging.decode(inst, ref_x)) if ref_x else {"valid": None}
                out.add(kind="reference", **common, method="HiGHS via scipy.optimize.milp", **ref,
                        reference_valid=chk["valid"], wall_s=time.perf_counter() - t0, untimed=True)
            for tl in a.time_limits:
                for seed in a.seeds:
                    if not a.skip_4colors:
                        try:
                            r = fourcolors.qubo.solve(m, time_limit_s=tl, seed=seed, client=client)
                        except fourcolors.errors.FourColorsError as exc:
                            out.add(kind="solve", **common, time_limit_s=tl, seed=seed,
                                    error_code=getattr(exc, "code", None), error=str(exc)[:300])
                            continue
                        x = r.solution
                        chk = charging.check(inst, charging.decode(inst, x)) if x is not None else {"valid": False}
                        row = out.add(kind="solve", **common, arm="4colors", time_limit_s=tl, seed=seed,
                                      **{k: chk.get(k) for k in ("valid", "value", "served", "peak_kw", "problems")},
                                      schedule=charging.decode(inst, x) if x is not None and size <= 200 else None,
                                      model_feasible=None if x is None else m.feasible(x),
                                      service_feasible=(r.constraints or {}).get("feasible"),
                                      **examplekit.result_fields(r))
                        print(f"ev {size} inst {iseed} {tl:g}s seed {seed}: value {row['value']} valid {row['valid']}",
                              flush=True)
                    if a.rival == "neal" and a.site_kw is None:
                        for nr in neal_runs(inst, tl, seed):
                            chk = charging.check(inst, charging.decode(inst, nr.pop("x")))
                            out.add(kind="rival", **common, arm=f"neal-{nr['mode']}", time_limit_s=tl, seed=seed,
                                    **{k: chk.get(k) for k in ("valid", "value", "served", "problems")}, **nr)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
