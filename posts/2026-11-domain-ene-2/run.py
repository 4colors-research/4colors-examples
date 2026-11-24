"""Partition power grids for parallel simulation: balanced min-cut with exact rows, against labeled references.

    python run.py --out runs/grid --case data/pglib_opf_case118_ieee.m data/pglib_opf_case1354_pegase.m \
        --parts 2 4 --imbalance 0.03 --time-limit 1 5 --seeds 1 2 3 --milp-max-buses 400

Per case and number of parts: an `instance` row (buses, branches, file sha256), a `reference` row from the MILP
(scipy/HiGHS, labeled `optimal` only when proven) for grids up to --milp-max-buses, and `solve` rows (our answer per
seed and time limit, decoded, cut and balance recomputed in plain Python, both feasibility checks).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors
import fourcolors.qubo

import examplekit
import grid as gr


def main(argv=None):
    p = examplekit.base_parser(__doc__, time_limit=1.0)
    p.set_defaults(time_limit=None)
    p.add_argument("--case", nargs="+", required=True, help="MATPOWER .m files (PGLib-OPF)")
    p.add_argument("--parts", type=int, nargs="+", default=[2, 4])
    p.add_argument("--imbalance", type=float, default=0.03)
    p.add_argument("--budgets", type=float, nargs="+", default=[1.0, 5.0], help="seconds per job")
    p.add_argument("--weight", choices=["unit", "susceptance"], default="unit")
    p.add_argument("--work", choices=["unit", "degree"], default="unit")
    p.add_argument("--milp-max-buses", type=int, default=400)
    p.add_argument("--milp-time-limit", type=float, default=120.0)
    a = p.parse_args(argv)
    budgets = [a.time_limit] if a.time_limit else a.budgets
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()

    for path in a.case:
        text = Path(path).read_text()
        g = gr.to_graph(gr.parse_matpower(text), a.weight)
        c = gr.work(g, a.work)
        name = Path(path).stem
        out.add(kind="instance", case=name, buses=g["n"], branches=len(g["edges"]), total_weight=sum(g["w"]),
                sha256=hashlib.sha256(text.encode()).hexdigest(), weight=a.weight, work=a.work)
        for parts in a.parts:
            common = dict(case=name, buses=g["n"], parts=parts, imbalance=a.imbalance, weight=a.weight, work=a.work)
            if g["n"] <= a.milp_max_buses:
                ref = gr.milp_reference(g, c, parts, a.imbalance, a.milp_time_limit)
                out.add(kind="reference", method="milp-highs", **common, status=ref["status"], cut=ref["cut"],
                        bound=ref["bound"], milp_time_limit_s=a.milp_time_limit)
            m = gr.bisection_model(g, c, a.imbalance) if parts == 2 else gr.kway_model(g, c, parts, a.imbalance)
            for budget in budgets:
                for seed in a.seeds:
                    try:
                        r = fourcolors.qubo.solve(m, time_limit_s=budget, seed=seed, client=client)
                    except fourcolors.errors.FourColorsError as exc:
                        out.add(kind="solve", arm="4colors", **common, seed=seed, time_limit_s=budget,
                                n_vars=m.n, n_rows=m.m, error_code=getattr(exc, "code", None), error=str(exc))
                        continue
                    x = r.solution
                    part = None if x is None else gr.decode(x, g["n"], parts)
                    row = out.add(kind="solve", arm="4colors", **common, seed=seed, time_limit_s=budget,
                                  n_vars=m.n, n_rows=m.m,
                                  cut=None if part is None else gr.cut_weight(g, part),
                                  balanced=None if part is None else gr.balanced(part, c, parts, a.imbalance),
                                  part_work=None if part is None else gr.part_work(part, c, parts),
                                  part=part,
                                  energy_python=None if x is None else m.energy(x),
                                  model_feasible=None if x is None else bool(m.feasible(x)),
                                  service_feasible=(r.constraints or {}).get("feasible"),
                                  **examplekit.result_fields(r))
                    print(f"{name} parts={parts} {budget:g}s seed={seed} cut={row['cut']} "
                          f"balanced={row['balanced']}")


if __name__ == "__main__":
    main()
