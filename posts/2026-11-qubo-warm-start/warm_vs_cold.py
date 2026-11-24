"""Cold solves against warm starts from yesterday's answer, day after day.

Day 0 is solved once (cold). On each later day, "yesterday" is the best cut found on the day before (any arm), and
every seed runs two jobs with the same short time limit: one cold, one with warm_start=yesterday. The trajectory of
each job (every improvement with its search time) is stored, so time-to-quality can be read off afterwards.

    python warm_vs_cold.py --out runs/warm --n 10000 --days 6 --time-limit 3 --seeds 1 2 3
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors.qubo

import examplekit
import daily_maxcut as dm


def run_one(out, client, m, edges, w, *, day, arm, seed, time_limit, warm, common):
    r = fourcolors.qubo.solve(m, time_limit_s=time_limit, seed=seed, client=client,
                              warm_start=warm if arm == "warm" else None)
    x = r.solution
    value = None if x is None else dm.cut(edges, w, x)            # recomputed here, not trusted
    out.add(kind="solve", day=day, arm=arm, seed=seed, time_limit_s=time_limit, cut=value,
            energy_python=None if x is None else m.energy(x), **common, **examplekit.result_fields(r))
    return value, x


def main(argv=None):
    p = examplekit.base_parser(__doc__, time_limit=3.0)
    p.add_argument("--n", type=int, default=10000, help="nodes (default %(default)s)")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--days", type=int, default=6, help="days after day 0 (default %(default)s)")
    p.add_argument("--share", type=float, default=0.05, help="share of edges re-weighted per day")
    p.add_argument("--instance-seed", type=int, default=11)
    p.add_argument("--day0-time-limit", type=float, default=10.0, help="seconds for the day-0 cold solve")
    a = p.parse_args(argv)

    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()
    common = {"n": a.n, "degree": a.degree, "share": a.share, "instance_seed": a.instance_seed}
    yesterday = None
    for day, edges, w in dm.days(a.n, a.degree, a.days, a.share, a.instance_seed):
        m = dm.model(a.n, edges, w)
        common_day = {**common, "n_vars": m.n, "n_quad": m.nnz}
        found = []
        if day == 0:
            found.append(run_one(out, client, m, edges, w, day=0, arm="cold", seed=a.seeds[0],
                                 time_limit=a.day0_time_limit, warm=None, common=common_day))
        else:
            # how good is yesterday's answer on today's weights, before any search?
            out.add(kind="carry_over", day=day, cut=dm.cut(edges, w, yesterday),
                    energy_python=m.energy(yesterday), **common_day)
            for seed in a.seeds:
                for arm in ("cold", "warm"):
                    found.append(run_one(out, client, m, edges, w, day=day, arm=arm, seed=seed,
                                         time_limit=a.time_limit, warm=yesterday, common=common_day))
        best = max((f for f in found if f[0] is not None), key=lambda f: f[0])
        yesterday = best[1]
        print(f"day {day}: best cut {best[0]} over {len(found)} job(s)")


if __name__ == "__main__":
    main()
