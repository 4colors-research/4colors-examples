"""Rebalance every morning: cold solves against warm starts from yesterday's portfolio.

Each morning mu and Sigma are re-estimated on a rolling window (market.mornings). Morning 0 is solved once, cold.
On each later morning, "yesterday" is the portfolio held after the morning before (the best answer any job found,
scored in plain Python), and every seed runs two jobs with the same time limit: `cold`, and `warm` with
warm_start=yesterday. A `carry_over` row scores yesterday's portfolio on today's model before any search; with
n <= --exact-max a `reference` row holds the exact optimum of the morning by enumeration.

    python run.py --out runs/rebalance --n 200 --k 20 --days 10 --time-limit 1 --seeds 1 2 3
    python run.py --out runs/rebalance-n30 --n 30 --k 6 --days 10 --time-limit 1 --seeds 1 2 3
    python run.py --out runs/mine --prices my_closes.csv --k 20       # your own closing prices [CHECK terms]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors
import fourcolors.qubo

import examplekit
import market
import rebalance as rb


def solve(out, client, m, args, *, day, arm, seed, time_limit, warm, common):
    try:
        r = fourcolors.qubo.solve(m, time_limit_s=time_limit, seed=seed, client=client,
                                  warm_start=warm if arm == "warm" else None)
    except fourcolors.errors.FourColorsError as exc:
        out.add(kind="solve", day=day, arm=arm, seed=seed, time_limit_s=time_limit, **common,
                error_code=getattr(exc, "code", None), error=str(exc))
        return None
    x = r.solution
    out.add(kind="solve", day=day, arm=arm, seed=seed, time_limit_s=time_limit, **common,
            objective_python=None if x is None else m.energy(x),
            model_feasible=None if x is None else bool(m.feasible(x)),
            service_feasible=(r.constraints or {}).get("feasible"),
            picks=None if x is None else rb.picks(x), **examplekit.result_fields(r))
    return None if x is None or not m.feasible(x) else (m.energy(x), list(x))


def main(argv=None):
    p = examplekit.base_parser(__doc__, time_limit=1.0)
    p.add_argument("--n", type=int, default=200, help="assets (synthetic series)")
    p.add_argument("--k", type=int, default=20, help="names held")
    p.add_argument("--q", type=float, default=2.0, help="risk aversion")
    p.add_argument("--days", type=int, default=10, help="mornings after morning 0")
    p.add_argument("--window", type=int, default=252, help="rolling estimation window, trading days")
    p.add_argument("--history", type=int, default=600, help="days of synthetic series")
    p.add_argument("--series-seed", type=int, default=7)
    p.add_argument("--prices", help="CSV of closing prices instead of the synthetic series")
    p.add_argument("--cost", type=float, default=0.0, help="trading cost per name changed (annual-return units)")
    p.add_argument("--day0-time-limit", type=float, default=5.0)
    p.add_argument("--exact-max", type=int, default=30, help="enumerate the exact optimum when n <= this")
    a = p.parse_args(argv)

    if a.prices:
        returns, names = market.returns_from_prices_csv(a.prices)
        source = "prices-csv"
    else:
        returns, names = market.simulate(a.n, a.history, a.series_seed)
        source = "synthetic"
    n = returns.shape[1]
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()
    common = {"n": n, "k": a.k, "q": a.q, "window": a.window, "cost": a.cost, "source": source,
              "series_seed": None if a.prices else a.series_seed}
    held = None
    for day, mu, sigma in market.mornings(returns, a.window, a.days):
        prev = held if a.cost else None
        m = rb.model(mu, sigma, a.k, a.q, prev=prev, cost=a.cost)
        common_day = {**common, "n_vars": m.n, "n_rows": m.m}
        if n <= a.exact_max:
            best, x = rb.exact_best(mu, sigma, a.k, a.q, prev=prev, cost=a.cost)
            out.add(kind="reference", day=day, method="enumeration", objective_python=best, picks=rb.picks(x),
                    **common_day)
        found = []
        if day == 0:
            found.append(solve(out, client, m, a, day=0, arm="cold", seed=a.seeds[0],
                               time_limit=a.day0_time_limit, warm=None, common=common_day))
        else:
            out.add(kind="carry_over", day=day, objective_python=m.energy(held), **common_day)
            for seed in a.seeds:
                for arm in ("cold", "warm"):
                    found.append(solve(out, client, m, a, day=day, arm=arm, seed=seed, time_limit=a.time_limit,
                                       warm=held, common=common_day))
        found = [f for f in found if f is not None]
        if not found:
            if held is None:
                raise SystemExit(f"morning {day}: no feasible answer and nothing held; stop")
            print(f"morning {day}: no feasible answer; keep yesterday's portfolio")
            continue
        new = min(found, key=lambda f: f[0])[1]
        turnover = None if held is None else rb.changes(new, held)
        out.add(kind="held", day=day, objective_python=m.energy(new), picks=rb.picks(new), changes=turnover,
                **common_day)
        held = new
        print(f"morning {day}: best objective {min(f[0] for f in found):.6f} over {len(found)} job(s), "
              f"names changed {turnover}")


if __name__ == "__main__":
    main()
