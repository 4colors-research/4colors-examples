"""Trace the efficient frontier: one solve per risk aversion q, with "exactly k assets" as an exact row.

    python frontier.py --out runs/n30 --n 30 --k 6 --time-limit 5 --seeds 1 2 3
    python frontier.py --out runs/n200 --n 200 --k 20 --time-limit 5 --seeds 1 2 3
    python frontier.py --out runs/csv --prices prices.csv --k 10        # your own daily closing prices
    python frontier.py --out runs/n30cap --n 30 --k 6 --sectors 5 --sector-cap 2   # optional sector cap

For each q the answer is judged here: the picks, their count, the portfolio's return and risk, and the objective,
all recomputed from the 0/1 solution. The reference is the exact optimum by enumeration when C(n, k) is at most
--brute-force-max, else a swap local search (a heuristic).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors.qubo

import examplekit
import portfolio as pf

QS = [0, 0.5, 1, 2, 4, 8, 16, 32]


def main(argv=None):
    p = examplekit.base_parser(__doc__)
    p.add_argument("--n", type=int, default=30, help="assets (synthetic data)")
    p.add_argument("--k", type=int, default=6, help="assets to pick")
    p.add_argument("--data-seed", type=int, default=1)
    p.add_argument("--prices", help="CSV of daily closing prices instead of synthetic data")
    p.add_argument("--q", type=float, nargs="+", default=QS, help="risk-aversion values")
    p.add_argument("--sectors", type=int, default=5, help="seeded sector labels (used only with --sector-cap)")
    p.add_argument("--sector-cap", type=int, default=None,
                   help="optional: at most this many assets per sector, one exact row per sector (off by default)")
    p.add_argument("--brute-force-max", type=int, default=1_000_000, help="largest C(n, k) to enumerate")
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()

    if a.prices:
        mu, sigma, names = pf.from_prices_csv(a.prices)
        source = f"csv:{Path(a.prices).name}"
    else:
        mu, sigma, names = pf.synthetic(a.n, a.data_seed)
        source = f"synthetic(seed={a.data_seed})"
    n, k = len(mu), a.k
    cap = a.sector_cap
    sector_of = pf.sectors(n, a.sectors, a.data_seed) if cap is not None else None
    sec = {"sector_of": sector_of, "sector_cap": cap}
    exact = math.comb(n, k) <= a.brute_force_max
    refs = (pf.brute_force(mu, sigma, k, a.q, **sec) if exact
            else {q: pf.swap_search(mu, sigma, k, q, **sec) for q in a.q})
    method = "enumeration (exact)" if exact else "swap local search (heuristic)"
    common = {"n": n, "k": k, "data": source, "time_limit_s": a.time_limit, "sector_cap": cap,
              "n_sectors": a.sectors if cap is not None else None}
    for q in a.q:
        ref_obj, ref_picks = refs[q]
        out.add(kind="reference", q=q, objective=ref_obj, picks=ref_picks, method=method,
                ret=pf.portfolio_return(mu, [int(i in ref_picks) for i in range(n)], k),
                risk=math.sqrt(pf.portfolio_variance(sigma, [int(i in ref_picks) for i in range(n)], k)), **common)
        m = pf.model(mu, sigma, k, q, **sec)
        for seed in a.seeds:
            r = fourcolors.qubo.solve(m, time_limit_s=a.time_limit, seed=seed, client=client)
            x = r.solution
            row = dict(kind="solve", q=q, seed=seed, n_vars=m.n, n_quad=m.nnz, n_rows=m.m, **common)
            if x is not None:
                chosen = pf.picks(x)
                obj = pf.objective(mu, sigma, k, q, x)
                row.update(picks=chosen, names=[names[i] for i in chosen], n_picked=len(chosen),
                           feasible=len(chosen) == k, ret=pf.portfolio_return(mu, x, k),
                           risk=math.sqrt(pf.portfolio_variance(sigma, x, k)), objective=obj,
                           reference=ref_obj, reference_method=method, gap_abs=obj - ref_obj,
                           model_feasible=m.feasible(x), energy_python=m.energy(x))
                if cap is not None:
                    row.update(sector_counts={str(s): c for s, c in sorted(pf.sector_counts(x, sector_of).items())},
                               sector_ok=pf.sector_ok(x, sector_of, cap))
                    row["feasible"] = row["feasible"] and row["sector_ok"]
            else:
                row.update(feasible=False)
            out.add(**row, **examplekit.result_fields(r))
            print(f"q={q:<5g} seed={seed} picked={row.get('n_picked')} return={row.get('ret')} "
                  f"risk={row.get('risk')} objective={row.get('objective')} (reference {ref_obj}, {method})")


if __name__ == "__main__":
    main()
