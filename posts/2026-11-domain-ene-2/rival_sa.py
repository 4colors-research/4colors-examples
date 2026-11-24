"""Open-source baseline for the bisection cells: simulated annealing (dwave-samplers) on the penalty form.

An annealer takes no rows, so the balance row becomes a penalty: cut(x) + lam * (sum_i c_i x_i - total / 2)^2.
Reads run in batches until the wall budget is spent; only answers that meet the balance row count, and the best
feasible cut is kept. lam is chosen beforehand by --tune (one seed, every lam in the grid, the full budget each;
the tuning time is not charged to the baseline, which favors it).

The penalty makes the model dense: n(n-1)/2 extra couplings (about 4.1M at 2,869 buses, 93M at 13,659), which is
why --max-buses (default 3,000) refuses larger grids. The exact row in our model stays one row of n entries.

Timed runs only: the budget is local wall time, so this script must run on booked, pinned cores of the same CPU
class and core count as the comparison states [MASTER BOOKING]. Without a timed booking its rows are not citable.

    python rival_sa.py --out runs/grid-sa --case data/pglib_opf_case1354_pegase.m --budgets 1 5 --seeds 1 2 3 --lam 0.5
    python rival_sa.py --out runs/grid-sa-tune --case data/pglib_opf_case1354_pegase.m --budgets 5 --seeds 1 --tune 0.1 0.3 1 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import examplekit
import grid as gr


def penalty_bqm(g, c, lam):
    import dimod

    target = sum(c) / 2
    bqm = dimod.BinaryQuadraticModel("BINARY")
    for (i, j), w in zip(g["edges"], g["w"]):
        bqm.add_linear(i, w)
        bqm.add_linear(j, w)
        bqm.add_quadratic(i, j, -2 * w)
    # lam * (sum c_i x_i - T)^2 = lam * [sum c_i^2 x_i + 2 sum_{i<j} c_i c_j x_i x_j - 2T sum c_i x_i + T^2]
    for i in range(g["n"]):
        bqm.add_linear(i, lam * (c[i] ** 2 - 2 * target * c[i]))
        for j in range(i + 1, g["n"]):
            bqm.add_quadratic(i, j, 2 * lam * c[i] * c[j])
    bqm.offset += lam * target ** 2
    return bqm


def anneal(g, c, imbalance, lam, budget_s, seed, sweeps=1000, batch=8):
    from dwave.samplers import SimulatedAnnealingSampler

    bqm = penalty_bqm(g, c, lam)
    sampler = SimulatedAnnealingSampler()
    best, reads, t0, k = None, 0, time.perf_counter(), 0
    while time.perf_counter() - t0 < budget_s:
        ss = sampler.sample(bqm, num_reads=batch, num_sweeps=sweeps, seed=seed * 100003 + k)
        k += 1
        for s in ss.samples():
            reads += 1
            part = [int(s[i]) for i in range(g["n"])]
            if gr.balanced(part, c, 2, imbalance):
                cut = gr.cut_weight(g, part)
                if best is None or cut < best:
                    best = cut
    return {"cut": best, "reads": reads, "wall_s": time.perf_counter() - t0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--case", nargs="+", required=True)
    p.add_argument("--budgets", type=float, nargs="+", default=[1.0, 5.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--imbalance", type=float, default=0.03)
    p.add_argument("--weight", choices=["unit", "susceptance"], default="unit")
    p.add_argument("--work", choices=["unit", "degree"], default="unit")
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--tune", type=float, nargs="*", help="lam grid: run every value, report, pick by hand")
    p.add_argument("--sweeps", type=int, default=1000)
    p.add_argument("--max-buses", type=int, default=3000, help="refuse denser penalty models than this")
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    for path in a.case:
        g = gr.to_graph(gr.parse_matpower(Path(path).read_text()), a.weight)
        c = gr.work(g, a.work)
        if g["n"] > a.max_buses:
            out.add(kind="rival", arm="sa-penalty", case=Path(path).stem, buses=g["n"], parts=2,
                    skipped=f"penalty model has {g['n'] * (g['n'] - 1) // 2} couplings; over --max-buses")
            continue
        for lam in (a.tune or [a.lam]):
            for budget in a.budgets:
                for seed in a.seeds:
                    res = anneal(g, c, a.imbalance, lam, budget, seed, sweeps=a.sweeps)
                    out.add(kind="rival", arm="sa-penalty", tuning=bool(a.tune), case=Path(path).stem, buses=g["n"],
                            parts=2, imbalance=a.imbalance, weight=a.weight, work=a.work, lam=lam, sweeps=a.sweeps,
                            seed=seed, time_limit_s=budget, **res)
                    print(f"{Path(path).stem} lam={lam} {budget:g}s seed={seed} cut={res['cut']} reads={res['reads']}")


if __name__ == "__main__":
    main()
