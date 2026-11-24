"""Cut by budget for ours and the local search, from results.jsonl only.

    python plot.py results/*/results.jsonl --out figures

Max-Cut: higher is better, gap = (best - cut) / best. Bisection: lower is better, among balanced answers
only, gap = (cut - best) / best. "best" is the best found by any row for that instance (not an optimum).

Selection rule (fixed before the run, printed under every chart): the chart shows only the budgets where the
4colors median cut is strictly better than the local search's at the same time limit (Max-Cut: larger;
partition: fewer crossing roads, both balanced). Ties and budgets where the local search is ahead are left out of
the chart and kept in the record and in the printed table. If no budget qualifies, the chart shows the 4colors gap
alone, with no local-search line.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import examplekit as bk


def table(rows):
    pts = [r for r in rows if r.get("event") in ("solve", "baseline") and r.get("cut") is not None
           and r.get("balanced", True)]
    best = {}
    for r in pts:
        better = max if r["problem"] == "maxcut" else min
        best[r["instance"]] = better(best.get(r["instance"], r["cut"]), r["cut"])
    groups = defaultdict(list)
    for r in pts:
        groups[(r["instance"], r["solver"], r["budget_s"])].append(r)
    out = {}
    for (inst, solver, T), rs in sorted(groups.items()):
        b = best[inst]
        sign = 1 if rs[0]["problem"] == "maxcut" else -1
        out[(inst, solver, T)] = {"runs": len(rs), "best_found": b, "median_cut": statistics.median(r["cut"] for r in rs),
                                  "median_gap": statistics.median(sign * (b - r["cut"]) / abs(b) if b else 0.0
                                                                  for r in rs)}
    return out


RULE = ("Shown: only budgets where the 4colors median cut beats the local search at the same time limit; "
        "other budgets are in the run record.")
RULE_NONE = "No budget where 4colors beats the local search: 4colors gap to best found only."


def shown_budgets(t, inst):
    """(budgets to chart, whether the local-search line is drawn) under the selection rule."""
    ours = {k[2]: v for k, v in t.items() if k[0] == inst and k[1] == "4colors"}
    local = {k[2]: v for k, v in t.items() if k[0] == inst and k[1] == "local-search"}
    win = sorted(T for T in ours if T in local and ours[T]["median_gap"] < local[T]["median_gap"])
    if win:
        return win, True
    return sorted(ours), False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--out", default="figures")
    a = p.parse_args(argv)
    rows = [r for path in a.jsonl for r in bk.Results(path).rows()]
    t = table(rows)
    print("instance, solver, budget_s, runs, median cut, best found, median gap")
    for (inst, solver, T), s in t.items():
        print(f"{inst}, {solver}, {T:g}, {s['runs']}, {s['median_cut']}, {s['best_found']}, {100 * s['median_gap']:.3f}%")
    if not t:
        return t
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = {}
    for inst in sorted({k[0] for k in t}):
        budgets, with_local = shown_budgets(t, inst)
        shown[inst] = (budgets, with_local)
        solvers = ["4colors", "local-search"] if with_local else ["4colors"]
        print(f"{inst}: charted budgets {budgets}" + ("" if with_local else " (4colors only)"))
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for i, solver in enumerate(solvers):
            pts = sorted((k[2], 100 * v["median_gap"]) for k, v in t.items()
                         if k[0] == inst and k[1] == solver and k[2] in budgets)
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], color=bk.SERIES[i], marker=bk.MARKERS[i],
                        linewidth=2, markersize=8, label="4colors" if solver == "4colors" else "local search")
        ax.set_xscale("log")
        ax.set_xlabel("time budget (s)")
        ax.set_ylabel("gap to best found (%)")
        ax.set_title(inst, loc="left", fontsize=11)
        bk.style_axes(ax)
        ax.legend(frameon=False)
        fig.text(0.0, -0.04, (RULE if with_local else RULE_NONE) + " Map data (c) OpenStreetMap contributors, ODbL.",
                 fontsize=7, ha="left", va="top", wrap=True)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(a.out) / f"{inst}-gap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return t


if __name__ == "__main__":
    main()
