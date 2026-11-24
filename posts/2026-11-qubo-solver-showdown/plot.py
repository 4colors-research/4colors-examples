"""Gap by budget and wall time against budget, per solver setting, from results.jsonl only.

    python plot.py results/try/results.jsonl --out figures

Aggregation (as in our recorded campaigns): for each instance, the median over seeds per setting and
budget; then the median of those over instances. The reference is the instance's own (a best known cut
given on the command line, or the spin-glass ground-state density); without one, the best answer found by
any row for that instance ("best found", never "optimum"). "Best other" is the lowest median gap among
the non-4colors settings at that budget. Rows with an error count as missing and are listed.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import examplekit as bk

ORDER = ["4colors", "mqlib-hh", "mqlib-burer", "neal", "neal-fill", "openjij", "openjij-fill"]


def best_found(rows):
    best = {}
    for r in rows:
        if r.get("objective") is None:
            continue
        pick = max if r["kind"] == "maxcut" else min
        best[r["instance"]] = pick(best.get(r["instance"], r["objective"]), r["objective"])
    return best


def gaps(rows):
    """{(instance, setting, budget): [gap per seed]} with the reference rule above."""
    best = best_found(rows)
    out = defaultdict(list)
    for r in rows:
        if r.get("objective") is None or r.get("budget_s") is None:
            continue
        ref = r["reference"] if r.get("reference") is not None else best[r["instance"]]
        g = (ref - r["objective"]) / abs(ref) if r["kind"] == "maxcut" else (r["objective"] - ref) / abs(ref)
        out[(r["instance"], r["setting"], r["budget_s"])].append(g)
    return out


def summary(rows):
    per_inst = {k: statistics.median(v) for k, v in gaps(rows).items()}
    by = defaultdict(list)
    for (inst, setting, T), g in per_inst.items():
        by[(setting, T)].append(g)
    agg = {k: statistics.median(v) for k, v in by.items()}
    budgets = sorted({T for _, T in agg})
    best_other = {T: min((g for (s, t), g in agg.items() if t == T and s != "4colors"), default=None) for T in budgets}
    wall = defaultdict(list)
    for r in rows:
        if r.get("wall_s") is not None and r.get("budget_s"):
            wall[(r["setting"], r["budget_s"])].append(r["wall_s"])
    errors = [r for r in rows if r.get("error")]
    return {"agg": agg, "best_other": best_other, "wall": wall, "errors": errors, "per_instance": per_inst}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--out", default="figures")
    a = p.parse_args(argv)
    rows = [r for path in a.jsonl for r in bk.Results(path).rows() if r.get("event") != "upload"]
    s = summary(rows)
    print("setting, budget_s, median gap over instances, median wall_s, max wall_s")
    for (setting, T) in sorted(s["agg"], key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1])):
        w = s["wall"].get((setting, T), [])
        print(f"{setting}, {T:g}, {100 * s['agg'][(setting, T)]:.3f}%, "
              f"{statistics.median(w):.2f}, {max(w):.2f}" if w else f"{setting}, {T:g}, -, -, -")
    for T, g in s["best_other"].items():
        print(f"best other at {T:g} s: {'-' if g is None else f'{100 * g:.3f}%'}")
    for r in s["errors"]:
        print(f"error: {r['setting']} {r['instance']} T={r.get('budget_s')} seed={r.get('seed')}: {r['error']}")
    if not s["agg"]:
        return s
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(a.out).mkdir(parents=True, exist_ok=True)
    settings = [x for x in ORDER if any(k[0] == x for k in s["agg"])]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for k, setting in enumerate(settings):
        pts = sorted((T, 100 * g) for (st, T), g in s["agg"].items() if st == setting)
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=bk.SERIES[ORDER.index(setting)],
                marker=bk.MARKERS[ORDER.index(setting)], linewidth=2, markersize=8, label=setting)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("time budget (s)"); ax.set_ylabel("median gap (%)")
    ax.set_title("Gap to the reference by budget", loc="left", fontsize=11)
    bk.style_axes(ax); ax.legend(frameon=False, fontsize=8)
    fig.savefig(Path(a.out) / "gap_by_budget.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for setting in settings:
        pts = sorted((T, max(w)) for (st, T), w in s["wall"].items() if st == setting)
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=bk.SERIES[ORDER.index(setting)],
                marker=bk.MARKERS[ORDER.index(setting)], linewidth=2, markersize=8, label=setting)
    lim = sorted({T for _, T in s["wall"]})
    ax.plot(lim, lim, color=bk.MUTED, linewidth=1, linestyle="--", label="wall = budget")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("time budget (s)"); ax.set_ylabel("longest wall time (s)")
    ax.set_title("Wall time against the budget asked for", loc="left", fontsize=11)
    bk.style_axes(ax); ax.legend(frameon=False, fontsize=8)
    fig.savefig(Path(a.out) / "wall_vs_budget.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return s


if __name__ == "__main__":
    main()
