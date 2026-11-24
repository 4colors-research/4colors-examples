"""Charts for the rebalancing post, from results.jsonl only.

    python plot.py runs/rebalance --out figures --tolerance-bp 0 1 5

1. rebalance_time_to_quality.png (hero): for each tolerance (in basis points of the objective, which is in annual
   return units), the median search time until a job is within it of the morning's reference; cold against warm.
   Labels count the jobs that got there.
2. rebalance_carry_over.png: per morning, how far yesterday's portfolio is from the morning's reference before any
   search (basis points). This is the drift the warm start has to absorb.
3. rebalance_turnover.png: names changed per morning in the held portfolio; with several inputs (for example a run
   with --cost 0 and one with a trading cost), one series per cost.

The morning's reference is the exact optimum (enumeration) where the run has one, else the best answer any job of
that morning found; the second is a reference, not a proven optimum.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
import examplekit

BP = 1e-4


def _key(r):
    return (r.get("n"), r.get("cost", 0.0), r["day"])


def morning_reference(rows):
    """{(n, cost, day): (objective, exact?)}: the enumeration optimum if recorded, else the best job objective."""
    exact, best = {}, defaultdict(lambda: None)
    for r in rows:
        key = _key(r) if "day" in r else None
        if r.get("kind") == "reference":
            exact[key] = r["objective_python"]
        elif r.get("kind") == "solve" and r.get("model_feasible") and r.get("objective_python") is not None:
            best[key] = r["objective_python"] if best[key] is None else min(best[key], r["objective_python"])
    out = {k: (v, False) for k, v in best.items()}
    out.update({k: (v, True) for k, v in exact.items()})
    return out


def times_to_quality(rows, tol):
    """{arm: [search time or None, ...]} for jobs on mornings >= 1, tolerance `tol` in objective units."""
    ref = morning_reference(rows)
    out = defaultdict(list)
    for r in rows:
        if r.get("kind") != "solve" or r.get("day", 0) == 0 or "trajectory" not in r:
            continue
        target = ref[_key(r)][0] + tol
        out[r["arm"]].append(examplekit.time_to_target(r["trajectory"], target + 1e-12))
    return dict(out)


def carry_over_gaps(rows):
    """[(day, gap in objective units)] of yesterday's portfolio on today's model."""
    ref = morning_reference(rows)
    return sorted((r["day"], r["objective_python"] - ref[_key(r)][0])
                  for r in rows if r.get("kind") == "carry_over")


def turnover(rows):
    """{(n, cost): [(day, names changed)]} of the held portfolio."""
    out = defaultdict(list)
    for r in rows:
        if r.get("kind") == "held" and r.get("changes") is not None:
            out[(r.get("n"), r.get("cost", 0.0))].append((r["day"], r["changes"]))
    return {c: sorted(v) for c, v in out.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--tolerance-bp", type=float, nargs="+", default=[0.0, 1.0, 5.0])
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    rows = examplekit.load_rows(a.inputs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    width = 0.38
    for j, arm in enumerate(["cold", "warm"]):
        meds, labels = [], []
        for bp in a.tolerance_bp:
            ts = times_to_quality(rows, bp * BP).get(arm, [])
            reached = [t for t in ts if t is not None]
            meds.append(examplekit.median(reached) or 0.0)
            labels.append(f"{len(reached)}/{len(ts)}")
        xs = [i + (j - 0.5) * width for i in range(len(a.tolerance_bp))]
        bars = ax.bar(xs, meds, width * 0.95, color=examplekit.SERIES[j], label=arm)
        for b, lab in zip(bars, labels):
            ax.annotate(lab, (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                        fontsize=8, color=examplekit.INK_2)
    ax.set_xticks(range(len(a.tolerance_bp)))
    ax.set_xticklabels(["exact" if bp == 0 else f"within {bp:g} bp" for bp in a.tolerance_bp])
    ax.set_ylabel("median search time (s)")
    ax.legend()
    ax.set_title("time to the morning's reference (labels: jobs that got there)", color=examplekit.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "rebalance_time_to_quality.png", dpi=150)
    plt.close(fig)

    gaps = carry_over_gaps(rows)
    fig, ax = plt.subplots(figsize=(6, 3.0))
    if gaps:
        ax.bar([d for d, _ in gaps], [g / BP for _, g in gaps], color=examplekit.SERIES[1])
    ax.set_xlabel("morning")
    ax.set_ylabel("yesterday's portfolio, gap (bp)")
    ax.set_title("how far the overnight re-estimate moves the optimum", color=examplekit.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "rebalance_carry_over.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.0))
    for j, ((n, cost), pts) in enumerate(sorted(turnover(rows).items())):
        ax.plot([d for d, _ in pts], [c for _, c in pts], marker="o", color=examplekit.SERIES[j],
                label=f"n = {n}, cost {cost:g} per name")
    ax.set_xlabel("morning")
    ax.set_ylabel("names changed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "rebalance_turnover.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
