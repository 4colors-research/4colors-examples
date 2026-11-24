"""Charts for the warm-start post, from results.jsonl only.

    python plot.py runs/warm --out figures --quality 0 0.001 0.005

1. warm_time_to_quality.png: for each quality level q, the median search time (over seeds and days) until a job's
   cut is within q of that day's best cut found by any job. The label gives the share of jobs that got there.
2. warm_progress.png: the gap to the day's best against search time, median over seeds and days, cold and warm.
   --tmax limits it to the first tmax seconds of search (default: the whole time limit).

The day's best is the best cut any recorded job found that day; it is a reference, not a proven optimum.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
import examplekit


def day_best(rows):
    best = defaultdict(lambda: None)
    for r in rows:
        if r.get("kind") == "solve" and r.get("cut") is not None and r["day"] > 0:
            best[r["day"]] = r["cut"] if best[r["day"]] is None else max(best[r["day"]], r["cut"])
    return dict(best)


def times_to_quality(rows, q):
    """{arm: [time or None, ...]} for quality q (relative gap to the day's best)."""
    best = day_best(rows)
    out = defaultdict(list)
    for r in rows:
        if r.get("kind") != "solve" or r["day"] == 0:
            continue
        target = -best[r["day"]] * (1 - q)          # trajectories are in energy terms: E = -cut
        out[r["arm"]].append(examplekit.time_to_target(r["trajectory"], target))
    return dict(out)


def gap_at(traj, t, best):
    """Gap to `best` of the best cut found by search time t; None before the first point."""
    vals = [-obj for ts, obj in traj if ts is not None and ts <= t]
    return None if not vals else (best - max(vals)) / best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--quality", type=float, nargs="+", default=[0.0, 0.001, 0.005])
    p.add_argument("--tmax", type=float, default=None,
                   help="progress chart: show only the first TMAX seconds of search (default: the time limit)")
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    rows = examplekit.load_rows(a.inputs)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    arms = ["cold", "warm"]

    fig, ax = plt.subplots(figsize=(6, 3.4))
    width = 0.38
    for k, arm in enumerate(arms):
        meds, labels = [], []
        for q in a.quality:
            ts = times_to_quality(rows, q).get(arm, [])
            reached = [t for t in ts if t is not None]
            meds.append(examplekit.median(reached) or 0.0)
            labels.append(f"{100 * len(reached) / len(ts):.0f}%" if ts else "")
        xs = [i + (k - 0.5) * width for i in range(len(a.quality))]
        bars = ax.bar(xs, meds, width * 0.95, color=examplekit.SERIES[k], label=arm)
        for b, lab in zip(bars, labels):
            ax.annotate(lab, (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                        fontsize=8, color=examplekit.INK_2)
    ax.set_xticks(range(len(a.quality)))
    ax.set_xticklabels([f"within {100 * q:g}%" for q in a.quality])
    ax.set_ylabel("median search time (s)")
    ax.legend()
    ax.set_title("time to reach the day's best cut (labels: share of jobs that got there)", color=examplekit.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(a.out) / "warm_time_to_quality.png", dpi=150)
    plt.close(fig)

    best = day_best(rows)
    solves = [r for r in rows if r.get("kind") == "solve" and r["day"] > 0]
    tmax = a.tmax or max((r["time_limit_s"] for r in solves), default=1.0)
    grid = [tmax * k / 100 for k in range(1, 101)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, arm in enumerate(arms):
        curve = []
        for t in grid:
            g = [gap_at(r["trajectory"], t, best[r["day"]]) for r in solves if r["arm"] == arm]
            curve.append(examplekit.median([100 * x for x in g if x is not None]))
        pts = [(t, c) for t, c in zip(grid, curve) if c is not None]
        if pts:
            ax.plot(*zip(*pts), color=examplekit.SERIES[k], label=arm)
    ax.set_xlim(0, tmax)
    ax.set_xlabel("search time (s)")
    ax.set_ylabel("median gap to the day's best (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(a.out) / "warm_progress.png", dpi=150)
    plt.close(fig)
    print(f"wrote {a.out}/warm_time_to_quality.png and {a.out}/warm_progress.png")


if __name__ == "__main__":
    main()
