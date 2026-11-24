"""Charts and a table from results.jsonl only (no API, no instance files).

    python plot.py results/L100/results.jsonl results/L215/results.jsonl --out figures

Writes figures/gap_by_budget.png (gap to the published ground-state density, one line per size) and
prints one table row per (size, budget): median gap over solver seeds, wall time, billed seconds.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import examplekit as bk


def table(rows):
    """{(n, budget): summary} from the solve rows that carry a re-evaluated answer."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("event") == "solve" and r.get("gap_pal1996") is not None:
            groups[(r["n"], r["budget_s"])].append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        out[key] = {"runs": len(rs), "all_agree": all(r.get("agrees") for r in rs),
                    "median_gap": statistics.median(r["gap_pal1996"] for r in rs),
                    "median_gap_later": statistics.median(r["gap_later"] for r in rs),
                    "median_wall_s": statistics.median(r["wall_s"] for r in rs),
                    "median_billed_s": statistics.median(r["job_billed_s"] for r in rs
                                                         if r.get("job_billed_s") is not None)
                    if any(r.get("job_billed_s") is not None for r in rs) else None}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--out", default="figures")
    a = p.parse_args(argv)
    rows = [r for path in a.jsonl for r in bk.Results(path).rows()]
    t = table(rows)
    print("n, budget_s, runs, median gap (Pal 1996), median gap (later estimate), wall_s, billed_s, all re-checks agree")
    for (n, T), s in t.items():
        print(f"{n:,}, {T:g}, {s['runs']}, {100 * s['median_gap']:.3f}%, {100 * s['median_gap_later']:.3f}%, "
              f"{s['median_wall_s']:.1f}, {s['median_billed_s']}, {s['all_agree']}")
    if not t:
        return t
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sizes = sorted({n for n, _ in t})
    for k, n in enumerate(sizes):
        pts = sorted((T, 100 * s["median_gap"]) for (m, T), s in t.items() if m == n)
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=bk.SERIES[k % 8], marker=bk.MARKERS[k % 8],
                linewidth=2, markersize=8, label=f"{n / 1e6:.0f}M spins" if n >= 1e6 else f"{n:,} spins")
    ax.set_xscale("log")
    ax.set_xlabel("time budget (s)")
    ax.set_ylabel("gap to ground-state density (%)")
    ax.set_title("3D ±J spin glass: gap to -1.7863 per spin", loc="left", fontsize=11)
    bk.style_axes(ax)
    ax.legend(frameon=False)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(a.out) / "gap_by_budget.png", dpi=150, bbox_inches="tight")
    return t


if __name__ == "__main__":
    main()
