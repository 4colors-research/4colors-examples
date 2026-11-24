"""Figures for the traffic-flow post, from the record only.

    python plot.py runs/traffic --out figures

- map.png: the street map twice, segment width and color by the number of cars on it: everyone on the shortest path
  (left) and the 4colors assignment (right); smallest car count, seed 1, the largest budget.
- congestion.png: congestion relative to "everyone takes the shortest path" (percent lower is better), per car
  count and budget, for 4colors and every other method recorded. Selection rule (printed under the chart): a car
  count is drawn only where every 4colors answer is valid and its median congestion is strictly below the median of
  every other method at the same budget.
- trajectory.png: 4colors congestion against service search seconds (the streamed improvements), largest drawn car
  count, seed 1. Ours only.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit

OURS = "4colors"
RULE = ("Shown: car counts where every 4colors answer is valid and its median congestion is below every other "
        "method's at the same budget.")


def method_medians(rows):
    """{(budget, cars): {method: (median congestion, all valid)}}; baselines count at every budget."""
    by = defaultdict(lambda: defaultdict(list))
    budgets = sorted({r["budget_s"] for r in rows if r.get("kind") == "solve"})
    for r in rows:
        if r.get("kind") == "solve" and "error_code" not in r:
            by[(r["budget_s"], r["cars"])][r["arm"]].append(r)
    for r in rows:
        if r.get("kind") == "baseline":
            for T in budgets:
                by[(T, r["cars"])][r["method"]].append(dict(r, valid=True))
    return {k: {m: (examplekit.median([x.get("congestion") for x in v]), all(x.get("valid") for x in v))
                for m, v in d.items()} for k, d in by.items()}


def winning_cells(rows):
    """[(budget, cars)] passing the selection rule."""
    keep = []
    for k, meds in sorted(method_medians(rows).items()):
        ours = meds.get(OURS)
        if ours is None or not ours[1] or ours[0] is None:
            continue
        others = [v[0] for m, v in meds.items() if m != OURS and v[0] is not None]
        if all(ours[0] < o for o in others):
            keep.append(k)
    return keep


def draw_loads(ax, graph, loads, title, vmax):
    xy = graph["xy"]
    for u, v in graph["segment_list"]:
        a, b = xy[str(u)], xy[str(v)]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=examplekit.GRID, linewidth=0.4, zorder=1)
    import matplotlib

    cmap = matplotlib.colormaps["Blues"]
    for u, v, L in sorted(loads, key=lambda t: t[2]):
        a, b = xy[str(u)], xy[str(v)]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=cmap(0.3 + 0.7 * L / vmax), linewidth=0.5 + 3.5 * L / vmax,
                zorder=2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = examplekit.load_rows(a.runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt = examplekit.plot_style()

    graph = next((r for r in rows if r.get("kind") == "graph" and r.get("xy")), None)
    solves = [r for r in rows if r.get("kind") == "solve" and r.get("arm") == OURS and r.get("segment_loads")]
    if graph and solves:
        cars = min(r["cars"] for r in solves)
        ours = max((r for r in solves if r["cars"] == cars), key=lambda r: (r["budget_s"], -r["seed"]))
        short = next((r for r in rows if r.get("kind") == "baseline" and r.get("method") == "shortest"
                      and r["cars"] == cars and r.get("segment_loads")), None)
        if short:
            vmax = max(L for *_, L in short["segment_loads"] + ours["segment_loads"])
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
            draw_loads(axes[0], graph, short["segment_loads"],
                       f"everyone on the shortest path: congestion {short['congestion']}", vmax)
            draw_loads(axes[1], graph, ours["segment_loads"], f"4colors: congestion {ours['congestion']}", vmax)
            fig.text(0.01, 0.01, "Map data (c) OpenStreetMap contributors, ODbL." if "OpenStreetMap" in
                     str(graph.get("source")) else str(graph.get("source")), fontsize=8, color=examplekit.MUTED)
            fig.savefig(out / "map.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    keep = winning_cells(rows)
    meds = method_medians(rows)
    if keep:
        methods = [OURS] + sorted({m for k in keep for m in meds[k] if m not in (OURS, "shortest")})
        fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(keep) * len(methods) / 2), 4))
        width = 0.8 / len(methods)
        for j, mth in enumerate(methods):
            xs, ys = [], []
            for i, k in enumerate(keep):
                base, v = meds[k].get("shortest", (None,))[0], meds[k].get(mth, (None,))[0]
                if base and v is not None:
                    xs.append(i + j * width)
                    ys.append(100 * (base - v) / base)
            label = {"sa-fill": "simulated annealing [OD-2]"}.get(mth, mth)
            ax.bar(xs, ys, width=width, label=label, color=examplekit.SERIES[j % len(examplekit.SERIES)])
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(keep))])
        ax.set_xticklabels([f"{c:,} cars\n{T:g} s" for T, c in keep])
        ax.set_ylabel("congestion below all-shortest (%)")
        ax.legend()
        fig.text(0.01, -0.06, RULE, fontsize=8, color=examplekit.MUTED)
        fig.savefig(out / "congestion.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        cars = max(c for _, c in keep)
        tr = [r for r in rows if r.get("kind") == "solve" and r.get("arm") == OURS and r.get("cars") == cars
              and r.get("trajectory") and r.get("seed") == 1]
        if tr:
            r = max(tr, key=lambda r: r["budget_s"])
            fig, ax = plt.subplots(figsize=(5, 4))
            t, e = zip(*r["trajectory"])
            ax.step(t, e, where="post", color=examplekit.SERIES[0])
            ax.set_xlabel("service search seconds")
            ax.set_ylabel("congestion (sum of squared loads)")
            ax.set_title(f"{cars:,} cars, {r['budget_s']:g} s")
            fig.savefig(out / "trajectory.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
