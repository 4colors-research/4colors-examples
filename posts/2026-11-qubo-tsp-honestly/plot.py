"""Charts for the TSP post, from results.jsonl only. No race chart: the figures show the encoding and its cost.

    python plot.py runs/tsp --out figures

tsp_model_size.png: variables and quadratic terms of the QUBO against n (log scale), beside what a routing solver
    reads for the same problem (one distance per city pair), from the `size` rows (by formula, no API).
tsp_feasibility.png: the share of answers that decode to a tour, per encoding (exact rows; each penalty weight) and
    n. Both are our own encodings of the same problem; no other solver is on the chart.
tsp_encoding.png: one exact-rows answer that is a tour (the smallest n with one, seed order): the tour on the map
    and the n x n 0/1 matrix x[city, position] it came from.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
import examplekit

RULE = ("Every encoding and size of the run is shown: both lines are our own encodings of the same problem, "
        "no other solver is compared.")


def arm_name(r):
    return "exact rows" if r["arm"] == "exact" else f"penalty, lam = {r['lam_factor']:g} x max distance"


def refused(r) -> bool:
    return r.get("status") == "refused"


def refusals(rows):
    """{(arm, n): refused jobs}: listed beside the chart, never counted as answers."""
    out = defaultdict(int)
    for r in rows:
        if r.get("kind") == "solve" and refused(r):
            out[(arm_name(r), r["n"])] += 1
    return dict(out)


def summarize(rows):
    """{arm: [(n, share of answers that are tours, runs)]}, sorted by n. Refused jobs gave no answer: they are not
    in the share (see `refusals`)."""
    by = defaultdict(list)
    for r in rows:
        if r.get("kind") == "solve" and not refused(r):
            by[(arm_name(r), r["n"])].append(r)
    out = defaultdict(list)
    for (arm, n), rs in sorted(by.items(), key=lambda kv: kv[0][1]):
        out[arm].append((n, sum(bool(r["feasible"]) for r in rs) / len(rs), len(rs)))
    return dict(out)


def encoding_example(rows):
    """(solve row, points) of the smallest-n exact-rows answer that is a tour; None if there is none."""
    pts = {r["n"]: r["points"] for r in rows if r.get("kind") in ("reference", "instance")}
    tours = sorted((r for r in rows if r.get("kind") == "solve" and r["arm"] == "exact" and r["feasible"]),
                   key=lambda r: (r["n"], r["seed"]))
    for r in tours:
        if r["n"] in pts:
            return r, pts[r["n"]]
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    rows = examplekit.load_rows(a.inputs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sizes = sorted((r for r in rows if r.get("kind") == "size"), key=lambda r: r["n"])
    if sizes:
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        pen = [r for r in sizes if r["model"] == "penalty"]
        ex = [r for r in sizes if r["model"] == "exact"]
        ns = [r["n"] for r in pen]
        ax.plot(ns, [r["n_vars"] for r in pen], marker="o", color=examplekit.SERIES[0], label="QUBO variables (n^2)")
        ax.plot([r["n"] for r in ex], [r["n_quad"] for r in ex], marker="o", color=examplekit.SERIES[1],
                label="quadratic terms, exact rows")
        ax.plot(ns, [r["n_quad"] for r in pen], marker="o", ls="--", color=examplekit.SERIES[1],
                label="quadratic terms, penalties")
        ax.plot(ns, [r["routing_input"] for r in pen], marker="s", color=examplekit.INK_2,
                label="routing solver input (city pairs)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("cities n")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "tsp_model_size.png", dpi=150)
        plt.close(fig)

    s = summarize(rows)
    if s:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for k, (arm, pts) in enumerate(s.items()):
            ax.plot([t[0] for t in pts], [100 * t[1] for t in pts], marker="o",
                    color=examplekit.SERIES[k % len(examplekit.SERIES)], label=arm)
        ax.set_xlabel("cities n")
        ax.set_ylabel("answers that are tours (%)")
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=8)
        fig.text(0.0, -0.02, RULE, fontsize=7, ha="left", va="top")
        fig.tight_layout()
        fig.savefig(out / "tsp_feasibility.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    ex = encoding_example(rows)
    if ex:
        r, pts = ex
        n, tour = r["n"], r["tour"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        loop = tour + tour[:1]
        ax1.plot([pts[i][0] for i in loop], [pts[i][1] for i in loop], color=examplekit.SERIES[0], lw=1.5)
        ax1.scatter([q[0] for q in pts], [q[1] for q in pts], color=examplekit.INK_2, zorder=3, s=18)
        for i, q in enumerate(pts):
            ax1.annotate(str(i), q, fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax1.set_aspect("equal"); ax1.set_xticks([]); ax1.set_yticks([])
        ax1.set_title(f"the decoded tour, n = {n}", loc="left", fontsize=10)
        grid = [[0] * n for _ in range(n)]
        for pos, city in enumerate(tour):
            grid[city][pos] = 1
        ax2.imshow(grid, cmap="Greys", vmin=0, vmax=1)
        ax2.set_xlabel("position p"); ax2.set_ylabel("city i")
        ax2.set_title("x[i, p]: one 1 per row and per column", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "tsp_encoding.png", dpi=150)
        plt.close(fig)
    for (arm, n), k in sorted(refusals(rows).items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"refused: {arm}, n = {n}: {k} job(s) (no answer; not in tsp_feasibility.png)")
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
