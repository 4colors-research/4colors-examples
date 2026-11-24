"""Figures for the MIS-at-a-million-atoms post, from the record only.

    python plot.py runs/gr2 --out figures

- lattice.png: the smallest instance (the published 289-atom size), atoms as dots, blockade edges in light ink, the
  chosen independent set filled (rows arm, seed 1, the largest budget).
- gap_vs_size.png: median gap to the 2 x 2 block bound against the number of atoms, one panel per budget. Selection
  rule (printed under the chart): a size is drawn only where every 4colors rows-arm answer is independent and its
  median gap is strictly below the rival's median gap at the same budget; when no rival rows exist, every size with
  independent answers is drawn.
- first_answer.png: service search seconds to the first feasible answer (from the trajectory), rows arm, by size.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit

OURS = "rows"
RIVAL = "sa-fill"
RULE = ("Shown: sizes where every 4colors answer is independent and its median gap to the block bound is below the "
        "rival's at the same budget.")


def medians(rows):
    """{(arm, budget, n): (median gap, all independent)} over seeds."""
    by = defaultdict(list)
    for r in rows:
        if r.get("kind") == "solve" and r.get("gap_to_bound") is not None:
            by[(r["arm"], r["budget_s"], r["n"])].append(r)
    return {k: (examplekit.median([r["gap_to_bound"] for r in v]), all(r["independent"] for r in v))
            for k, v in by.items()}


def winning_sizes(rows):
    """{budget: [n, ...]} drawn under the selection rule."""
    med = medians(rows)
    has_rival = any(k[0] == RIVAL for k in med)
    out = defaultdict(list)
    for (arm, T, n), (gap, indep) in sorted(med.items()):
        if arm != OURS or not indep:
            continue
        if has_rival:
            rv = med.get((RIVAL, T, n))
            if rv is None or not gap < rv[0]:
                continue
        out[T].append(n)
    return dict(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = examplekit.load_rows(a.runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt = examplekit.plot_style()

    insts = sorted((r for r in rows if r.get("kind") == "instance" and r.get("positions")), key=lambda r: r["n"])
    if insts:
        inst = insts[0]
        sols = [r for r in rows if r.get("kind") == "solve" and r.get("arm") == OURS and r.get("n") == inst["n"]
                and r.get("chosen") is not None]
        sol = max(sols, key=lambda r: (r["budget_s"], -r["seed"]), default=None)
        fig, ax = plt.subplots(figsize=(6, 6))
        P = inst["positions"]
        for u, v in inst["edges"]:
            ax.plot([P[u][1], P[v][1]], [P[u][0], P[v][0]], color=examplekit.GRID, linewidth=0.6, zorder=1)
        chosen = set(sol["chosen"]) if sol else set()
        ax.scatter([q[1] for q in P], [q[0] for q in P], s=28, facecolors="white", edgecolors=examplekit.MUTED, zorder=2)
        ax.scatter([P[i][1] for i in chosen], [P[i][0] for i in chosen], s=28, color=examplekit.SERIES[0], zorder=3)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{inst['n']} atoms, {len(chosen)} in the set (block bound {inst['block_bound']})")
        fig.savefig(out / "lattice.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    keep = winning_sizes(rows)
    med = medians(rows)
    if keep:
        budgets = sorted(keep)
        fig, axes = plt.subplots(1, len(budgets), figsize=(5 * len(budgets), 4), squeeze=False)
        for ax, T in zip(axes[0], budgets):
            ns = keep[T]
            for k, (arm, label) in enumerate([(OURS, "4colors, exact rows"), ("penalty", "4colors, penalty form"),
                                              (RIVAL, "simulated annealing [OD-2]")]):
                pts = [(n, 100 * med[(arm, T, n)][0]) for n in ns if (arm, T, n) in med]
                if pts:
                    ax.plot(*zip(*pts), marker=examplekit.MARKERS[k], label=label, color=examplekit.SERIES[k])
            ax.set_xscale("log")
            ax.set_xlabel("atoms")
            ax.set_ylabel("gap to block bound (%)")
            ax.set_title(f"{T:g} s per job")
            ax.legend()
        fig.text(0.01, -0.04, RULE, fontsize=8, color=examplekit.MUTED)
        fig.savefig(out / "gap_vs_size.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    first = defaultdict(list)
    for r in rows:
        if r.get("kind") == "solve" and r.get("arm") == OURS and r.get("trajectory") and r.get("independent"):
            first[r["n"]].append(r["trajectory"][0][0])
    if first:
        fig, ax = plt.subplots(figsize=(5, 4))
        ns = sorted(first)
        ax.plot(ns, [examplekit.median(first[n]) for n in ns], marker="o", color=examplekit.SERIES[0])
        ax.set_xscale("log")
        ax.set_xlabel("atoms")
        ax.set_ylabel("search seconds to first answer (median)")
        fig.savefig(out / "first_answer.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
