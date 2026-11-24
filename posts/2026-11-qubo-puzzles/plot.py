"""Pictures of the answers, from results.jsonl only (the first valid answer of each kind).

    python plot.py runs/puzzles --out figures

usmap_k3.png / usmap_k4.png: states at their centers, borders as lines, conflicting borders drawn thick.
nqueens_<n>.png: the board. sudoku.png: the filled grid with the clues in dark ink.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import examplekit
import puzzles as pz


def first(rows, **match):
    for r in rows:
        if r.get("kind") == "solve" and r.get("answer") is not None and all(r.get(k) == v for k, v in match.items()):
            return r
    return None


def draw_map(plt, row, path):
    nodes, edges = pz.us_edges()
    colors = row["answer"]
    bad = {tuple(e) for e in row["conflicts"]}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for u, v in edges:
        (la1, lo1), (la2, lo2) = pz.US_LATLON[u], pz.US_LATLON[v]
        hit = (u, v) in bad
        ax.plot([lo1, lo2], [la1, la2], color=examplekit.INK if hit else examplekit.GRID, lw=3 if hit else 1, zorder=1)
    for s in nodes:
        la, lo = pz.US_LATLON[s]
        c = colors[s]
        ax.scatter([lo], [la], s=260, color=examplekit.SERIES[c] if c >= 0 else examplekit.SURFACE,
                   edgecolor=examplekit.SURFACE, linewidth=2, zorder=2)
        ax.annotate(s, (lo, la), ha="center", va="center", fontsize=6, color=examplekit.INK, zorder=3)
    ax.set_axis_off()
    ax.set_title(f"US map, {row['colors']} colors: {row['n_conflicts']} same-colored border(s)",
                 color=examplekit.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_queens(plt, row, path):
    n = row["size"]
    fig, ax = plt.subplots(figsize=(4, 4))
    for r in range(n):
        for c in range(n):
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color=examplekit.GRID if (r + c) % 2 else examplekit.SURFACE))
    for r, c in row["answer"]:
        ax.scatter([c + 0.5], [r + 0.5], s=max(20, 2400 / n ** 2 * 8), color=examplekit.SERIES[0])
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal"); ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_sudoku(plt, row, clues, path):
    size = len(row["answer"])
    b = int(round(size ** 0.5))
    fig, ax = plt.subplots(figsize=(4, 4))
    for i in range(size + 1):
        lw = 2 if i % b == 0 else 0.5
        ax.plot([0, size], [i, i], color=examplekit.INK_2, lw=lw)
        ax.plot([i, i], [0, size], color=examplekit.INK_2, lw=lw)
    for r in range(size):
        for c in range(size):
            given = clues[r][c] != 0
            ax.text(c + 0.5, size - r - 0.5, str(row["answer"][r][c]), ha="center", va="center",
                    fontsize=12, color=examplekit.INK if given else examplekit.SERIES[0],
                    fontweight="bold" if given else "normal")
    ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_aspect("equal"); ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    rows = examplekit.load_rows(a.inputs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in sorted({r["colors"] for r in rows if r.get("puzzle") == "usmap" and r.get("kind") == "solve"}):
        row = first(rows, puzzle="usmap", colors=k, variant="conflicts_in_objective")
        if row:
            draw_map(plt, row, out / f"usmap_k{k}.png")
    for n in sorted({r["size"] for r in rows if r.get("puzzle") == "nqueens" and r.get("kind") == "solve"}):
        row = first(rows, puzzle="nqueens", size=n, valid=True)
        if row:
            draw_queens(plt, row, out / f"nqueens_{n}.png")
    ref = next((r for r in rows if r.get("kind") == "reference" and r.get("puzzle") == "sudoku"), None)
    row = first(rows, puzzle="sudoku", valid=True)
    if row and ref:
        draw_sudoku(plt, row, ref["grid"], out / "sudoku.png")
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
