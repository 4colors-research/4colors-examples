"""Figures for the RNA folding post, from results.jsonl only.

    python plot.py runs/example runs/rand --out figures [--arcs fox2022-example]

arcs_<instance>.png: the sequence along x; the exact optimum's base pairs as arcs above the line, the 4colors
answer's (seed shown) below it. Crossing stems (pseudoknots) are drawn in the second color.
size_ladder.png: candidate stems (QUBO variables) against sequence length, one dot per instance where every seed's
answer equals the exact optimum (the selection rule; other instances are counted in the printed line, not drawn).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import examplekit
import rna


def _stems_of_pairs(pairs):
    """Group sorted base pairs back into stacked runs (for drawing only)."""
    runs, cur = [], []
    for a, b in sorted(map(tuple, pairs)):
        if cur and a == cur[-1][0] + 1 and b == cur[-1][1] - 1:
            cur.append((a, b))
        else:
            if cur:
                runs.append(cur)
            cur = [(a, b)]
    if cur:
        runs.append(cur)
    return [rna.Stem(r[0][0], r[0][1], len(r)) for r in runs]


def _draw(ax, pairs, up: bool):
    import numpy as np

    st = _stems_of_pairs(pairs)
    for s in st:
        knot = any(rna.crossing(s, t) for t in st if t != s)
        color = examplekit.SERIES[1] if knot else examplekit.SERIES[0]
        for a, b in s.pairs():
            t = np.linspace(0, np.pi, 40)
            c, r = (a + b) / 2, (b - a) / 2
            y = r * np.sin(t) * (1 if up else -1)
            ax.plot(c + r * np.cos(t), y, color=color, linewidth=1.2)


def arcs(rows, instance, out, plt):
    inst = next(r for r in rows if r.get("kind") == "instance" and r["instance"] == instance)
    sol = [r for r in rows if r.get("kind") == "solve" and r["instance"] == instance and r.get("matches_exact")]
    if not sol or inst.get("reference_status") != "exact":
        print(f"{instance}: no answer equal to an exact optimum; arcs figure not drawn")
        return None
    s = min(sol, key=lambda r: r["seed"])
    L = inst["length"]
    fig, ax = plt.subplots(figsize=(min(12, 3 + L / 8), 3.6))
    _draw(ax, inst["reference_pairs"], up=True)
    _draw(ax, s["pairs"], up=False)
    ax.axhline(0, color=examplekit.INK_2, linewidth=0.8)
    if L <= 80:
        for i, ch in enumerate(inst["sequence"]):
            ax.text(i, 0, ch, ha="center", va="center", fontsize=6, color=examplekit.INK,
                    bbox={"boxstyle": "square,pad=0.1", "fc": examplekit.SURFACE, "ec": "none"})
    ax.set_yticks([])
    ax.set_xlabel("base position")
    ax.grid(False)
    ax.text(0.0, 1.0, "exact optimum (above)", transform=ax.transAxes, fontsize=8, color=examplekit.INK_2, va="top")
    ax.text(0.0, 0.0, f"4colors answer, seed {s['seed']} (below)", transform=ax.transAxes, fontsize=8,
            color=examplekit.INK_2, va="bottom")
    ax.set_title(f"{instance}: {L} nt, {inst['n_stems']} candidate stems, H = {s['energy']:g}",
                 color=examplekit.INK, fontsize=10)
    fig.tight_layout()
    path = out / f"arcs_{instance}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def size_ladder(rows, out, plt):
    insts = {r["instance"]: r for r in rows if r.get("kind") == "instance" and r.get("reference_status")}
    by = defaultdict(list)
    for r in rows:
        if r.get("kind") == "solve" and r["instance"] in insts:
            by[r["instance"]].append(bool(r.get("matches_exact")))
    shown = [insts[k] for k, v in by.items() if v and all(v)]
    print(f"size ladder: {len(shown)} of {len(by)} instances shown (every seed equal to the exact optimum)")
    if not shown:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.scatter([r["length"] for r in shown], [r["n_stems"] for r in shown], color=examplekit.SERIES[0],
               edgecolor=examplekit.SURFACE, linewidth=2, zorder=3, label="answer = exact optimum, every seed")
    ax.set_yscale("log")
    ax.set_xlabel("sequence length (nt)")
    ax.set_ylabel("candidate stems (QUBO variables)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = out / "size_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--arcs", nargs="*", default=None, help="instances to draw as arc diagrams (default: all)")
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    rows = examplekit.load_rows(a.inputs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.arcs if a.arcs is not None else sorted({r["instance"] for r in rows if r.get("kind") == "instance"})
    for name in names:
        arcs(rows, name, out, plt)
    size_ladder(rows, out, plt)


if __name__ == "__main__":
    main()
