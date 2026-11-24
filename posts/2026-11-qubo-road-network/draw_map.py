"""Draw a partition on the road map from the files run.py saved (no API, no OSM download).

    python draw_map.py results/utrecht/solutions/utrecht-bisection-unit-graph.npz \
        results/utrecht/solutions/utrecht-bisection-unit-4colors-T60-s0.npy --out figures/utrecht-split.png

Roads inside one side are drawn in that side's color; cut roads in dark ink, slightly thicker.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import examplekit as bk
import roads


def draw(graph_npz, answer_npy, out, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    g = np.load(graph_npz)
    n = len(g["x"])
    x = np.unpackbits(np.load(answer_npy))[:n]
    u, v = g["u"], g["v"]
    seg = np.stack([np.stack([g["x"][u], g["y"][u]], 1), np.stack([g["x"][v], g["y"][v]], 1)], 1)
    cut = x[u] != x[v]
    fig, ax = plt.subplots(figsize=(8, 8))
    for side in (0, 1):
        k = ~cut & (x[u] == side)
        ax.add_collection(LineCollection(seg[k], colors=bk.SERIES[side], linewidths=0.4, rasterized=True,
                                         label=f"side {side} ({int((x == side).sum()):,} nodes)"))
    ax.add_collection(LineCollection(seg[cut], colors=bk.INK, linewidths=0.8, rasterized=True,
                                     label=f"cut roads ({int(cut.sum()):,})"))
    ax.autoscale(); ax.set_aspect("equal"); ax.set_axis_off()
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_title(title or Path(answer_npy).stem, loc="left", fontsize=11)
    ax.text(1.0, 0.0, roads.OSM_CREDIT, transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=bk.MUTED)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graph_npz"); p.add_argument("answer_npy"); p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    a = p.parse_args()
    draw(a.graph_npz, a.answer_npy, a.out, a.title)
