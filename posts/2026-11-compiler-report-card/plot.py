"""Figures and the selection file of the report card, from the run records only.

    python plot.py RECORD_DIR [RECORD_DIR ...] --figs figs [--families-run 29]

Writes figs/fig_card.png (hero), figs/fig_counts.png, figs/fig_depth.png, figs/selection.md and figs/card.md.
Only cells the selection rule shows (report.py) are drawn.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit
import report

NOTE = "Shown: cells where 4colors beat Qiskit O3 by more than 2 % (rule under the chart). Gate counts, not a hardware run."


def fig_card(by, path):
    """One row per shown cell, sorted by margin: % fewer two-qubit gates than Qiskit O3."""
    plt = examplekit.plot_style()
    keys = report.shown(by)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * max(len(keys), 3) + 1.2))
    y = list(range(len(keys)))[::-1]
    vals = [by[k]["pct_fewer"] for k in keys]
    ax.hlines(y, 0, vals, color=examplekit.GRID, linewidth=2)
    ax.plot(vals, y, "o", color=examplekit.SERIES[0])
    for yy, v in zip(y, vals):
        ax.text(v + 0.5, yy, f"{v:.0f} %", va="center", fontsize=8, color=examplekit.INK_2)
    ax.set_yticks(y, [f"{k[0]}  ({k[1]} q)" for k in keys], fontsize=8)
    ax.set_xlabel("fewer two-qubit gates than Qiskit O3, same 10 s (%)")
    ax.set_xlim(0, max(vals + [10]) * 1.15)
    ax.set_title("Which circuits gain most", loc="left")
    fig.text(0.01, 0.005, NOTE, fontsize=7, color=examplekit.INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _bars(by, path, field_rival, field_ours, ylabel, title, with_input=False):
    import numpy as np

    plt = examplekit.plot_style()
    keys = report.shown(by)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(keys) + 2), 4))
    x = np.arange(len(keys))
    w = 0.38
    ax.bar(x - w / 2, [by[k][field_rival] for k in keys], w, label="Qiskit O3", color=examplekit.SERIES[1])
    ax.bar(x + w / 2, [by[k][field_ours] for k in keys], w, label="4colors, 10 s", color=examplekit.SERIES[0])
    if with_input:
        ax.plot(x, [by[k]["input_2q"] for k in keys], "_", ms=18, color=examplekit.INK, label="input (before routing)")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{k[0]}\n{k[1]} q" for k in keys], fontsize=7, rotation=60, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.005, NOTE, fontsize=7, color=examplekit.INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("records", nargs="+")
    p.add_argument("--figs", required=True)
    p.add_argument("--budget", type=float, default=10.0)
    p.add_argument("--families-run", type=int, default=None, help="families in the run (default: from the rows)")
    a = p.parse_args(argv)
    rows = examplekit.load_rows(a.records)
    by = report.cells(rows, a.budget)
    n_run = a.families_run or len({r["family"] for r in rows if r.get("kind") == "circuit"})
    os.makedirs(a.figs, exist_ok=True)
    with open(os.path.join(a.figs, "selection.md"), "w") as f:
        f.write(report.selection_md(by, n_run) + "\n")
    with open(os.path.join(a.figs, "card.md"), "w") as f:
        f.write(report.card_table(by) + "\n\n" + report.caption(by, n_run) + "\n")
    if report.shown(by):
        fig_card(by, os.path.join(a.figs, "fig_card.png"))
        _bars(by, os.path.join(a.figs, "fig_counts.png"), "median_rival", "median_ours", "two-qubit gates (median)",
              "Two-qubit gates, shown cells", with_input=True)
        _bars(by, os.path.join(a.figs, "fig_depth.png"), "median_depth_rival", "median_depth_ours",
              "two-qubit depth (median)", "Two-qubit depth, shown cells")
    print(report.card_table(by))
    print(report.caption(by, n_run))


if __name__ == "__main__":
    main()
