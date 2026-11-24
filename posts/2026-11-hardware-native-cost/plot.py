"""Figures and the selection file for the hardware-native post, from recorded results only.

    python plot.py RUN_API/out [RUN_QISKIT/out] --figs figs

fig_tradeoff.png (hero): for each problem, compiled two-qubit gates (p = 1, 4colors) against the share of the full
problem's cut that survives, one marker per variant (full, native, random, spectral). Qiskit O3's full-graph compile
is added only where the 4colors full-graph compile used strictly fewer two-qubit gates.
fig_retained.png: share of the cut retained per sparsifier and problem (median over seeds).
fig_layers.png: full-graph two-qubit gates against QAOA layers, 4colors and Qiskit O3, shown cells only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit

VARIANTS = ("full", "native", "random", "spectral")
LAYER_RULE = ("Problems and layer counts where 4colors used fewer two-qubit gates than Qiskit O3 on the full "
              "graph, published calibration, same budget. The full record, every cell included, is linked.")
RETAINED_RULE = ("Share of the full problem's cut that the sparsified problem's answer keeps, scored on the "
                 "original graph; the reference is the best 4colors answer on the full graph over the seeds "
                 "(not a proven optimum, except where enumeration confirms it). All problems shown: this chart "
                 "compares sparsifiers, not solvers.")


def retained(rows):
    """{(problem, variant): median share of the reference cut} and {problem: reference cut}."""
    ref, cuts = {}, {}
    for r in rows:
        if r.get("kind") != "solve" or r.get("cut_on_full") is None:
            continue
        cuts.setdefault((r["problem"], r["variant"]), []).append(r["cut_on_full"])
        if r["variant"] == "full":
            ref[r["problem"]] = max(ref.get(r["problem"], r["cut_on_full"]), r["cut_on_full"])
    share = {k: examplekit.median([c / ref[k[0]] for c in v]) for k, v in cuts.items() if ref.get(k[0])}
    return share, ref


def gates(rows):
    """{(problem, variant, layers, arm): two-qubit gates} for 4colors and Qiskit's full-map arm."""
    out = {}
    for r in rows:
        if r.get("kind") != "result" or r.get("post") != "hardware-native-cost" or r.get("two_qubit_gates") is None:
            continue
        arm = "4colors" if r.get("arm") == "4colors" else ("qiskit" if r.get("coupler_set") == "full" else None)
        if arm:
            out.setdefault((r["problem"], r["variant"], r["layers"], arm), []).append(r["two_qubit_gates"])
    return {k: examplekit.median(v) for k, v in out.items()}


def layer_cells(g):
    keys = sorted({(p, lay) for (p, v, lay, a) in g if v == "full"})
    both = [k for k in keys if (k[0], "full", k[1], "4colors") in g and (k[0], "full", k[1], "qiskit") in g]
    shown = [k for k in both if g[(k[0], "full", k[1], "4colors")] < g[(k[0], "full", k[1], "qiskit")]]
    return shown, both


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--figs", required=True)
    a = ap.parse_args(argv)
    rows = examplekit.load_rows(a.runs)
    share, ref = retained(rows)
    g = gates(rows)
    shown, both = layer_cells(g)
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    plt = examplekit.plot_style()
    problems = sorted({p for p, _ in share} | {k[0] for k in g})

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for v in VARIANTS:
        pts = [(g[(p, v, 1, "4colors")], 100 * share[(p, v)]) for p in problems
               if (p, v, 1, "4colors") in g and (p, v) in share]
        if pts:
            ax.scatter(*zip(*pts), label=f"{v} (4colors compile)", zorder=3)
    q = [(g[(p, "full", 1, "qiskit")], 100 * share[(p, "full")]) for p, lay in shown if lay == 1 and (p, "full") in share]
    if q:
        ax.scatter(*zip(*q), facecolors="none", edgecolors="#666666", label="full (Qiskit O3 compile)", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("two-qubit gates after compile, p = 1"); ax.set_ylabel("cut retained on the original graph (%)")
    ax.legend(fontsize=8)
    fig.savefig(figs / "fig_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if share:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        width = 0.8 / 3
        for i, v in enumerate(("native", "random", "spectral")):
            xs = [j + (i - 1) * width for j, p in enumerate(problems) if (p, v) in share]
            ys = [100 * share[(p, v)] for p in problems if (p, v) in share]
            ax.bar(xs, ys, width=width, label=v)
        ax.set_xticks(range(len(problems))); ax.set_xticklabels(problems, rotation=30, fontsize=8)
        ax.set_ylabel("cut retained (%)"); ax.legend(fontsize=8)
        fig.savefig(figs / "fig_retained.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if shown:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for p in sorted({p for p, _ in shown}):
            for arm, style in (("4colors", "-"), ("qiskit", "--")):
                pts = sorted((lay, g[(p, "full", lay, arm)]) for pp, lay in shown if pp == p)
                ax.plot(*zip(*pts), ls=style, marker="o", label=f"{p}, {'4colors' if arm == '4colors' else 'Qiskit O3'}")
        ax.set_xlabel("QAOA layers p"); ax.set_ylabel("two-qubit gates, full graph"); ax.legend(fontsize=7)
        fig.savefig(figs / "fig_layers.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    lines = ["# Selection", "", "## fig_retained.png / fig_tradeoff.png", "", RETAINED_RULE, "",
             "| problem | variant | median share retained |", "|---|---|---|"]
    lines += [f"| {p} | {v} | {share[(p, v)]:.4f} |" for p, v in sorted(share)]
    lines += ["", "## fig_layers.png and the Qiskit points on fig_tradeoff.png", "",
              LAYER_RULE, f"- internal count (not for publication): {len(shown)} of {len(both)}",
              f"- shown: {shown}", f"- left out: {[k for k in both if k not in shown]}", ""]
    (figs / "selection.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
