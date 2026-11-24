"""Figures and the selection file for the paint shop post, from recorded results only.

    python plot.py RUN_SOLVE/out [RUN_MULTI/out] [RUN_COMPILE/out ...] --figs figs

Writes fig_line.png (hero), fig_changes.png, fig_multicar.png, fig_compile.png (each only if its rows exist) and
figs/selection.md: the cells each chart shows, the cells left out, and the line that goes under each chart.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import examplekit
import paintshop as ps

SOLVE_RULE = ("Sizes where the 4colors Optimizer's median color-change count is below the greedy heuristic's on "
              "every instance shown, and, where the exact optimum is known, every answer equals it. The full record, every cell included, is linked.")
MULTI_RULE = ("Sizes where every constrained answer met every class quota (checked here) and the median switch "
              "count is below the sequence-order heuristic's on every instance shown. The full record, every cell included, is linked.")
COMPILE_RULE = ("Sizes and layers where 4colors used fewer two-qubit gates than Qiskit O3 on the published "
                "calibration at the same budget (median of seeds). ESP is a model from the published calibration, "
                "not a measurement. The full record, every cell included, is linked.")


def solve_cells(rows, kind_model: str, ref: str):
    """{(n, instance): (median changes, reference, optimum, all_ok)} and the selected n values."""
    cells = {}
    for r in rows:
        if r.get("kind") != "solve" or r.get("kind_model") != kind_model:
            continue
        cells.setdefault((r["n"], r["instance"]), []).append(r)
    stats = {}
    for key, rs in cells.items():
        ch = [r.get("color_changes") for r in rs]
        ok = all(c is not None for c in ch)
        if kind_model == "multicar":
            ok = ok and all(r.get("quotas_met") for r in rs)
        opt = rs[0].get("optimum")
        if opt is not None:
            ok = ok and all(c == opt for c in ch)
        stats[key] = (examplekit.median(ch), rs[0].get(ref), opt, ok)
    sizes = sorted({n for n, _ in stats})
    shown = [n for n in sizes
             if all(st[3] and st[0] is not None and st[1] is not None and st[0] < st[1]
                    for (m, _), st in stats.items() if m == n)]
    return stats, sizes, shown


def compile_cells(rows):
    """Median two-qubit gates per (cars, layers, budget) for 4colors and Qiskit's primary arm, and the wins."""
    groups = {}
    for r in rows:
        if r.get("kind") != "result" or r.get("family") != "paint_shop_qaoa":
            continue
        arm = "4colors" if r.get("arm") == "4colors" else ("qiskit" if r.get("coupler_set") == "full" else None)
        if arm is None or r.get("two_qubit_gates") is None:
            continue
        groups.setdefault((r["cars"], r["layers"], r["budget_s"]), {}).setdefault(arm, []).append(r["two_qubit_gates"])
    med = {k: {a: examplekit.median(v) for a, v in g.items()} for k, g in groups.items()}
    shown = [k for k, m in sorted(med.items()) if "4colors" in m and "qiskit" in m and m["4colors"] < m["qiskit"]]
    return med, shown


def fig_line(rows, figs, plt):
    """The hero: one small line, painted by the optimizer's answer and by greedy (the first recorded binary row)."""
    pick = next((r for r in rows if r.get("kind") == "solve" and r.get("kind_model") == "binary"
                 and r.get("solution") and r.get("color_changes") is not None), None)
    if pick is None:
        return None
    word = ps.random_word(pick["n"], seed=pick["instance"])
    x = [int(v) for v in pick.get("solution") or []]
    if not x:
        return None
    lines = {"4colors Optimizer": ps.colors(word, x), "greedy": ps.colors(word, ps.greedy(word))}
    fig, axes = plt.subplots(len(lines), 1, figsize=(9, 2.2))
    for ax, (name, seq) in zip(axes, lines.items()):
        for i, c in enumerate(seq):
            ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color="#e34948" if c else "#ffffff", ec="#888888", lw=0.5))
            ax.text(i + 0.45, 0.5, str(word[i]), ha="center", va="center", fontsize=7)
        ax.set_xlim(0, len(seq)); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(f"{name}: {ps.count_changes(seq)} color changes", loc="left", fontsize=9)
    fig.savefig(figs / "fig_line.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "fig_line.png"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--figs", required=True)
    a = ap.parse_args(argv)
    rows = examplekit.load_rows(a.runs)
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    plt = examplekit.plot_style()
    sel = ["# Selection", ""]

    made = fig_line(rows, figs, plt)
    if made:
        sel += [f"- {made}: one recorded binary answer at n <= 20 against greedy on the same line.", ""]

    for kind_model, ref, name, rule, ylabel in (
            ("binary", "greedy", "fig_changes.png", SOLVE_RULE, "color changes per car"),
            ("multicar", "heuristic", "fig_multicar.png", MULTI_RULE, "color switches per body")):
        stats, sizes, shown = solve_cells(rows, kind_model, ref)
        if not stats:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for label, idx in (("4colors Optimizer (median)", 0), (f"{ref} (labeled heuristic)", 1)):
            pts = sorted((n, examplekit.median([st[idx] for (m, _), st in stats.items() if m == n]) / n)
                         for n in shown)
            if pts:
                ax.plot(*zip(*pts), marker="o", label=label)
        opt = sorted((n, st[2] / n) for (n, _), st in stats.items() if n in shown and st[2] is not None)
        if opt:
            ax.scatter(*zip(*opt), marker="x", color="#222222", zorder=5, label="exact optimum")
        ax.set_xscale("log"); ax.set_xlabel("cars" if kind_model == "binary" else "bodies"); ax.set_ylabel(ylabel)
        ax.legend()
        fig.savefig(figs / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        sel += [f"## {name}", "", rule, f"- internal count (not for publication): {len(shown)} of {len(sizes)}",
                f"- shown: {shown}", f"- left out: {[n for n in sizes if n not in shown]}", ""]

    med, shown = compile_cells(rows)
    if med:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for arm in ("4colors", "qiskit"):
            for layers in sorted({k[1] for k in shown}):
                for budget in sorted({k[2] for k in shown}):
                    pts = sorted((k[0], med[k][arm]) for k in shown if k[1] == layers and k[2] == budget)
                    if pts:
                        lab = "4colors" if arm == "4colors" else "Qiskit O3"
                        ax.plot(*zip(*pts), marker="o", label=f"{lab}, p={layers}, {budget:g} s")
        ax.set_xlabel("cars (qubits)"); ax.set_ylabel("two-qubit gates (median)")
        ax.legend(fontsize=7)
        fig.savefig(figs / "fig_compile.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        sel += ["## fig_compile.png", "", COMPILE_RULE, f"- internal count (not for publication): {len(shown)} of {len(med)}",
                f"- shown: {shown}", f"- left out: {[k for k in sorted(med) if k not in shown]}", ""]

    (figs / "selection.md").write_text("\n".join(sel) + "\n")
    print("\n".join(sel))


if __name__ == "__main__":
    main()
