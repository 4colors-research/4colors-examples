"""Charts for the feature-selection post, from results.jsonl only.

    python plot.py runs/fs --out figures --show german-numeric:8:0.7

1. fs_vs_greedy.png (hero): per cell (dataset, k, alpha), how much lower our median objective is than the better of
   the two greedy references, in percent. Selection rule: only cells where our median objective is strictly lower
   than both greedy references are drawn; the title says how many of all cells that is. A star marks cells where our
   median equals the enumerated optimum.
2. fs_selected.png: which features each method picked in one cell (--show dataset:k:alpha).
3. fs_auc.png: cross-validated AUC (context) for the drawn cells where ours is at least the greedy AUC; written only if
   the run used --auc.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
import examplekit

GREEDY = ("greedy_qubo", "greedy_mrmr")


def cells(rows):
    """[{dataset, k, alpha, ours (median E), greedy (best greedy E), exact, gain_pct, at_optimum, ...}]."""
    refs = defaultdict(dict)
    ours = defaultdict(list)
    aucs = defaultdict(list)
    for r in rows:
        key = (r.get("dataset"), r.get("k"), r.get("alpha"))
        if r.get("kind") == "reference":
            refs[key][r["method"]] = r
        elif r.get("kind") == "solve" and r.get("model_feasible") and r.get("energy_python") is not None:
            ours[key].append(r["energy_python"])
            if r.get("cv_auc") is not None:
                aucs[key].append(r["cv_auc"])
    out = []
    for key, vals in sorted(ours.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        g = [refs[key][m] for m in GREEDY if m in refs[key]]
        if not g:
            continue
        best_g = min(g, key=lambda r: r["energy_python"])
        med = examplekit.median(vals)
        ex = refs[key].get("exact", {}).get("energy_python")
        out.append({"dataset": key[0], "k": key[1], "alpha": key[2], "ours": med, "greedy": best_g["energy_python"],
                    "greedy_method": best_g["method"], "exact": ex,
                    "gain_pct": 100 * (best_g["energy_python"] - med) / abs(best_g["energy_python"]),
                    "at_optimum": ex is not None and abs(med - ex) <= 1e-9 * max(1.0, abs(ex)),
                    "ours_auc": examplekit.median(aucs[key]) if aucs[key] else None,
                    "greedy_auc": best_g.get("cv_auc")})
    return out


def wins(cs):
    """Cells where our median objective is strictly below every greedy reference (the selection rule)."""
    return [c for c in cs if c["ours"] < c["greedy"] - 1e-12]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--show", help="dataset:k:alpha for fs_selected.png (default: the largest win)")
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    rows = examplekit.load_rows(a.inputs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cs = cells(rows)
    shown = wins(cs)

    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.32 * len(shown) + 1.2)))
    labels = [f"{c['dataset']}  k={c['k']}  a={c['alpha']:g}" + ("  *" if c["at_optimum"] else "") for c in shown]
    ax.barh(range(len(shown)), [c["gain_pct"] for c in shown], color=examplekit.SERIES[0])
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("objective lower than the best greedy reference (%)")
    ax.set_title("cells where the QUBO answer beats both greedy references  (* = exact optimum)",
                 color=examplekit.INK, fontsize=10)
    print(f"fs_vs_greedy: {len(shown)} of {len(cs)} cells drawn (internal)")
    fig.tight_layout()
    fig.savefig(out / "fs_vs_greedy.png", dpi=150)
    plt.close(fig)

    pick = None
    if a.show:
        ds, k, al = a.show.rsplit(":", 2)
        pick = (ds, int(k), float(al))
    elif shown:
        top = max(shown, key=lambda c: c["gain_pct"])
        pick = (top["dataset"], top["k"], top["alpha"])
    if pick:
        sel = {}
        d = None
        for r in rows:
            if (r.get("dataset"), r.get("k"), r.get("alpha")) != pick or r.get("features") is None:
                continue
            d = r.get("d", d)
            if r["kind"] == "reference":
                sel[r["method"]] = r["features"]
            elif r["kind"] == "solve" and r.get("model_feasible") and r.get("seed") == 1:
                sel["4colors (seed 1)"] = r["features"]
        methods = [m for m in ("4colors (seed 1)", "exact", "greedy_qubo", "greedy_mrmr") if m in sel]
        if d and methods:
            fig, ax = plt.subplots(figsize=(min(10, 0.28 * d + 2), 0.45 * len(methods) + 1))
            for yi, mth in enumerate(methods):
                ax.scatter(sel[mth], [yi] * len(sel[mth]), marker="s", s=60, color=examplekit.SERIES[yi])
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(methods)
            ax.set_xlim(-0.5, d - 0.5)
            ax.set_xlabel("feature index")
            ax.set_title(f"{pick[0]}, k = {pick[1]}, alpha = {pick[2]:g}: features picked", color=examplekit.INK,
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "fs_selected.png", dpi=150)
            plt.close(fig)

    ctx = [c for c in shown if c["ours_auc"] is not None and c["greedy_auc"] is not None
           and c["ours_auc"] >= c["greedy_auc"]]
    if ctx:
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(ctx) + 1.4))
        ys = range(len(ctx))
        ax.scatter([c["greedy_auc"] for c in ctx], ys, color=examplekit.SERIES[1], label="best greedy")
        ax.scatter([c["ours_auc"] for c in ctx], ys, color=examplekit.SERIES[0], label="QUBO answer")
        ax.set_yticks(list(ys))
        ax.set_yticklabels([f"{c['dataset']} k={c['k']} a={c['alpha']:g}" for c in ctx], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("5-fold logistic-regression AUC (context)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "fs_auc.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
