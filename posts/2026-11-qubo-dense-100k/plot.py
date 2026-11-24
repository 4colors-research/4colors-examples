"""Tables, the pre-registered selection rule and the figures, from the recorded rows only.

    python plot.py runs/dense/results.jsonl runs/dense-rivals/results.jsonl --out figures

Gap = (energy - best found in this study) / |best found|, per instance; median over solver seeds, then over the
instance seeds of a cell. A cell is (family label, budget). The selection rule (printed under every chart):
a cell is shown when 4colors' median gap is lower than the best rival setting's median gap at the same budget by
more than TIE (0.05 percentage points), every 4colors seed returned an answer, and the rival setting stayed within
1.1 x the budget on every run (otherwise it is not a same-time setting and is left out of that cell). Cells closer
than 2 x TIE need 10 seeds before they can be shown (`needs_10_seeds`).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import examplekit as bk
import dense as D

TIE = 0.0005
RULE = ("Selection rule: we ran {n_cells} (family, budget) cells; this chart shows the {k} where 4colors' median gap "
        "is lower than the best open-source setting's at the same wall time by more than 0.05 points. "
        "Gap is to the best answer found by any solver in this study.")


def label(r) -> str:
    return f"degree {r['degree']}" if r["family"] == "reg" else f"fully dense n={r['n']:,}"


def cells(rows):
    """{(label, budget): {arm: {"gaps": [...per instance median...], "answered": bool, "same_time": bool, "seeds": k}}}"""
    best = D.best_found(rows)
    per = defaultdict(lambda: defaultdict(list))              # (label, T, arm, instance) -> gaps
    flags = defaultdict(lambda: {"answered": True, "same_time": True, "seeds": set()})
    for r in rows:
        if r.get("event") not in ("solve", "refused", "error"):
            continue
        key = (label(r), float(r["budget_s"]), r.get("arm", "4colors"))
        f = flags[key]
        f["seeds"].add(r.get("solver_seed"))
        if r.get("event") != "solve" or r.get("energy") is None:
            f["answered"] = False
            continue
        if r.get("over_budget"):
            f["same_time"] = False
        per[key][r["instance"]].append(D.gap(r["energy"], best[r["instance"]]))
    out = defaultdict(dict)
    for key, f in flags.items():
        lab, T, arm = key
        inst_medians = [bk.median(g) for g in per[key].values()]
        out[(lab, T)][arm] = {"gap": bk.median(inst_medians) if inst_medians else None, "answered": f["answered"],
                              "same_time": f["same_time"], "seeds": len(f["seeds"])}
    return dict(out)


def select_wins(rows, tie: float = TIE):
    """The rows of the win table, plus the count of cells run (for the rule line)."""
    table, n_cells = [], 0
    for (lab, T), arms in sorted(cells(rows).items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ours = arms.get("4colors")
        if ours is None:
            continue
        n_cells += 1
        rivals = {a: v for a, v in arms.items() if a != "4colors" and v["answered"] and v["same_time"]
                  and v["gap"] is not None}
        if not ours["answered"] or ours["gap"] is None or not rivals:
            continue
        best_arm = min(rivals, key=lambda a: rivals[a]["gap"])
        margin = rivals[best_arm]["gap"] - ours["gap"]
        if margin > tie:
            table.append({"cell": lab, "budget_s": T, "ours_gap": ours["gap"], "best_rival": best_arm,
                          "rival_gap": rivals[best_arm]["gap"], "margin": margin,
                          "needs_10_seeds": margin < 2 * tie and min(ours["seeds"], rivals[best_arm]["seeds"]) < 10})
    return table, n_cells


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = bk.load_rows(a.results)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table, n_cells = select_wins(rows)
    shown = [t for t in table if not t["needs_10_seeds"]]
    rule = RULE.format(n_cells=n_cells, k=len(shown))
    with open(out / "wins.md", "w") as fh:
        fh.write("| cell | budget (s) | 4colors gap (%) | best open-source setting | its gap (%) |\n|---|---|---|---|---|\n")
        for t in shown:
            fh.write(f"| {t['cell']} | {t['budget_s']:g} | {100 * t['ours_gap']:.3f} | {t['best_rival']} | "
                     f"{100 * t['rival_gap']:.3f} |\n")
        fh.write(f"\n{rule}\n")
    print(rule)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bk.plot_style()
    # Figure 1: the shown cells, ours against the best same-time open-source setting, one panel per family label.
    labs = sorted({t["cell"] for t in shown})
    fig, axes = plt.subplots(1, max(1, len(labs)), figsize=(3.2 * max(1, len(labs)), 3.2), squeeze=False)
    for ax, lab in zip(axes[0], labs or ["(no cell shown)"]):
        ts = sorted((t for t in shown if t["cell"] == lab), key=lambda t: t["budget_s"])
        x = [t["budget_s"] for t in ts]
        ax.plot(x, [100 * t["ours_gap"] for t in ts], marker="o", label="4colors")
        ax.plot(x, [100 * t["rival_gap"] for t in ts], marker="s", label="best open-source setting")
        ax.set_xscale("log")
        ax.set_title(lab)
        ax.set_xlabel("budget (s)")
        bk.style_axes(ax)
    axes[0][0].set_ylabel("gap to best found (%)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.text(0.01, -0.06, rule, fontsize=7, wrap=True)
    fig.savefig(out / "gap_by_budget.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: time to the first answer, ours, by degree (the cost of loading the terms), with no rival.
    first = defaultdict(list)
    for r in rows:
        if r.get("event") == "solve" and r.get("arm") == "4colors" and r.get("first_answer_s") is not None \
                and r["family"] == "reg":
            first[r["degree"]].append(r["first_answer_s"])
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ds = sorted(first)
    ax.plot(ds, [bk.median(first[d]) for d in ds], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("degree (n = 100,000)")
    ax.set_ylabel("search time of the first answer (s)")
    bk.style_axes(ax)
    fig.savefig(out / "first_answer_by_degree.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return table


if __name__ == "__main__":
    main()
