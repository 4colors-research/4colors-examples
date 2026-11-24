"""Tables, the pre-registered selection rule and the figures, from the recorded rows only.

    python plot.py runs/structure/results.jsonl runs/structure-rivals/results.jsonl --out figures

A cell is (problem, n, budget). Score per run: knapsack gap to the exact optimum (an infeasible or missing answer
scores +inf); number partitioning the difference (missing = +inf). Median over solver seeds, then over instances.
Selection rule (printed under every chart): a cell is shown when the 4colors penalty-QUBO arm answered feasibly on
every run and its median beats the best same-time open-source setting's median by more than the tie band
(knapsack 0.05 points of gap; number partitioning: a strictly smaller difference). A setting that ran past 1.1 x the
budget on any run is not same-time and is left out of that cell. Cells within 2 x the knapsack tie band need 10
seeds before they are shown.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import examplekit as bk

OURS = "4colors-penalty"
TIE = {"knapsack": 0.0005, "numpart": 0.0}
RULE = ("Selection rule: we ran {n_cells} (problem, size, budget) cells; this chart shows the {k} where the 4colors "
        "answer to the plain penalty QUBO beats the best open-source setting at the same wall time (knapsack: by more "
        "than 0.05 points of gap to the exact optimum; number partitioning: a smaller difference).")


def run_score(r) -> float:
    if r.get("error") or r.get("error_code"):
        return math.inf
    if r["problem"] == "knapsack":
        return math.inf if r.get("gap") is None else r["gap"]
    return math.inf if r.get("diff") is None else float(r["diff"])


def cells(rows):
    per = defaultdict(lambda: defaultdict(list))
    flags = defaultdict(lambda: {"same_time": True, "seeds": set()})
    for r in rows:
        if r.get("kind") != "solve":
            continue
        key = (r["problem"], r["n"], float(r["budget_s"]), r["arm"])
        per[key][r["instance"]].append(run_score(r))
        flags[key]["seeds"].add(r.get("solver_seed"))
        if r.get("over_budget"):
            flags[key]["same_time"] = False
    out = defaultdict(dict)
    for key, inst in per.items():
        allv = [v for vs in inst.values() for v in vs]
        out[key[:3]][key[3]] = {"median": bk.median([bk.median(v) for v in inst.values()]),
                                "all_answered": all(math.isfinite(v) for v in allv),
                                "same_time": flags[key]["same_time"], "seeds": len(flags[key]["seeds"])}
    return dict(out)


def select_wins(rows):
    table, n_cells = [], 0
    for (problem, n, T), arms in sorted(cells(rows).items()):
        ours = arms.get(OURS)
        if ours is None:
            continue
        n_cells += 1
        rivals = {a: v for a, v in arms.items() if not a.startswith("4colors") and v["same_time"]}
        if not ours["all_answered"] or not rivals:
            continue
        best = min(rivals, key=lambda a: rivals[a]["median"])
        margin = rivals[best]["median"] - ours["median"]
        if margin > TIE[problem]:
            close = problem == "knapsack" and margin < 2 * TIE[problem]
            table.append({"problem": problem, "n": n, "budget_s": T, "ours": ours["median"], "best_rival": best,
                          "rival": rivals[best]["median"],
                          "needs_10_seeds": close and min(ours["seeds"], rivals[best]["seeds"]) < 10})
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
        fh.write("| problem | n | budget (s) | 4colors (penalty QUBO) | best open-source setting | its score |\n"
                 "|---|---|---|---|---|---|\n")
        for t in shown:
            f = (lambda v: f"{100 * v:.3f} %") if t["problem"] == "knapsack" else (lambda v: f"{v:.0f}")
            fh.write(f"| {t['problem']} | {t['n']} | {t['budget_s']:g} | {f(t['ours'])} | {t['best_rival']} | "
                     f"{f(t['rival'])} |\n")
        fh.write(f"\n{rule}\n")
    print(rule)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bk.plot_style()
    refs = {r["instance"]: r for r in rows if r.get("kind") == "reference"}
    for problem, fname, ylab in (("knapsack", "knapsack_gap.png", "gap to the exact optimum (%)"),
                                 ("numpart", "numpart_difference.png", "difference of the two sums")):
        ts = [t for t in shown if t["problem"] == problem]
        budgets = sorted({t["budget_s"] for t in ts})
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for k, T in enumerate(budgets):
            sel = sorted((t for t in ts if t["budget_s"] == T), key=lambda t: t["n"])
            scale = 100 if problem == "knapsack" else 1
            ax.plot([t["n"] for t in sel], [scale * t["ours"] for t in sel], marker=bk.MARKERS[k],
                    label=f"4colors, {T:g} s")
            ax.plot([t["n"] for t in sel], [scale * t["rival"] for t in sel], marker=bk.MARKERS[k], linestyle="--",
                    label=f"best open-source, {T:g} s")
        if problem == "numpart":
            kk = defaultdict(list)
            for r in refs.values():
                if r["problem"] == "numpart":
                    kk[r["n"]].append(r["kk"])
            ns = sorted(kk)
            if ns:
                ax.plot(ns, [bk.median(kk[n]) for n in ns], color=bk.MUTED, linestyle=":",
                        label="Karmarkar-Karp (reference)")
            ax.set_yscale("symlog", linthresh=1)
        ax.set_xscale("log")
        ax.set_xlabel("items" if problem == "knapsack" else "numbers")
        ax.set_ylabel(ylab)
        bk.style_axes(ax)
        ax.legend(frameon=False, fontsize=7)
        fig.text(0.01, -0.08, rule, fontsize=7, wrap=True)
        fig.savefig(out / fname, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Figure 3: our own two arms on the knapsack, the same instances: the plain penalty QUBO and the explicit row.
    c = cells(rows)
    pairs = sorted((k, v) for k, v in c.items() if k[0] == "knapsack" and OURS in v and "4colors-rows" in v)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for k, T in enumerate(sorted({key[2] for key, _ in pairs})):
        sel = [(key[1], v) for key, v in pairs if key[2] == T]
        ax.plot([n for n, _ in sel], [100 * v[OURS]["median"] for _, v in sel], marker=bk.MARKERS[k],
                label=f"penalty QUBO, {T:g} s")
        ax.plot([n for n, _ in sel], [100 * v["4colors-rows"]["median"] for _, v in sel], marker=bk.MARKERS[k],
                linestyle="--", label=f"explicit row, {T:g} s")
    ax.set_xscale("log")
    ax.set_xlabel("items")
    ax.set_ylabel("gap to the exact optimum (%)")
    bk.style_axes(ax)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(out / "penalty_vs_row.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return table


if __name__ == "__main__":
    main()
