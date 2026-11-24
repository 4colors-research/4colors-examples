"""Charts for the exact-constraints post, from the results.jsonl files only.

    python plot.py runs/penalty runs/exact --out figures

feasibility_vs_lambda.png (the hero): one panel per (problem, size), the share of penalty answers that meet every
    constraint (judged in plain Python) against the penalty weight lam (log scale), with the exact-row share as a
    dashed line. Penalty and exact rows are two ways to send the same problem to the same solver, so this is a
    before/after picture, not a comparison with another product.
<problem>_n<size>_penalty_sweep.png: per (problem, size), the feasible share (left) and the median gap to the
    exact optimum over the feasible answers (right), against lam; exact rows dashed.
cost_of_search.md: per (problem, size), what each way costs: jobs run, variables, quadratic terms, rows, slack bits,
    and the first lam factor whose answers were all feasible.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
import examplekit


def summarize(rows):
    """{(problem, size): {"penalty": [(factor, feasible_share, median_gap, n), ...],
                          "exact": (feasible_share, median_gap, n) or None}}"""
    out = {}
    by = defaultdict(list)
    for r in rows:
        if r.get("kind") in ("penalty", "exact"):
            by[(r["problem"], r["size"], r["kind"], r.get("lam_factor"))].append(r)
    for (problem, size, kind, factor), rs in by.items():
        share = sum(bool(r["feasible"]) for r in rs) / len(rs)
        gap = examplekit.median([r["gap"] for r in rs if r["feasible"]])
        d = out.setdefault((problem, size), {"penalty": [], "exact": None})
        if kind == "penalty":
            d["penalty"].append((factor, share, gap, len(rs)))
        else:
            d["exact"] = (share, gap, len(rs))
    for d in out.values():
        d["penalty"].sort()
    return out


def first_all_feasible(penalty):
    """The smallest lam factor whose answers were all feasible (None if there is none)."""
    return next((f for f, share, _, _ in penalty if share == 1.0), None)


def cost_table(rows, summary) -> str:
    """The cost of each way, as a Markdown table: model sizes come from the rows the scripts wrote."""
    size_of = {}
    for r in rows:
        if r.get("kind") in ("penalty", "exact"):
            size_of.setdefault((r["problem"], r["size"], r["kind"]), (r["n_vars"], r["n_quad"], r["n_rows"]))
    lines = ["| problem | size | way | jobs | variables | quadratic terms | rows | slack bits | first lam factor with every answer feasible |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (problem, size), d in sorted(summary.items()):
        for kind in ("penalty", "exact"):
            if (problem, size, kind) not in size_of:
                continue
            nv, nq, nr = size_of[(problem, size, kind)]
            jobs = sum(t[3] for t in d["penalty"]) if kind == "penalty" else (d["exact"] or (0, 0, 0))[2]
            slack = nv - size if problem == "knapsack" else 0
            first = first_all_feasible(d["penalty"]) if kind == "penalty" else "none needed"
            lines.append(f"| {problem} | {size} | {kind} | {jobs} | {nv} | {nq} | {nr} | {slack} | {first} |")
    return "\n".join(lines) + "\n"


def unit(problem):
    return "cost" if problem == "assignment" else "value"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="results.jsonl files or their directories")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    rows = examplekit.load_rows(a.inputs)
    summary = summarize(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(summary)

    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.3), squeeze=False)
    for ax, key in zip(axes[0], keys):
        d = summary[key]
        ax.plot([t[0] for t in d["penalty"]], [100 * t[1] for t in d["penalty"]], marker="o",
                color=examplekit.SERIES[0], label="penalty, lam swept")
        if d["exact"] is not None:
            ax.axhline(100 * d["exact"][0], ls="--", color=examplekit.SERIES[1], label="exact rows, no lam")
        ax.set_xscale("log")
        ax.set_ylim(-5, 105)
        ax.set_title(f"{key[0]}, n = {key[1]}", loc="left", fontsize=9)
        ax.set_xlabel(f"lam (x largest {unit(key[0])})")
    axes[0][0].set_ylabel("answers meeting every constraint (%)")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "feasibility_vs_lambda.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'feasibility_vs_lambda.png'}")

    for (problem, size), d in summary.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
        f = [t[0] for t in d["penalty"]]
        ax1.plot(f, [100 * t[1] for t in d["penalty"]], marker="o", color=examplekit.SERIES[0], label="penalty")
        gaps = [(t[0], 100 * t[2]) for t in d["penalty"] if t[2] is not None]
        if gaps:
            ax2.plot(*zip(*gaps), marker="o", color=examplekit.SERIES[0], label="penalty")
        if d["exact"] is not None:
            ax1.axhline(100 * d["exact"][0], ls="--", color=examplekit.SERIES[1], label="exact rows")
            if d["exact"][1] is not None:
                ax2.axhline(100 * d["exact"][1], ls="--", color=examplekit.SERIES[1], label="exact rows")
        for ax in (ax1, ax2):
            ax.set_xscale("log")
            ax.set_xlabel(f"penalty weight lam (x largest {unit(problem)})")
        ax1.set_ylabel("feasible answers (%)")
        ax1.set_ylim(-5, 105)
        ax2.set_ylabel("median gap to optimum (%)")
        ax1.legend(loc="lower right")
        fig.suptitle(f"{problem}, n = {size}: penalty weight versus exact rows", color=examplekit.INK)
        fig.tight_layout()
        name = out / f"{problem}_n{size}_penalty_sweep.png"
        fig.savefig(name, dpi=150)
        plt.close(fig)
        print(f"wrote {name}")

    table = cost_table(rows, summary)
    (out / "cost_of_search.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
