"""Table and figures for the depth-or-gates post, from results.jsonl only.

    python plot.py results/depth --figs figs

fig_scatter.png  every streamed incumbent as (two-qubit gates, two-qubit depth), colored by when it arrived;
                 circles for objective="two_qubit", triangles for objective="depth"; Qiskit's two picks as squares.
                 Two-qubit depth is the main axis: rz is virtual on this chip and single-qubit pulses are short, so
                 the two-qubit layers set most of the circuit's duration.
fig_pairs.png    the pairs the selection rule shows: Qiskit's fewest-gates pick beside ours at objective="two_qubit"
                 (two-qubit gates), and Qiskit's smallest-two-qubit-depth pick beside ours at objective="depth"
                 (two-qubit depth); medians over seeds
selection.md     which pairs are shown, which are left out and why, and the line that goes under fig_pairs

Selection rule: a pair (circuit, budget, objective) is shown when ours has a strictly smaller median on the pair's
metric than Qiskit's matching pick at the same budget, and no Qiskit seed overran the budget in that cell
(an over-budget cell is reported apart, never as a win or a loss). Ties, losses and refusals stay in the record.
"""
import argparse
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck

MARKERS = {"two_qubit": "o", "depth": "^"}
QISKIT_SET, BY_DEPTH = "full", "full, by two-qubit depth"
PAIRS = [(QISKIT_SET, "two_qubit", "two_qubit_gates", "fewest two-qubit gates"),
         (BY_DEPTH, "depth", "two_qubit_depth", "smallest two-qubit depth")]


def median_of(rows, metric, **match):
    return examplekit.median([r.get(metric) for r in ck.results(rows, **match) if r.get(metric) is not None])


def select_pairs(rows):
    """(shown, left): shown is a list of dicts (circuit_id, budget_s, objective, metric, ours, qiskit); left is a
    list of (circuit_id, budget_s, objective, why)."""
    shown, left = [], []
    res = ck.results(rows)
    for cid in sorted({r["circuit_id"] for r in res}):
        for b in sorted({r["budget_s"] for r in res if r["circuit_id"] == cid}):
            for qset, objective, metric, _ in PAIRS:
                ours = median_of(rows, metric, circuit_id=cid, budget_s=b, arm="4colors", coupler_set=objective)
                q = ck.results(rows, circuit_id=cid, budget_s=b, arm="qiskit", coupler_set=qset)
                theirs = median_of(rows, metric, circuit_id=cid, budget_s=b, arm="qiskit", coupler_set=qset)
                if ours is None or theirs is None:
                    left.append((cid, b, objective, "an arm has no answer"))
                elif any(r.get("over_budget") for r in q):
                    left.append((cid, b, objective, "a Qiskit seed overran the budget: reported apart"))
                elif not ours < theirs:
                    left.append((cid, b, objective, "tie or behind"))
                else:
                    shown.append({"circuit_id": cid, "budget_s": b, "objective": objective, "metric": metric,
                                  "ours": ours, "qiskit": theirs})
    return shown, left


def selection_md(shown, left, total):
    lines = ["# Pairs shown in fig_pairs.png", "",
             "Under the chart: \"Pairs where the 4colors compile at the matching objective beat Qiskit O3's matching "
             "pick from its seeds at the same budget (median of seeds; Qiskit on the published calibration as-is). "
             "The full record, every pair included, is linked from the run id.\"", "",
             f"Internal (not for publication): pairs shown {len(shown)} of {total}.", "",
             "| circuit | budget (s) | objective | metric | 4colors | Qiskit |", "|---|---|---|---|---|---|"]
    lines += [f"| {s['circuit_id']} | {s['budget_s']:g} | {s['objective']} | {s['metric']} | {s['ours']} | "
              f"{s['qiskit']} |" for s in shown]
    lines += ["", "Left out (kept in the record):", ""]
    lines += [f"- {c}, {b:g} s, {o}: {why}" for c, b, o, why in left] or ["- none"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--figs", default="figs")
    args = p.parse_args(argv)
    rows = examplekit.load_rows(args.inputs)
    os.makedirs(args.figs, exist_ok=True)
    plt = examplekit.plot_style()
    from matplotlib.colors import LinearSegmentedColormap

    res = ck.results(rows)
    print(ck.table(res, ("circuit_id", "arm", "coupler_set", "budget_s"), ("two_qubit_gates", "two_qubit_depth", "depth")))
    shown, left = select_pairs(rows)
    with open(os.path.join(args.figs, "selection.md"), "w") as f:
        f.write(selection_md(shown, left, len(shown) + len(left)))
    cids = sorted({r["circuit_id"] for r in res})
    inc = [r for r in rows if r.get("kind") == "incumbent" and r.get("two_qubit_gates") is not None]
    tmax = max([r["t_client_s"] for r in inc] or [1.0])
    cmap = LinearSegmentedColormap.from_list("time", ["#cde2fb", "#3987e5", "#0d366b"])

    fig, axes = plt.subplots(1, len(cids), figsize=(4.6 * len(cids), 4.2), squeeze=False)
    sc = None
    for ax, cid in zip(axes[0], cids):
        for obj, marker in MARKERS.items():
            pts = [r for r in inc if r["circuit_id"] == cid and r["coupler_set"] == obj]
            if pts:
                sc = ax.scatter([r["two_qubit_gates"] for r in pts], [r["two_qubit_depth"] for r in pts],
                                c=[r["t_client_s"] for r in pts], cmap=cmap, vmin=0, vmax=tmax, marker=marker,
                                s=36, edgecolors=examplekit.SURFACE, linewidths=1, label=f'4colors, objective="{obj}"')
        for (qset, _, _, label), color in zip(PAIRS, (examplekit.SERIES[1], examplekit.SERIES[3])):
            q = ck.results(rows, circuit_id=cid, arm="qiskit", coupler_set=qset)
            if q:
                ax.scatter([r["two_qubit_gates"] for r in q], [r["two_qubit_depth"] for r in q], marker="s", s=36,
                           color=color, edgecolors=examplekit.SURFACE, linewidths=1, label=f"Qiskit O3, {label}")
        ax.set_title(cid, loc="left", fontsize=9)
        ax.set_xlabel("two-qubit gates")
        ax.set_ylabel("two-qubit depth")
    axes[0][0].legend(fontsize=7)
    if sc is not None:
        cb = fig.colorbar(sc, ax=list(axes[0]), fraction=0.02, pad=0.01)
        cb.set_label("seconds since submit", color=examplekit.INK_2, fontsize=8)
    fig.savefig(os.path.join(args.figs, "fig_scatter.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), squeeze=False)
    for ax, (_, objective, metric, label) in zip(axes[0], PAIRS):
        pairs = [s for s in shown if s["objective"] == objective]
        names = [f"{s['circuit_id']}\n{s['budget_s']:g} s" for s in pairs]
        xs = range(len(pairs))
        ax.bar([x - 0.2 for x in xs], [s["ours"] for s in pairs], 0.38, color=examplekit.SERIES[0],
               label=f'4colors, objective="{objective}"')
        ax.bar([x + 0.2 for x in xs], [s["qiskit"] for s in pairs], 0.38, color=examplekit.SERIES[1],
               label=f"Qiskit O3, {label} of its seeds")
        ax.set_xticks(list(xs), names, fontsize=7)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{metric.replace('_', ' ')} (median)", loc="left", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_pairs.png"), dpi=200); plt.close(fig)
    print(f"pairs shown: {len(shown)} of {len(shown) + len(left)}; figures and selection.md in {args.figs}")


if __name__ == "__main__":
    main()
