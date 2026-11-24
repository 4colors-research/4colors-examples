"""Figures and a table for the BV post, from results.jsonl only.

    python plot.py results/bv [more dirs or .jsonl files] --figs figs/

fig_graphs.png  the circuit's interaction graph (a star) beside the heavy-hex device map
fig_counts.png  two-qubit gates against BV size, 4colors and Qiskit's primary arm (median over seeds), one panel
                per budget, only the sizes the selection rule picks
fig_where.png   where the two-qubit gates land on the chip, largest selected size, first seed, largest budget

Selection rule (owner, 22 Nov 2026: "choose ones where we win, but focus on the process while you can show the
win"): a (size, budget) cell is shown when the 4colors median two-qubit count is strictly below the median of
Qiskit's primary arm (`full`: FakeFez as shipped, its published calibration as-is) at the same budget, and every
4colors row of the cell passed the Clifford check (when the check ran). Ties, losses and unchecked cells stay in
results.jsonl and in selection.md, which also prints the line that goes under each chart. Qiskit's `working`
and `lower_of_both` arms are labeled extras in the table only.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck

ARMS = [("4colors", "fakefez", "4colors"), ("qiskit", "full", "Qiskit O3, published calibration as-is")]


def _medians(rows, arm, cset, budget):
    pts = {}
    for r in ck.results(rows, arm=arm, coupler_set=cset, budget_s=budget):
        if r.get("two_qubit_gates") is not None:
            pts.setdefault(r["qubits"] - 1, []).append(r["two_qubit_gates"])
    return {n: examplekit.median(v) for n, v in pts.items()}


def select_cells(rows):
    """{budget: [sizes shown]} and the list of (size, budget, why) left out, by the rule in the module docstring."""
    shown, left = {}, []
    for b in sorted({r["budget_s"] for r in ck.results(rows)}):
        ours, theirs = _medians(rows, "4colors", "fakefez", b), _medians(rows, "qiskit", "full", b)
        for n in sorted(set(ours) | set(theirs)):
            checks = [r.get("clifford_check") for r in ck.results(rows, arm="4colors", budget_s=b, qubits=n + 1)]
            if n not in ours or n not in theirs:
                left.append((n, b, "an arm has no answer"))
            elif not ours[n] < theirs[n]:
                left.append((n, b, "tie or behind"))
            elif any(c not in (None, "returns_secret") for c in checks):
                left.append((n, b, "a 4colors circuit failed the Clifford check"))
            else:
                shown.setdefault(b, []).append(n)
    return shown, left


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--figs", default="figs")
    args = p.parse_args(argv)
    rows = examplekit.load_rows(args.inputs)
    os.makedirs(args.figs, exist_ok=True)
    plt = examplekit.plot_style()

    res = ck.results(rows)
    print(ck.table(res, ("circuit_id", "arm", "coupler_set", "budget_s")))
    checked = [r for r in res if r.get("clifford_check")]
    print(f"Clifford check: {sum(r['clifford_check'] == 'returns_secret' for r in checked)} of {len(checked)} "
          f"compiled circuits return the secret")
    device = next(r for r in rows if r.get("kind") == "device")
    stars = [r for r in rows if r.get("kind") == "interaction"]

    # 1. the star and the chip
    star = max(stars, key=lambda r: r["qubits"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 2.2]})
    hub = star["qubits"] - 1
    leaves = sorted({a for a, b, _ in star["edges"]} | {b for a, b, _ in star["edges"]} - {hub})
    for k, q in enumerate(leaves):
        ang = 2 * math.pi * k / max(1, len(leaves))
        a1.plot([0, math.cos(ang)], [0, math.sin(ang)], color=examplekit.SERIES[0], lw=0.8)
    a1.scatter([0], [0], s=60, color=examplekit.INK, zorder=3)
    a1.set_aspect("equal"); a1.axis("off")
    a1.set_title(f"{star['circuit_id']}: every CNOT touches the ancilla", loc="left", fontsize=10)
    ck.draw_device(a2, device["n_qubits"], device["couplers"], failed=device["failed"],
                   title="FakeFez heavy-hex map (failed couplers dashed)")
    fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_graphs.png"), dpi=200); plt.close(fig)

    # 2. two-qubit gates against size, one panel per budget, selected sizes only
    shown, left = select_cells(rows)
    bmax = max(r["budget_s"] for r in res)
    budgets = sorted(shown) or [bmax]
    fig, axes = plt.subplots(1, len(budgets), figsize=(6.0 * len(budgets), 4), squeeze=False)
    inputs = {s["qubits"] - 1: sum(e[2] for e in s["edges"]) for s in stars}
    for ax, b in zip(axes[0], budgets):
        xs = shown.get(b, [])
        for k, (arm, cset, label) in enumerate(ARMS):
            med = _medians(rows, arm, cset, b)
            if xs:
                ax.plot(xs, [med[x] for x in xs], marker="o", color=examplekit.SERIES[k], label=label)
        if xs:
            ax.plot(xs, [inputs[x] for x in xs], ls=":", color=examplekit.INK_2, label="input CNOTs (no routing)")
        ax.set_xlabel("secret bits n"); ax.set_ylabel("two-qubit gates (median over seeds)")
        ax.set_title(f"{b:g} s per compile", loc="left")
        if xs:
            ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_counts.png"), dpi=200); plt.close(fig)
    lines = ["Cells shown: " + "; ".join(f"{b:g} s: n = {', '.join(str(n) for n in ns)}" for b, ns in sorted(shown.items()))
             if shown else "Cells shown: none", "",
             "Under each chart: sizes and budgets where 4colors used fewer two-qubit gates than Qiskit O3 on the "
             "published calibration (median of seeds) and every 4colors circuit passed the Clifford check.", "",
             "Left out (kept in results.jsonl):"] + [f"- n = {n}, {b:g} s: {why}" for n, b, why in left]
    lines += ["", "Two-qubit blocks in the shown cells (median over seeds; blocks.py classes):", "",
              "| n | budget s | arm | stand-alone SWAP blocks | CNOT-and-move blocks | CNOT blocks | CZ in SWAP blocks |",
              "|---|---|---|---|---|---|---|"]
    for b, ns in sorted(shown.items()):
        for n in ns:
            for arm, cset, label in ARMS:
                bc = [r["block_counts"] for r in ck.results(rows, arm=arm, coupler_set=cset, budget_s=b, qubits=n + 1)
                      if r.get("block_counts")]
                if bc:
                    med = lambda f: examplekit.median([f(x) for x in bc])  # noqa: E731
                    lines.append(f"| {n} | {b:g} | {label} | {med(lambda x: x['blocks']['swap'])} | "
                                 f"{med(lambda x: x['blocks']['cx_move'])} | {med(lambda x: x['blocks']['cx'])} | "
                                 f"{med(lambda x: x['two_qubit_gates']['swap'])} |")
    open(os.path.join(args.figs, "selection.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # 3. where the gates land
    big = 1 + max(shown.get(bmax) or [max(r["qubits"] for r in res) - 1])
    seed = min(r["seed"] for r in res)
    panels = [(a, c, l) for a, c, l in ARMS
              if ck.results(rows, arm=a, coupler_set=c, qubits=big, seed=seed, budget_s=bmax)]
    if panels:
        picks = [ck.results(rows, arm=a, coupler_set=c, qubits=big, seed=seed, budget_s=bmax)[0] for a, c, _ in panels]
        vmax = max([1] + [x[2] for r in picks for x in (r.get("coupler_counts") or [])])
        fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 4.4 * len(panels)), squeeze=False)
        for ax, (a, c, label), r in zip(axes[:, 0], panels, picks):
            ck.draw_device(ax, device["n_qubits"], device["couplers"], counts=r.get("coupler_counts"),
                           failed=device["failed"], vmax=vmax,
                           title=f"{label}: {r['two_qubit_gates']} two-qubit gates, "
                                 f"{r['gates_on_failed_couplers']} on failed")
        fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_where.png"), dpi=200); plt.close(fig)
    print(f"figures in {args.figs}")


if __name__ == "__main__":
    main()
