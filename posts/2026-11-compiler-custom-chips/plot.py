"""Tables and figures for the custom-chips post, from results.jsonl only.

    python plot.py results/chips --figs figs

Prints the topology-tax table (median extra two-qubit gates over the all-to-all chip), the design-lattice table
(UCCSD LiH on four full-size designs) and the refusals; writes fig_tax.png, fig_scale.png, fig_design.png and
selection.md.

Selection rule (owner, 22 Nov 2026): our compiler's own numbers across chips are the process and are shown for
every chip. A Qiskit number is shown only in a (circuit, chip) cell where the 4colors median two-qubit count is
strictly below Qiskit's median at the same budget; ties, losses and refused cells are left out of the tables and
charts, kept in results.jsonl and listed in selection.md with the line that goes under each chart.
"""
import argparse
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck

ARMS = [("4colors", "4colors"), ("qiskit", "Qiskit level 3")]
BASE = "all_to_all_25"


def med(rows, **m):
    return examplekit.median([r.get("two_qubit_gates") for r in ck.results(rows, **m)])


def select_cells(rows, part):
    """{(circuit_id, chip)} where 4colors is strictly ahead of Qiskit (medians), and [(cid, chip, why)] left out."""
    rs = [r for r in rows if r.get("part") == part]
    shown, left = set(), []
    for cid, chip in sorted({(r["circuit_id"], r["chip"]) for r in rs}):
        ours, theirs = med(rs, circuit_id=cid, chip=chip, arm="4colors"), med(rs, circuit_id=cid, chip=chip,
                                                                               arm="qiskit")
        if ours is None or theirs is None:
            left.append((cid, chip, "an arm has no answer (refused or not run)"))
        elif not ours < theirs:
            left.append((cid, chip, "tie or behind"))
        else:
            shown.add((cid, chip))
    return shown, left


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--figs", default="figs")
    args = p.parse_args(argv)
    rows = examplekit.load_rows(args.inputs)
    os.makedirs(args.figs, exist_ok=True)
    plt = examplekit.plot_style()
    budget = max((r["budget_s"] for r in ck.results(rows)), default=None)
    rows = ck.results(rows, budget_s=budget)
    sel = {part: select_cells(rows, part) for part in ("tax", "scale", "design")}

    tax = [r for r in rows if r.get("part") == "tax"]
    chips_ = list(dict.fromkeys(r["chip"] for r in tax))
    cids = sorted({r["circuit_id"] for r in tax})
    shown = sel["tax"][0]
    print(f"Topology tax at {budget:g} s: median extra two-qubit gates over {BASE} (Qiskit only where 4colors leads)")
    print("| circuit | arm | " + " | ".join(chips_) + " |")
    print("|" + "---|" * (2 + len(chips_)))
    for cid in cids:
        for arm, _ in ARMS:
            base = med(tax, circuit_id=cid, arm=arm, chip=BASE)
            cells = []
            for chip in chips_:
                v = med(tax, circuit_id=cid, arm=arm, chip=chip)
                if arm == "qiskit" and (cid, chip) not in shown:
                    cells.append("")
                elif v is None:
                    cells.append("refused")
                else:
                    cells.append(f"{v - base:+g}" if base is not None else f"{v:g}")
            print(f"| {cid} | {arm} | " + " | ".join(cells) + " |")

    design = [r for r in rows if r.get("part") == "design"]
    dshown = sel["design"][0]
    if design:
        print("\nUCCSD LiH (12 qubits) on full-size design lattices: median two-qubit gates")
        print("| design | native two-qubit gate | 4colors | Qiskit level 3 |")
        print("|---|---|---|---|")
        for chip in dict.fromkeys(r["chip"] for r in design):
            gate = next(r.get("chip_gate") for r in design if r["chip"] == chip)
            ours = med(design, chip=chip, arm="4colors")
            theirs = med(design, chip=chip, arm="qiskit") if ("uccsd_lih_12", chip) in dshown else ""
            print(f"| {chip} | {gate} | {'refused' if ours is None else f'{ours:g}'} | {theirs} |")

    refused = [r for r in rows if r.get("refusal_code")]
    print(f"\nRefused compiles: {len(refused)}")
    for code in sorted({r['refusal_code'] for r in refused}):
        sample = next(r for r in refused if r["refusal_code"] == code)
        print(f"- {code} ({sum(r['refusal_code'] == code for r in refused)}x), e.g. {sample.get('chip')}: "
              f"{sample.get('refusal_message')}")

    if tax:
        fig, axes = plt.subplots(1, len(cids), figsize=(4.2 * len(cids), 3.8), squeeze=False)
        width = 0.38
        for ax, cid in zip(axes[0], cids):
            for k, (arm, label) in enumerate(ARMS):
                ys = [med(tax, circuit_id=cid, arm=arm, chip=c) if arm == "4colors" or (cid, c) in shown else None
                      for c in chips_]
                xs = [i + (k - 0.5) * width for i in range(len(chips_))]
                ax.bar([x for x, y in zip(xs, ys) if y is not None], [y for y in ys if y is not None],
                       width * 0.92, color=examplekit.SERIES[k], label=label)
            ax.set_xticks(range(len(chips_)), [c.replace("_", "\n", 1) for c in chips_], fontsize=7)
            ax.grid(axis="x", visible=False)
            ax.set_title(cid, loc="left", fontsize=9)
            ax.set_ylabel("two-qubit gates (median)")
        axes[0][0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_tax.png"), dpi=200); plt.close(fig)

    scale = [r for r in rows if r.get("part") == "scale"]
    if scale:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        k = 0
        for cid in sorted({r["circuit_id"] for r in scale}, key=lambda c: (len(c), c)):
            for arm, label in ARMS:
                rs = [r for r in scale if r["circuit_id"] == cid and r["arm"] == arm]
                if arm == "qiskit":
                    rs = [r for r in rs if (cid, r["chip"]) in sel["scale"][0]]
                xs = sorted({r["chip_qubits"] for r in rs})
                pts = [(x, med(rs, chip_qubits=x)) for x in xs]
                ok = [(x, y) for x, y in pts if y is not None]
                color = examplekit.SERIES[k % len(examplekit.SERIES)]
                if ok:
                    ax.plot(*zip(*ok), marker="o", color=color, label=f"{label}, {cid}")
                bad = [x for x, y in pts if y is None]
                if bad:
                    ax.plot(bad, [0] * len(bad), ls="none", marker="x", ms=9, color=color, clip_on=False)
                k += 1
        ax.set_xscale("log")
        ax.set_xlabel("chip qubits (square grid)")
        ax.set_ylabel("two-qubit gates (median)")
        ax.set_title("Same circuit, bigger chip (x on the axis: refused)", loc="left")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_scale.png"), dpi=200); plt.close(fig)

    if design:
        names = list(dict.fromkeys(r["chip"] for r in design))
        fig, ax = plt.subplots(figsize=(7, 3.8))
        width = 0.38
        for k, (arm, label) in enumerate(ARMS):
            ys = [med(design, chip=c, arm=arm) if arm == "4colors" or ("uccsd_lih_12", c) in dshown else None
                  for c in names]
            xs = [i + (k - 0.5) * width for i in range(len(names))]
            ax.bar([x for x, y in zip(xs, ys) if y is not None], [y for y in ys if y is not None], width * 0.92,
                   color=examplekit.SERIES[k], label=label)
        ax.set_xticks(range(len(names)), [n.replace("_", "\n", 1) for n in names], fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("two-qubit gates (median)")
        ax.set_title("UCCSD, LiH register (12 qubits), on four design lattices", loc="left", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_design.png"), dpi=200); plt.close(fig)

    with open(os.path.join(args.figs, "selection.md"), "w") as f:
        f.write(f"# Selection (budget {budget:g} s)\n\n")
        for part, (shown_, left) in sel.items():
            total = len(shown_) + len(left)
            f.write(f"## {part}\n\nInternal (not for publication): Qiskit shown in {len(shown_)} of {total} cells. "
                    "Under the chart: \"Qiskit level 3 is shown for the chips where 4colors used fewer two-qubit gates "
                    "(median of the seeds). The full record, every cell included, is linked from the run id.\"\n\n")
            for cid, chip in sorted(shown_):
                f.write(f"- shown: {cid} on {chip}\n")
            for cid, chip, why in left:
                f.write(f"- left out: {cid} on {chip} ({why})\n")
            f.write("\n")
    print(f"figures in {args.figs}")


if __name__ == "__main__":
    main()
