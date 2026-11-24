"""Table and figures for the biggest-circuits post, from results.jsonl only.

    python plot.py results/big --figs figs

Prints one line per circuit and budget (4colors and Qiskit counts, refusal codes) and writes:

  fig_size.png       our arm only: two-qubit gates out against two-qubit gates in, one line per budget;
                     refused compiles as x marks on the bottom axis, skipped ones (not sent) left out.
  fig_vs_qiskit.png  4colors against Qiskit level 3 on FakeFez as published (`full`), at the same budget, for
                     the (circuit, budget) cells the selection rule keeps (below).
  selection.json     the cells shown, and every cell left out with its reason.

Selection rule (stated under fig_vs_qiskit.png): a cell is shown when both arms returned a circuit, Qiskit's best
was found within the budget (its first transpile did not overrun: `over_budget` cells are reported apart, never
as a Qiskit loss), and the 4colors median two-qubit count is lower than Qiskit's `full` median. Qiskit's count is
annotated with its gates on dead couplers (its map lists them; ours is told by the calibration).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck


def cells(rows):
    """Per (circuit, budget): medians of both arms and the selection verdict."""
    res = ck.results(rows)
    out = []
    for cid in sorted({r["circuit_id"] for r in res}, key=lambda c: min(r["input_2q"] for r in res
                                                                         if r["circuit_id"] == c)):
        mine = [r for r in res if r["circuit_id"] == cid]
        for b in sorted({r["budget_s"] for r in mine}):
            four = [r for r in mine if r["arm"] == "4colors" and r["budget_s"] == b]
            q = ck.results(mine, arm="qiskit", coupler_set="full", budget_s=b)
            f_med = examplekit.median([r.get("two_qubit_gates") for r in four if r.get("two_qubit_gates") is not None])
            q_med = examplekit.median([r.get("two_qubit_gates") for r in q if r.get("two_qubit_gates") is not None])
            cell = {"circuit_id": cid, "input_2q": mine[0]["input_2q"], "budget_s": b, "fourcolors_2q": f_med,
                    "qiskit_full_2q": q_med,
                    "qiskit_full_on_dead": examplekit.median([r.get("gates_on_failed_couplers") for r in q
                                                           if r.get("gates_on_failed_couplers") is not None]),
                    "qiskit_over_budget": any(r.get("over_budget") for r in q),
                    "refusals": sorted({r["refusal_code"] for r in four if r.get("refusal_code")})}
            if f_med is None:
                cell["shown"], cell["why"] = False, "no 4colors circuit (refused or not sent)"
            elif q_med is None:
                cell["shown"], cell["why"] = False, "no Qiskit row at this budget"
            elif cell["qiskit_over_budget"]:
                cell["shown"], cell["why"] = False, "Qiskit's first transpile overran the budget: reported apart"
            elif f_med < q_med:
                cell["shown"], cell["why"] = True, "4colors lower"
            else:
                cell["shown"], cell["why"] = False, "4colors not lower (tie or behind): named in the text"
            out.append(cell)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--figs", default="figs")
    args = p.parse_args(argv)
    rows = examplekit.load_rows(args.inputs)
    os.makedirs(args.figs, exist_ok=True)
    plt = examplekit.plot_style()
    res = ck.results(rows)
    skips = [r for r in rows if r.get("kind") == "skip"]
    for lim in (r for r in rows if r.get("kind") == "limits"):
        print(f"limits (GET /v1/license): {json.dumps(lim.get('limits'))} plan_max_budget={lim.get('plan_max_budget')}")

    print("| circuit | input 2q | budget s | 4colors 2q | refusal | Qiskit full 2q (on dead couplers) |")
    print("|---|---|---|---|---|---|")
    table = cells(rows)
    for c in table:
        dead = "" if c["qiskit_full_on_dead"] is None else f" ({c['qiskit_full_on_dead']})"
        over = " over budget" if c["qiskit_over_budget"] else ""
        print(f"| {c['circuit_id']} | {c['input_2q']} | {c['budget_s']:g} | {c['fourcolors_2q']} | "
              f"{' '.join(c['refusals'])} | {c['qiskit_full_2q']}{dead}{over} |")
    for r in skips:
        print(f"not sent: {r['circuit_id']} {r['budget_s']:g} s ({r.get('note')})")
    refusals = [r for r in res if r.get("refusal_code")]
    for code in sorted({r["refusal_code"] for r in refusals}):
        r = next(x for x in refusals if x["refusal_code"] == code)
        print(f"refusal {code}: {r.get('refusal_message')} field={r.get('refusal_field')} limit={r.get('limit')}")

    # Figure 1: our arm only (no contest): how the output grows with the input, and where the refusals start.
    fig, ax = plt.subplots(figsize=(7, 4.6))
    four = [r for r in res if r["arm"] == "4colors"]
    budgets = sorted({r["budget_s"] for r in four})
    for k, b in enumerate(budgets):
        rs = [r for r in four if r["budget_s"] == b]
        xs = sorted({r["input_2q"] for r in rs})
        pts = [(x, examplekit.median([r["two_qubit_gates"] for r in rs if r["input_2q"] == x
                                   and r.get("two_qubit_gates") is not None])) for x in xs]
        ok = [(x, y) for x, y in pts if y is not None]
        if ok:
            ax.plot(*zip(*ok), marker="o", color=examplekit.SERIES[k % len(examplekit.SERIES)], label=f"4colors, {b:g} s")
        bad = [x for x, y in pts if y is None]
        if bad:
            ax.plot(bad, [ax.get_ylim()[0] if ok else 1] * len(bad), ls="none", marker="x", ms=9,
                    color=examplekit.SERIES[k % len(examplekit.SERIES)], clip_on=False)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("two-qubit gates in (rzz)")
    ax.set_ylabel("two-qubit gates out (cz)")
    ax.set_title("QAOA on 150 qubits, FakeFez heavy-hex, by time budget (x: refused)", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_size.png"), dpi=200); plt.close(fig)

    # Figure 2: the cells the selection rule keeps.
    shown = [c for c in table if c["shown"]]
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.45 * max(1, len(shown))))
    labels = [f"{c['circuit_id']}, {c['budget_s']:g} s" for c in shown]
    ys = list(range(len(shown)))
    ax.barh([y + 0.2 for y in ys], [c["qiskit_full_2q"] for c in shown], height=0.4, color=examplekit.INK_2,
            label="Qiskit level 3, FakeFez as published")
    ax.barh([y - 0.2 for y in ys], [c["fourcolors_2q"] for c in shown], height=0.4, color=examplekit.SERIES[0],
            label="4colors")
    for y, c in zip(ys, shown):
        if c["qiskit_full_on_dead"]:
            ax.text(c["qiskit_full_2q"], y + 0.2, f"  {c['qiskit_full_on_dead']:g} on dead couplers", va="center",
                    fontsize=7, color=examplekit.INK_2)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("two-qubit gates out (median over seeds)")
    ax.set_title("Same budget, same device; cells where 4colors is lower (rule in selection.json)", loc="left",
                 fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_vs_qiskit.png"), dpi=200); plt.close(fig)

    with open(os.path.join(args.figs, "selection.json"), "w") as fh:
        json.dump({"rule": "both arms returned a circuit; Qiskit's best found within the budget (over_budget "
                           "reported apart); 4colors median two-qubit gates lower than Qiskit full",
                   "shown": shown, "left_out": [c for c in table if not c["shown"]]}, fh, indent=1)
    print(f"figures in {args.figs}: {len(shown)} cells shown, {len(table) - len(shown)} left out")


if __name__ == "__main__":
    main()
