"""Figures for the give-it-seconds post, from results.jsonl only.

    python plot.py results/seconds --figs figs

fig_budget.png  (hero) two-qubit gates against the time budget, one panel per circuit, 4colors and Qiskit O3 on the
                published calibration (median over seeds); only the cells the selection rule shows
fig_margin.png  how much fewer two-qubit gates ours used than Qiskit at each budget (percent, median over seeds),
                one line per circuit size: the margin against the seconds given
fig_stream.png  one compile's incumbents as they arrived, beside Qiskit's best so far as seeds are added
selection.md    the cells shown and left out, and the line that goes under fig_budget and fig_margin

Selection rule: a cell (circuit, budget) is shown when ours has strictly fewer two-qubit gates than Qiskit O3 on
FakeFez as published at that budget (medians over seeds), and no Qiskit seed of the cell overran the budget. An
over-budget cell (one level-3 transpile took longer than T) is reported apart, never as a win or a loss. Ties,
losses and refusals are left out of the charts and kept in the record.
"""
import argparse
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck

OURS = ("4colors", ck.HEAVY_HEX_DEVICE, "4colors")
QISKIT = ("qiskit", "full", "Qiskit O3, published calibration")


def panels(n):
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    return rows, cols


def medians(rows, arm, cset, cid):
    """{budget: median two-qubit gates} over seeds, ignoring refusals."""
    rs = [r for r in ck.results(rows, circuit_id=cid, arm=arm, coupler_set=cset) if r.get("two_qubit_gates") is not None]
    return {b: examplekit.median([r["two_qubit_gates"] for r in rs if r["budget_s"] == b])
            for b in sorted({r["budget_s"] for r in rs})}


def select_cells(rows):
    """({circuit_id: {budget: (ours, qiskit)}} shown, [(circuit_id, budget, why)] left out)."""
    shown, left = {}, []
    for cid in sorted({r["circuit_id"] for r in ck.results(rows)}, key=lambda c: (len(c), c)):
        ours, theirs = medians(rows, *OURS[:2], cid), medians(rows, *QISKIT[:2], cid)
        for b in sorted(set(ours) | set(theirs)):
            over = any(r.get("over_budget") for r in ck.results(rows, circuit_id=cid, arm="qiskit", coupler_set="full",
                                                                 budget_s=b))
            if b not in ours or b not in theirs:
                left.append((cid, b, "an arm has no answer (a refusal is a row in the record)"))
            elif over:
                left.append((cid, b, "a Qiskit seed overran the budget: reported apart"))
            elif not ours[b] < theirs[b]:
                left.append((cid, b, "tie or behind"))
            else:
                shown.setdefault(cid, {})[b] = (ours[b], theirs[b])
    return shown, left


def margin_pct(ours, theirs):
    """Percent fewer two-qubit gates than Qiskit: 100 * (theirs - ours) / theirs."""
    return 100.0 * (theirs - ours) / theirs


def selection_md(shown, left):
    n_shown = sum(len(v) for v in shown.values())
    total = n_shown + len(left)
    lines = ["# Cells shown in fig_budget.png and fig_margin.png", "",
             "Under the charts: \"Circuits and budgets where the 4colors compile used fewer two-qubit gates than Qiskit "
             "O3 given the same seconds on the published calibration (median of 5 seeds). The full record, every "
             "cell included, is linked from the run id.\"", "",
             f"Internal (not for publication): cells shown {n_shown} of {total}.", "",
             "| circuit | budget (s) | 4colors | Qiskit O3 | fewer (%) |", "|---|---|---|---|---|"]
    for cid, cells in shown.items():
        for b, (o, q) in sorted(cells.items()):
            lines.append(f"| {cid} | {b:g} | {o} | {q} | {margin_pct(o, q):.1f} |")
    lines += ["", "Left out (kept in the record):", ""]
    lines += [f"- {c}, {b:g} s: {why}" for c, b, why in left] or ["- none"]
    return "\n".join(lines) + "\n"


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
    shown, left = select_cells(rows)
    with open(os.path.join(args.figs, "selection.md"), "w") as f:
        f.write(selection_md(shown, left))
    cids = list(shown)
    nr, nc = panels(max(1, len(cids)))

    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr), squeeze=False)
    for ax, cid in zip(axes.flat, cids):
        bs = sorted(shown[cid])
        for k, label in ((0, OURS[2]), (1, QISKIT[2])):
            ax.plot(bs, [shown[cid][b][k] for b in bs], marker="o", color=examplekit.SERIES[k], label=label)
        ax.set_xscale("log")
        ax.set_title(cid, loc="left", fontsize=9)
        ax.set_xlabel("time budget (s)")
        ax.set_ylabel("two-qubit gates")
    for ax in list(axes.flat)[len(cids):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_budget.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, cid in enumerate(cids):
        bs = sorted(shown[cid])
        ax.plot(bs, [margin_pct(*shown[cid][b]) for b in bs], marker=examplekit.MARKERS[k % len(examplekit.MARKERS)],
                color=examplekit.SERIES[k % len(examplekit.SERIES)], label=cid)
    ax.set_xscale("log")
    ax.set_xlabel("time budget (s), the same for both compilers")
    ax.set_ylabel("fewer two-qubit gates than Qiskit O3 (%)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_margin.png"), dpi=200); plt.close(fig)

    inc = [r for r in rows if r.get("kind") == "incumbent"]
    trials = [r for r in rows if r.get("kind") == "trial" and r.get("coupler_set") == "full"]
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr), squeeze=False)
    for ax, cid in zip(axes.flat, cids):
        mine = sorted((r for r in inc if r["circuit_id"] == cid and r.get("two_qubit_gates") is not None),
                      key=lambda r: r["index"])
        if mine:
            ax.step([r["t_client_s"] for r in mine], [r["two_qubit_gates"] for r in mine], where="post",
                    marker="o", color=examplekit.SERIES[0], label="4colors incumbents")
        seed = mine[0]["seed"] if mine else min((r["seed"] for r in trials if r["circuit_id"] == cid), default=None)
        q = sorted((r for r in trials if r["circuit_id"] == cid and r["seed"] == seed), key=lambda r: r["cumulative_s"])
        best, xs, ys = None, [], []
        for r in q:
            best = r["two_qubit_gates"] if best is None else min(best, r["two_qubit_gates"])
            xs.append(r["cumulative_s"]); ys.append(best)
        if xs:
            ax.step(xs, ys, where="post", marker="o", color=examplekit.SERIES[1], label="Qiskit O3 best so far")
        ax.set_title(cid, loc="left", fontsize=9)
        ax.set_xlabel("seconds since start")
        ax.set_ylabel("two-qubit gates")
    for ax in list(axes.flat)[len(cids):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_stream.png"), dpi=200); plt.close(fig)
    print(f"cells shown: {sum(len(v) for v in shown.values())} of {sum(len(v) for v in shown.values()) + len(left)}; "
          f"figures and selection.md in {args.figs}")


if __name__ == "__main__":
    main()
