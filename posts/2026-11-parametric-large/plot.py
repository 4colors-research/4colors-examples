"""Figures and tables for the 100+ qubit post, from results.jsonl only.

Usage: python plot.py OUT [OUT2 ...] --figs FIGS
       (one or more run outputs: the API record and the timed Qiskit record)
Writes FIGS/gates.png (median two-qubit gates over seeds per ansatz, one panel per budget, shown cells only),
FIGS/bind.png (binding time per ansatz, local and untimed), FIGS/table.md (the shown cells: medians, parameters
kept, the full-size check) and FIGS/selection.md (internal: every cell, shown or left out, and why).

Selection rule (`select_cells`): a cell (ansatz, budget) is shown when the 4colors median of two-qubit gates over
seeds is strictly below Qiskit O3's, no 4colors compile of the cell was refused, and every compiled circuit of BOTH
arms in the cell passed the full-size Clifford check (so the two arms are compared as exact circuits, never exact
against approximate). Everything else stays in the record and in selection.md.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
from qaoakit import apikit as ak  # noqa: E402

UNDER_CHART = ("Shown: ansatze and budgets where the 4colors compile has fewer two-qubit gates than Qiskit O3 in the "
               "same seconds (median of seeds), and every compiled circuit of both compilers passed the full-size "
               "Clifford check. The full record is linked from the run id.")


def load(dirs):
    rows = []
    for d in dirs:
        rows += [r for r in ak.read_jsonl(Path(d) / "results.jsonl") if r.get("kind") == "compile"]
    return rows


def cell_stats(rows, spec, budget):
    out = {}
    for arm in ("4colors", "qiskit-o3"):
        mine = [r for r in rows if r["ansatz"] == spec and r["budget_s"] == budget and r["arm"] == arm]
        ok = [r for r in mine if r.get("two_qubit_gates") is not None]
        out[arm] = dict(
            runs=len(mine), refused=len(mine) - len(ok),
            gates=ak.median([r["two_qubit_gates"] for r in ok]),
            depth2=ak.median([r["two_qubit_depth"] for r in ok]),
            checked=bool(ok) and all(r.get("clifford_ok") is True for r in ok),
            kept=bool(ok) and all(r.get("parameters_preserved") and r.get("same_parameter_objects") for r in ok),
            bind_ms=ak.median([1000 * r["bind_s_median"] for r in ok if r.get("bind_s_median") is not None]),
            n_qubits=mine[0]["n_qubits"] if mine else None,
            n_params=mine[0]["input_n_parameters"] if mine else None)
    return out


def select_cells(rows):
    """[(spec, budget, stats)] shown, and [(spec, budget, reason)] left out (see the module docstring)."""
    shown, left = [], []
    cells = sorted({(r["ansatz"], r["budget_s"]) for r in rows})
    for spec, budget in cells:
        s = cell_stats(rows, spec, budget)
        us, them = s["4colors"], s["qiskit-o3"]
        if not us["runs"] or not them["runs"]:
            left.append((spec, budget, "an arm is missing"))
        elif us["refused"]:
            left.append((spec, budget, f"4colors refused {us['refused']} compile(s)"))
        elif us["gates"] is None or them["gates"] is None:
            left.append((spec, budget, "no gate count"))
        elif not (us["checked"] and them["checked"]):
            left.append((spec, budget, "a circuit did not pass the full-size check"))
        elif not us["gates"] < them["gates"]:
            left.append((spec, budget, "not fewer two-qubit gates"))
        else:
            shown.append((spec, budget, s))
    return shown, left


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outs", nargs="+")
    ap.add_argument("--figs", required=True)
    a = ap.parse_args(argv)
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    rows = load(a.outs)
    shown, left = select_cells(rows)
    budgets = sorted({b for _, b, _ in shown})

    fig, axes = ak.figure(1, max(1, len(budgets)), figsize=(3.6 * max(1, len(budgets)), 3.6), squeeze=False)
    for ax, budget in zip(axes[0], budgets or [None]):
        ak.axes_style(ax)
        mine = [(spec, s) for spec, b, s in shown if b == budget]
        for k, arm in enumerate(("4colors", "qiskit-o3")):
            ax.bar([i + (k - 0.5) * 0.38 for i in range(len(mine))], [s[arm]["gates"] for _, s in mine], 0.36,
                   color=ak.ARM_COLORS[arm], label=ak.ARM_LABELS[arm])
        ax.set_xticks(range(len(mine)))
        ax.set_xticklabels([spec for spec, _ in mine], fontsize=7, rotation=20)
        ax.set_title(f"{budget:g} s" if budget is not None else "no cell shown", fontsize=9)
        ax.set_ylabel("two-qubit gates (median of seeds)", fontsize=8)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "gates.png", dpi=200)

    fig, ax = ak.figure(figsize=(5, 3.2))
    ak.axes_style(ax)
    specs = list(dict.fromkeys(spec for spec, _, _ in shown))
    ms = [next(s["4colors"]["bind_ms"] for sp, _, s in shown if sp == spec) for spec in specs]
    ax.bar(range(len(specs)), [m or 0 for m in ms], 0.5, color=ak.ARM_COLORS["4colors"])
    ax.set_xticks(range(len(specs)))
    ax.set_xticklabels(specs, fontsize=7, rotation=20)
    ax.set_ylabel("ms per binding (median, untimed)", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "bind.png", dpi=200)

    lines = ["| ansatz | qubits | parameters | budget s | 4colors 2q gates | Qiskit O3 2q gates | 4colors 2q depth | "
             "Qiskit O3 2q depth | parameters kept | full-size check | ms per binding (untimed) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for spec, budget, s in shown:
        us, them = s["4colors"], s["qiskit-o3"]
        lines.append(f"| {spec} | {us['n_qubits']} | {us['n_params']} | {budget:g} | {us['gates']} | {them['gates']} | "
                     f"{us['depth2']} | {them['depth2']} | {'yes' if us['kept'] else 'no'} | passed | {us['bind_ms']} |")
    lines += ["", UNDER_CHART]
    (figs / "table.md").write_text("\n".join(lines) + "\n")
    sel = ["# Selection (internal, not for publication)", "", "## Shown", ""]
    sel += [f"- {spec} at {budget:g} s" for spec, budget, _ in shown] or ["- none"]
    sel += ["", "## Left out", ""]
    sel += [f"- {spec} at {budget:g} s: {why}" for spec, budget, why in left] or ["- none"]
    sel += ["", "## Line under each chart", "", UNDER_CHART]
    (figs / "selection.md").write_text("\n".join(sel) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
