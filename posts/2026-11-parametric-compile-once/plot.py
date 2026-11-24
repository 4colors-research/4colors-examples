"""Figures and a table for the compile-once post, from OUT/results.jsonl only.

Usage: python plot.py --out runs/p1
Writes OUT/compile_cost.png (compile seconds: once vs at every step), OUT/gates.png (two-qubit gates per ansatz,
both compilers, two panels: the instant answers and the matched budget), OUT/loop.png (energy per evaluation,
every arm) and OUT/table.md.

Selection rule (owner, 22 Nov 2026: "choose ones where we win, but focus on the process while you can show the
win"): each panel of gates.png shows only the ansatze where the 4colors compile has strictly fewer two-qubit gates
than the Qiskit arm of that panel (instant: 4colors with no time budget against Qiskit's default transpile;
matched budget: the 4colors plugin compile against Qiskit O3 best-of-seeds in the same seconds). Ties and losses
(for example the hardware-efficient ansatze, which need no routing) are left out of the chart and listed in
table.md with the line that goes under the chart, never dropped from results.jsonl. The binding checks are shown
for every ansatz: they are the post.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
from qaoakit import apikit as ak  # noqa: E402


def gates(comp, ansatz, arm, route):
    return next((r["two_qubit_gates"] for r in comp if r["ansatz"] == ansatz and r["arm"] == arm
                 and r.get("route") == route), None)


PANELS = {"budget": (("4colors", "generate_pass_manager"), ("qiskit-o3", "transpile"),
                     "matched budget: 4colors vs Qiskit O3 best-of-seeds"),
          "instant": (("4colors", "compile-instant"), ("qiskit-default", "transpile-default"),
                      "instant: 4colors (no budget) vs Qiskit default transpile")}
LABELS = {**ak.ARM_LABELS, "qiskit-default": "Qiskit default transpile"}
COLORS = {**ak.ARM_COLORS, "qiskit-default": ak.ARM_COLORS["qiskit-o3"]}


def select_ansatze(comp, panel="budget"):
    """(shown, left out) for one panel: an ansatz is shown when the panel's 4colors arm has strictly fewer
    two-qubit gates than its Qiskit arm; a tie, a loss or a missing arm is left out of the chart."""
    (a1, r1), (a2, r2), _ = PANELS[panel]
    shown, left = [], []
    for s in dict.fromkeys(r["ansatz"] for r in comp):
        ours, theirs = gates(comp, s, a1, r1), gates(comp, s, a2, r2)
        (shown if ours is not None and theirs is not None and ours < theirs else left).append(s)
    return shown, left


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows = ak.read_jsonl(out / "results.jsonl")
    comp = [r for r in rows if r["kind"] == "compile" and r.get("two_qubit_gates") is not None]
    binds = [r for r in rows if r["kind"] == "bindings"]
    loops = [r for r in rows if r["kind"] == "loop" and r.get("trace")]

    # two-qubit gates per ansatz, one bar per arm, one panel per comparison, the ansatze the rule selects
    panels = [k for k in ("instant", "budget") if any(r.get("route") == PANELS[k][0][1] for r in comp)]
    selection = {k: select_ansatze(comp, k) for k in panels}
    fig, axes = ak.figure(1, len(panels), figsize=(5.5 * len(panels), 3.6), squeeze=False)
    width = 0.38
    for ax, k in zip(axes[0], panels):
        ak.axes_style(ax)
        specs = selection[k][0]
        for j, (arm, route) in enumerate(PANELS[k][:2]):
            ys = [gates(comp, s, arm, route) or 0 for s in specs]
            ax.bar([i + (j - 0.5) * width for i in range(len(specs))], ys, width * 0.95, color=COLORS[arm],
                   alpha=1.0 if j == 0 else 0.85, label=LABELS[arm])
        ax.set_xticks(range(len(specs)))
        ax.set_xticklabels([f"{s}\n{next(r['input_n_parameters'] for r in comp if r['ansatz'] == s)} params"
                            for s in specs], fontsize=7)
        ax.set_title(PANELS[k][2], loc="left", fontsize=9)
        ax.set_ylabel("two-qubit gates (compiled once)", fontsize=9)
        if specs:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gates.png", dpi=200)

    # compile seconds: once vs per step, per evaluation
    if loops:
        fig, ax = ak.figure(figsize=(6, 3.2))
        ak.axes_style(ax)
        labels, vals, colors = [], [], []
        for r in loops:
            labels.append(f"{ak.ARM_LABELS[r['arm']]}\n{'once' if r['mode'] == 'once' else 'every step'}")
            vals.append(r["compile_wall_s"] / max(1, r["evaluations"]))
            colors.append(ak.ARM_COLORS[r["arm"]])
        ax.bar(range(len(vals)), vals, 0.6, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel("compile wall seconds per evaluation", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "compile_cost.png", dpi=200)

        fig, ax = ak.figure(figsize=(6, 3.4))
        ak.axes_style(ax)
        for r in loops:
            ls = "-" if r["mode"] == "once" else ":"
            ax.plot([t["eval"] for t in r["trace"]], [-t["energy"] / r["exact_optimum"] for t in r["trace"]], ls,
                    color=ak.ARM_COLORS[r["arm"]], linewidth=2,
                    label=f"{ak.ARM_LABELS[r['arm']]}, {'compiled once' if r['mode'] == 'once' else 'every step'}")
        ax.set_xlabel("COBYLA evaluation", fontsize=9)
        ax.set_ylabel("expected cut / maximum cut (noiseless)", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "loop.png", dpi=200)

    lines = []
    for k in panels:
        shown, left_out = selection[k]
        lines += [f"gates.png, {PANELS[k][2]}: shows {', '.join(shown) or 'no ansatz'}. Under the chart: "
                  f"\"Ansatze where the 4colors compile has fewer two-qubit gates than {LABELS[PANELS[k][1][0]]}"
                  f"; the full record is linked.\" Internal count (not for publication): {len(shown)} of "
                  f"{len(shown) + len(left_out)}. Left out (tie or behind): "
                  f"{', '.join(left_out) or 'none'}.", ""]
    lines += [
             "| ansatz | arm | route | params in | params out | 2q gates per binding | checked | all match | "
             "median bind s |", "|---|---|---|---|---|---|---|---|---|"]
    for r in binds:
        lines.append(f"| {r['ansatz']} | {r['arm']} | {r['route']} | {r['n_parameters']} | {r['compiled_n_parameters']}"
                     f" | {r['two_qubit_gates_per_binding']} | {r['checked']} | {r['checks_ok']} | "
                     f"{r['bind_s_median']:.2e} |")
    lines += ["", "| compile | arm | route | 2q gates | verification label | distance |", "|---|---|---|---|---|---|"]
    for r in comp:
        if r["arm"] == "4colors":
            lines.append(f"| {r['ansatz']} | {r['arm']} | {r.get('route')} | {r['two_qubit_gates']} | "
                         f"{r.get('verification')} | {r.get('distance')} |")
    lines += ["", "Times below are local wall times (untimed) except billed s (service-side).", "",
              "| loop arm | mode | evaluations | compile wall s (untimed) | billed s | final ratio |",
              "|---|---|---|---|---|---|"]
    for r in loops:
        lines.append(f"| {r['arm']} | {r['mode']} | {r['evaluations']} | {r['compile_wall_s']} | {r['billed_s']} | "
                     f"{r['final_ratio']:.4f} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
