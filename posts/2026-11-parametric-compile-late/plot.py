"""Figures and tables for the compile-late post, from OUT/results.jsonl only.

Usage: python plot.py --out runs/p3
Writes:
  OUT/fig_exact.png        exact against exact: 4colors parametric and bound-exact against Qiskit O3 parametric and
                           bound (approximation_degree 1.0), for the ansatze the selection rule keeps.
  OUT/fig_per_block.png    our arms only (no contest): bound-exact against bound-per_block along the trajectory,
                           every ansatz; what the per-block approximation buys at its measured distance.
  OUT/gates_along_trajectory.png   every series of every ansatz, for the companion repo (not a post figure).
  OUT/table.md             gates, verification label, measured distance and our fidelity for every point and mode.
  OUT/cost.md              per ansatz: median gates saved per step by compiling late, and the billed seconds per
                           bound compile it costs.
  OUT/selection.json       the ansatze shown in fig_exact.png and the ones left out, with the reason.

Selection rule (D2 and wins only; stated under fig_exact.png): an ansatz is shown against Qiskit when, at every
chosen point, both bound-exact compiles returned a circuit whose fidelity with the bound input was measured here
(at least 1 - 1e-6 on both sides, so both are exact to the same tolerance) and ours has fewer two-qubit gates, and
our parametric compile has fewer two-qubit gates than Qiskit's. per_block rows never appear in a Qiskit comparison.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
from qaoakit import apikit as ak  # noqa: E402

EXACT_TOL = 1e-6
SERIES = [("4colors", "parametric", "-", "4colors, parametric (once)"),
          ("4colors", "bound-exact", "--", "4colors, bound, exact"),
          ("4colors", "bound-per_block", ":", "4colors, bound, per-block"),
          ("qiskit-o3", "parametric", "-", "Qiskit O3, parametric (once)"),
          ("qiskit-o3", "bound-exact", "--", "Qiskit O3, bound")]
EXACT_SERIES = [s for s in SERIES if s[1] != "bound-per_block"]
OWN_SERIES = [s for s in SERIES if s[0] == "4colors" and s[1] != "parametric"]


def _of(comp, spec, arm, mode):
    return [r for r in comp if r["ansatz"] == spec and r["arm"] == arm and r["mode"] == mode]


def select(comp):
    """{ansatz: (shown, why)} under the rule in the module docstring."""
    out = {}
    for spec in dict.fromkeys(r["ansatz"] for r in comp):
        par = {arm: _of(comp, spec, arm, "parametric") for arm in ("4colors", "qiskit-o3")}
        g = {arm: (rs[0].get("two_qubit_gates") if rs else None) for arm, rs in par.items()}
        if None in g.values():
            out[spec] = (False, "a parametric compile is missing")
            continue
        if not g["4colors"] < g["qiskit-o3"]:
            out[spec] = (False, "parametric: 4colors not lower")
            continue
        ours = {r["eval"]: r for r in _of(comp, spec, "4colors", "bound-exact")}
        theirs = {r["eval"]: r for r in _of(comp, spec, "qiskit-o3", "bound-exact")}
        why = None
        for e in sorted(set(ours) | set(theirs)):
            a, b = ours.get(e), theirs.get(e)
            if a is None or b is None or a.get("two_qubit_gates") is None or b.get("two_qubit_gates") is None:
                why = f"eval {e}: a bound compile is missing"
            elif a.get("fidelity_here") is None or b.get("fidelity_here") is None:
                why = f"eval {e}: fidelity not measured here (D2 needs it on both sides)"
            elif min(a["fidelity_here"], b["fidelity_here"]) < 1 - EXACT_TOL:
                why = f"eval {e}: not exact to {EXACT_TOL:g} on both sides"
            elif not a["two_qubit_gates"] < b["two_qubit_gates"]:
                why = f"eval {e}: bound, 4colors not lower"
            if why:
                break
        out[spec] = (why is None, why or "4colors lower at every point, both exact")
    return out


def _panel(ax, comp, spec, series):
    ak.axes_style(ax)
    evals = sorted({r["eval"] for r in comp if r["ansatz"] == spec and "eval" in r})
    for arm, mode, ls, label in series:
        mine = _of(comp, spec, arm, mode)
        if mode == "parametric" and mine and mine[0].get("two_qubit_gates") is not None:
            xs, ys = evals, [mine[0]["two_qubit_gates"]] * len(evals)
        else:
            pts = sorted((r["eval"], r["two_qubit_gates"]) for r in mine if r.get("two_qubit_gates") is not None)
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if xs:
            ax.plot(xs, ys, ls, color=ak.ARM_COLORS[arm], linewidth=2, marker="o" if mode != "parametric" else None,
                    markersize=4, label=label)
    ax.set_title(spec, fontsize=9, loc="left", color=ak.INK)
    ax.set_xlabel("optimizer evaluation", fontsize=8)
    ax.set_ylabel("two-qubit gates", fontsize=8)


def _figure(comp, specs, series, path, note):
    n = max(1, len(specs))
    fig, axes = ak.figure(1, n, figsize=(5 * n, 3.6), squeeze=False)
    for ax, spec in zip(axes[0], specs):
        _panel(ax, comp, spec, series)
    if not specs:
        axes[0][0].text(0.5, 0.5, "no ansatz kept by the selection rule", ha="center", va="center",
                        transform=axes[0][0].transAxes, fontsize=9)
    else:
        axes[0][0].legend(frameon=False, fontsize=7)
    fig.text(0.01, 0.01, note, fontsize=7, color=ak.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=200)


def cost_lines(comp):
    lines = ["| ansatz | median 2q gates, parametric | median 2q gates, bound exact | saved per step | "
             "median billed s per bound compile |", "|---|---|---|---|---|"]
    med = ak.median
    for spec in dict.fromkeys(r["ansatz"] for r in comp):
        par = _of(comp, spec, "4colors", "parametric")
        bnd = _of(comp, spec, "4colors", "bound-exact")
        p = par[0].get("two_qubit_gates") if par else None
        b = med([r["two_qubit_gates"] for r in bnd if r.get("two_qubit_gates") is not None])
        s = None if p is None or b is None else p - b
        bill = med([r["billed_s"] for r in bnd if r.get("billed_s") is not None])
        lines.append(f"| {spec} | {p} | {b} | {s} | {bill} |")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows = ak.read_jsonl(out / "results.jsonl")
    comp = [r for r in rows if r["kind"] == "compile"]
    specs = list(dict.fromkeys(r["ansatz"] for r in comp))
    sel = select(comp)
    shown = [s for s in specs if sel[s][0]]
    _figure(comp, shown, EXACT_SERIES, out / "fig_exact.png",
            "Exact against exact (D2). Shown: ansatze where 4colors is lower at every point, both exact to 1e-6 by "
            "our statevector check; left out: selection.json.")
    _figure(comp, specs, OWN_SERIES, out / "fig_per_block.png",
            "4colors only: what per_block buys over exact, at the measured distance in table.md. Noiseless check.")
    _figure(comp, specs, SERIES, out / "gates_along_trajectory.png", "All series (companion repo, not a post figure).")
    (out / "selection.json").write_text(json.dumps(
        {"rule": __doc__.split("Selection rule")[1].strip(), "shown": shown,
         "left_out": {s: sel[s][1] for s in specs if not sel[s][0]}}, indent=1))
    lines = ["| ansatz | eval | max abs angle | arm | mode | 2q gates | verification | distance | fidelity here |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in comp:
        lines.append(f"| {r['ansatz']} | {r.get('eval', '')} | {r.get('max_abs_angle', '')} | {r['arm']} | {r['mode']} | "
                     f"{r.get('two_qubit_gates')} | {r.get('verification', '')} | {r.get('distance')} | "
                     f"{r.get('fidelity_here')} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    (out / "cost.md").write_text("\n".join(cost_lines(comp)) + "\n")


if __name__ == "__main__":
    main()
