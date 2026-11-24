"""Encodings tables and charts, from OUT/results.jsonl only.

Usage: python plot.py --out runs/enc [--figs FIGS]   (FIGS defaults to OUT)
Writes
  FIGS/encodings.png  the three encodings side by side, compiled by 4colors: two-qubit gates (small graphs and the
                      larger count-only graphs) and the noisy ratio against each encoding's own start-state ratio
  FIGS/compiler.png   4colors against Qiskit O3, shown cells only (see `select_cells`)
  FIGS/table.md       the encodings table, the shown compiler cells and the line under that chart, the larger
                      graphs' gate counts, and the optimizer's rows
  FIGS/selection.md   internal: every compiler cell, shown or left out, and why

Compiler selection rule (`select_cells`): a cell (encoding, p) is shown when, over the small graphs and seeds, the
4colors median of compiled two-qubit gates is strictly below Qiskit O3's, the 4colors median noisy ratio is not
below Qiskit O3's, no 4colors compile was refused, and every simulated row of both arms passed the noiseless check
(`check_ok`: the compiled circuit gives the tuned circuit's expectation). The encodings table compares encodings
with one compiler (ours) and is shown whole: it is a comparison of models, not of compilers.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
from qaoakit import apikit as ak  # noqa: E402

UNDER_CHART = ("Shown: encodings and depths where the 4colors compile has fewer two-qubit gates than Qiskit O3 in "
               "the same seconds and a noisy ratio at least as high (medians over graphs and seeds), and every "
               "compiled circuit of both compilers passed the noiseless check. The full record is linked from the "
               "run id.")


def fmt(x, nd=4):
    return "" if x is None else f"{x:.{nd}f}"


def arm_stats(rows):
    ok = [r for r in rows if r.get("two_qubit_gates") is not None]
    sim = [r for r in ok if r.get("simulated")]
    return dict(runs=len(rows), refused=len(rows) - len(ok), gates=ak.median([r["two_qubit_gates"] for r in ok]),
                noisy=ak.median([r.get("noisy_ratio_mean") for r in sim]),
                feasible=ak.median([r.get("noisy_feasible") for r in sim]),
                ideal=ak.median([r.get("ideal_ratio") for r in rows]),
                start=ak.median([r.get("start_ratio") for r in rows]),
                qubits=ak.median([r.get("n_qubits") for r in rows]),
                checked=bool(sim) and all(r.get("check_ok") is True for r in sim))


def select_cells(qaoa_rows):
    """[(encoding, p, {arm: stats})] shown, and [(encoding, p, reason)] left out (see the module docstring)."""
    shown, left = [], []
    for enc, p in dict.fromkeys((r["encoding"], r["p"]) for r in qaoa_rows):
        s = {arm: arm_stats([r for r in qaoa_rows if r["encoding"] == enc and r["p"] == p and r["arm"] == arm])
             for arm in ("4colors", "qiskit-o3")}
        us, them = s["4colors"], s["qiskit-o3"]
        if not us["runs"] or not them["runs"]:
            left.append((enc, p, "an arm is missing"))
        elif us["refused"]:
            left.append((enc, p, f"4colors refused {us['refused']} compile(s)"))
        elif not (us["checked"] and them["checked"]):
            left.append((enc, p, "a circuit was not checked or did not pass the noiseless check"))
        elif us["gates"] is None or them["gates"] is None or not us["gates"] < them["gates"]:
            left.append((enc, p, "not fewer two-qubit gates"))
        elif us["noisy"] is None or them["noisy"] is None or us["noisy"] < them["noisy"]:
            left.append((enc, p, "lower noisy ratio"))
        else:
            shown.append((enc, p, s))
    return shown, left


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--figs")
    a = ap.parse_args(argv)
    out = Path(a.out)
    figs = Path(a.figs) if a.figs else out
    figs.mkdir(parents=True, exist_ok=True)
    rows = ak.read_jsonl(out / "results.jsonl")
    q = [r for r in rows if r["kind"] == "qaoa"]
    cnt = [r for r in rows if r["kind"] == "count"]
    cells = list(dict.fromkeys((r["encoding"], r["p"]) for r in q))
    ours = {c: arm_stats([r for r in q if (r["encoding"], r["p"]) == c and r["arm"] == "4colors"]) for c in cells}

    lines = ["## The three encodings (compiled by 4colors; medians over graphs and seeds; simulated, noise model as "
             "stated)", "",
             "| encoding | p | qubits | 2q gates | start-state ratio | ideal ratio | noisy ratio | P(feasible) |",
             "|---|---|---|---|---|---|---|---|"]
    for (enc, p), s in ours.items():
        lines.append(f"| {enc} | {p} | {s['qubits']} | {s['gates']} | {fmt(s['start'])} | {fmt(s['ideal'])} | "
                     f"{fmt(s['noisy'])} | {fmt(s['feasible'])} |")

    shown, left = select_cells(q)
    lines += ["", "## Compiled by 4colors and by Qiskit O3 (shown cells only)", "",
              "| encoding | p | 4colors 2q gates | Qiskit O3 2q gates | 4colors noisy ratio | Qiskit O3 noisy ratio |",
              "|---|---|---|---|---|---|"]
    for enc, p, s in shown:
        lines.append(f"| {enc} | {p} | {s['4colors']['gates']} | {s['qiskit-o3']['gates']} | "
                     f"{fmt(s['4colors']['noisy'])} | {fmt(s['qiskit-o3']['noisy'])} |")
    lines += ["", UNDER_CHART]

    big = list(dict.fromkeys((r["instance"], r["encoding"], r["p"]) for r in cnt))
    lines += ["", "## Larger graphs, gate counts only (compiled by 4colors; median over seeds)", "",
              "| graph | nodes | edges | encoding | p | qubits | input 2q gates | compiled 2q gates | refused |",
              "|---|---|---|---|---|---|---|---|---|"]
    for inst, enc, p in big:
        m = [r for r in cnt if (r["instance"], r["encoding"], r["p"]) == (inst, enc, p)]
        ok = [r for r in m if r.get("two_qubit_gates") is not None]
        lines.append(f"| {inst} | {m[0]['n']} | {m[0]['n_edges']} | {enc} | {p} | {m[0]['n_qubits']} | "
                     f"{m[0]['input_two_qubit_gates']} | {ak.median([r['two_qubit_gates'] for r in ok])} | "
                     f"{len(m) - len(ok)} |")

    opt = [r for r in rows if r["kind"] == "optimizer"]
    lines += ["", "## The optimizer (4colors Optimizer for QUBO)", "",
              "| graphs | model | variables | rows | time limit s | runs | median ratio | all feasible |",
              "|---|---|---|---|---|---|---|---|"]
    for large in (False, True):
        grp = [r for r in opt if bool(r.get("large")) == large]
        label = "larger (Brooks certificate)" if large else "small (enumerated)"
        for enc in dict.fromkeys(r["encoding"] for r in grp):
            mine = [r for r in grp if r["encoding"] == enc]
            if mine[0].get("note"):
                lines.append(f"| {label} | {enc} | {mine[0]['model_variables']} | | | | {mine[0]['note']} | |")
                continue
            for lim in sorted({r["time_limit_s"] for r in mine}):
                m = [r for r in mine if r["time_limit_s"] == lim]
                lines.append(f"| {label} | {enc} | {ak.median([r['model_variables'] for r in m])} | "
                             f"{ak.median([r['model_rows'] for r in m])} | {lim:g} | {len(m)} | "
                             f"{fmt(ak.median([r['classical_ratio'] for r in m]))} | "
                             f"{all(r['feasible'] for r in m)} |")
    (figs / "table.md").write_text("\n".join(lines) + "\n")

    # encodings.png: ours only, the three encodings side by side
    fig, axes = ak.figure(1, 2, figsize=(8, 3.4))
    for ax in axes:
        ak.axes_style(ax)
    labels = [f"{e}\np = {p}" for e, p in cells]
    axes[0].bar(range(len(cells)), [ours[c]["gates"] or 0 for c in cells], 0.55, color=ak.ARM_COLORS["4colors"])
    axes[0].set_ylabel("compiled two-qubit gates (median)", fontsize=8)
    axes[1].bar([i - 0.19 for i in range(len(cells))], [ours[c]["start"] or 0 for c in cells], 0.36,
                color=ak.ARM_COLORS["ideal"], label="start state")
    axes[1].bar([i + 0.19 for i in range(len(cells))], [ours[c]["noisy"] or 0 for c in cells], 0.36,
                color=ak.ARM_COLORS["4colors"], label="after QAOA, noisy")
    axes[1].set_ylabel("ratio to the optimum (median)", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(labels, fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "encodings.png", dpi=200)

    # compiler.png: shown cells only
    fig, ax = ak.figure(figsize=(5.5, 3.2))
    ak.axes_style(ax)
    for k, arm in enumerate(("4colors", "qiskit-o3")):
        ax.bar([i + (k - 0.5) * 0.38 for i in range(len(shown))], [s[arm]["gates"] for _, _, s in shown], 0.36,
               color=ak.ARM_COLORS[arm], label=ak.ARM_LABELS[arm])
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([f"{e}\np = {p}" for e, p, _ in shown], fontsize=7)
    ax.set_ylabel("compiled two-qubit gates (median)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "compiler.png", dpi=200)

    sel = ["# Compiler cells (internal, not for publication)", "", "## Shown", ""]
    sel += [f"- {e} p = {p}" for e, p, _ in shown] or ["- none"]
    sel += ["", "## Left out", ""]
    sel += [f"- {e} p = {p}: {why}" for e, p, why in left] or ["- none"]
    sel += ["", "## Line under the chart", "", UNDER_CHART]
    (figs / "selection.md").write_text("\n".join(sel) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
