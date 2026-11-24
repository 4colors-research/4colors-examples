"""Heatmaps and landscape metrics for the landscape post, from OUT/results.jsonl only.

Usage: python plot.py --out runs/p2
Writes:
  OUT/landscape-<instance>.png   ideal, then Qiskit O3 and 4colors under noise (seed 1), one color scale
  OUT/kept-vs-predicted.png      contrast kept (measured) against the first-order prediction from the gate counts
  OUT/metrics.md                 per noisy row: two-qubit gates, 1q pulses, contrast, contrast kept, predicted
                                 contrast kept, argmax distance, max ratio, mirror and repeat checks; then the
                                 instances the selection rule shows and the line that goes under the figure

Contrast is max minus min of the expected-cut ratio over the grid; contrast kept is noisy over ideal.
Predicted contrast kept, first order (a global-depolarizing approximation): every CZ keeps a fraction (1 - e2) of a
Pauli expectation, every sx or x a fraction (1 - e1), and symmetric readout flips scale each ZZ term by
(1 - 2 e_ro)^2, so  kept ~ (1 - e2)^N_cz * (1 - e1)^N_1q * (1 - 2 e_ro)^2.  Under this uniform noise model the
landscape is, to first order, a gate-count plot; the argmax distance is the part the counts do not predict.
The argmax distance is measured to the nearer of the ideal best point and its mirror (pi - gamma, beta), since the
p = 1 landscape of a 3-regular graph has both.

Selection rule: an instance is shown when the 4colors circuit keeps more contrast than Qiskit O3's (median over
seeds, same budget, same noise model). Every row stays in metrics.md.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
from qaoakit import apikit as ak  # noqa: E402


def argmax_point(r):
    g = np.array(r["ratio_grid"])
    i, j = np.unravel_index(int(g.argmax()), g.shape)
    return r["gammas"][i], r["betas"][j]


def predicted_kept(n_cz, n_1q, e2, e1, e_ro):
    """First-order contrast kept under the uniform model (see the module docstring)."""
    return (1 - e2) ** n_cz * (1 - e1) ** n_1q * (1 - 2 * e_ro) ** 2


def argmax_distance(r, ideal):
    (a, b), (c, d) = argmax_point(r), argmax_point(ideal)
    return min(math.hypot(a - c, b - d), math.hypot((math.pi - a) - c, b - d))


def metrics(ideal, r):
    g, gi = np.array(r["ratio_grid"]), np.array(ideal["ratio_grid"])
    out = dict(contrast=float(np.ptp(g)), contrast_kept=float(np.ptp(g) / np.ptp(gi)),
               argmax_distance=argmax_distance(r, ideal), max_ratio=float(g.max()))
    if r.get("condition") == "noisy" and r.get("one_qubit_pulses") is not None:
        out["predicted_kept"] = predicted_kept(r["two_qubit_gates"], r["one_qubit_pulses"], r["e2"], r["e1"],
                                               r["e_ro"])
    return out


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else (xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2)


def select_instances(rows):
    """(shown, left): instances where ours keeps more contrast than Qiskit O3 (median over seeds)."""
    shown, left = [], []
    for inst in dict.fromkeys(r["instance"] for r in rows):
        ideal = next(r for r in rows if r["instance"] == inst and r["arm"] == "ideal")
        kept = {arm: median([metrics(ideal, r)["contrast_kept"] for r in rows if r["instance"] == inst
                             and r.get("condition") == "noisy" and r["arm"] == arm]) for arm in ("4colors", "qiskit-o3")}
        if kept["4colors"] is None or kept["qiskit-o3"] is None:
            left.append((inst, "an arm has no noisy landscape (refused or not simulated)"))
        elif kept["4colors"] > kept["qiskit-o3"]:
            shown.append((inst, kept["4colors"], kept["qiskit-o3"]))
        else:
            left.append((inst, "tie or behind"))
    return shown, left


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows = ak.read_jsonl(out / "results.jsonl")
    lines = ["| instance | arm | seed | 2q gates | 1q pulses | e2 | contrast | contrast kept | predicted kept | "
             "argmax distance (rad) | max ratio | mirror check | repeat check |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    shown, left = select_instances(rows)
    scatter = []
    for inst in dict.fromkeys(r["instance"] for r in rows):
        ideal = next(r for r in rows if r["instance"] == inst and r["arm"] == "ideal")
        noisy = [r for r in rows if r["instance"] == inst and r.get("condition") == "noisy"]
        m = metrics(ideal, ideal)
        lines.append(f"| {inst} | ideal | | | | 0 | {m['contrast']:.4f} | 1 | | 0 | {m['max_ratio']:.4f} | "
                     f"{ideal.get('ideal_symmetry_max_abs', '')} | |")
        for r in noisy:
            m = metrics(ideal, r)
            scatter.append((r["arm"], m.get("predicted_kept"), m["contrast_kept"]))
            fmt = lambda v: "" if v is None else f"{v:.2e}"
            lines.append(f"| {inst} | {r['arm']} | {r['seed']} | {r['two_qubit_gates']} | {r.get('one_qubit_pulses')} | "
                         f"{r['e2']:g} | {m['contrast']:.4f} | {m['contrast_kept']:.3f} | "
                         f"{'' if m.get('predicted_kept') is None else format(m['predicted_kept'], '.3f')} | "
                         f"{m['argmax_distance']:.3f} | {m['max_ratio']:.4f} | {fmt(r.get('mirror_check_max_abs'))} | "
                         f"{fmt(r.get('repeat_check_max_abs'))} |")
        if inst not in [s[0] for s in shown]:
            continue
        first = lambda arm: min((r for r in noisy if r["arm"] == arm), key=lambda r: r["seed"])
        panels = [ideal] + [first(arm) for arm in ("qiskit-o3", "4colors") if any(r["arm"] == arm for r in noisy)]
        lo = min(np.min(r["ratio_grid"]) for r in panels)
        hi = max(np.max(r["ratio_grid"]) for r in panels)
        fig, axes = ak.figure(1, len(panels), figsize=(3.2 * len(panels), 3.0), squeeze=False)
        for ax, r in zip(axes[0], panels):
            ext = [r["betas"][0], r["betas"][-1], r["gammas"][-1], r["gammas"][0]]
            im = ax.imshow(r["ratio_grid"], extent=ext, aspect="auto", cmap="Blues", vmin=lo, vmax=hi)
            title = ak.ARM_LABELS[r["arm"]] + ("" if r["arm"] == "ideal" else
                                               f"\n{r['two_qubit_gates']} two-qubit gates, noisy")
            ax.set_title(title, fontsize=8, color=ak.INK, loc="left")
            ax.set_xlabel("beta", fontsize=8)
            ax.tick_params(labelsize=7, colors=ak.MUTED, labelcolor=ak.INK)
        axes[0][0].set_ylabel("gamma", fontsize=8)
        fig.colorbar(im, ax=list(axes[0]), shrink=0.8, label="expected cut / maximum cut (simulated)")
        fig.savefig(out / f"landscape-{inst}.png", dpi=200)
    pts = [(arm, p, k) for arm, p, k in scatter if p is not None]
    if pts:
        fig, axes = ak.figure(1, 1, figsize=(4.0, 3.6), squeeze=False)
        ax = axes[0][0]
        for arm in ("qiskit-o3", "4colors"):
            xs = [p for a_, p, _ in pts if a_ == arm]
            ys = [k for a_, _, k in pts if a_ == arm]
            if xs:
                ax.scatter(xs, ys, s=24, color=ak.ARM_COLORS[arm], label=ak.ARM_LABELS[arm])
        lo = min(min(p, k) for _, p, k in pts)
        ax.plot([lo, 1], [lo, 1], ls="--", color=ak.MUTED, lw=1, label="kept = predicted")
        ax.set_xlabel("predicted from gate counts (first order)", fontsize=8)
        ax.set_ylabel("contrast kept (simulated)", fontsize=8)
        ax.legend(fontsize=7)
        fig.savefig(out / "kept-vs-predicted.png", dpi=200, bbox_inches="tight")
    total = len(shown) + len(left)
    lines += ["", f"Under the figure: \"Graphs where the 4colors circuit kept more of the landscape's contrast than "
                  f"Qiskit O3's, compiled with the same seconds and simulated under the same stated noise model "
                  f"(median of seeds). The full record, every graph included, is linked.\"",
              f"Internal count (not for publication): {len(shown)} of {total} graphs shown.", ""]
    lines += [f"- shown: {i} (kept {o:.3f} vs {q:.3f})" for i, o, q in shown]
    lines += [f"- left out: {i}: {why}" for i, why in left]
    (out / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
