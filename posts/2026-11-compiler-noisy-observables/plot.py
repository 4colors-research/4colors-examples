"""Tables and figures for the noisy-observables post, from sim.jsonl (simulate.py) only.

    python plot.py results/sim --figs figs

fig_observables.png  error of each shown circuit's observable under the noise model, both arms (hero)
fig_trotter.png      TFIM magnetization against Trotter steps: ideal, 4colors and Qiskit under the noise model
fig_gates.png        two-qubit gates against observable error, every shown cell, both arms

Selection rule (owner, 22 Nov 2026: wins only, rule stated): a circuit is shown when the 4colors median two-qubit
count is strictly below Qiskit O3's AND the 4colors median observable error is below Qiskit's by more than twice
the larger shot-noise standard error (medians over compile seeds). A TFIM lattice is shown in fig_trotter when
4colors reaches strictly more Trotter steps than Qiskit before the magnetization error first exceeds THRESHOLD.
Everything else stays in sim.jsonl and is listed in selection.md with the line that goes under each chart. Every
chart says: "simulated with a noise model built from the published calibration; not a hardware run".
"""
import argparse
import os
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit

ARMS = [("4colors", "4colors"), ("qiskit", "Qiskit O3, published calibration as-is")]
LABEL = "Simulated with a noise model built from the published calibration; not a hardware run."
THRESHOLD = 0.1          # magnetization error at which a Trotter series "breaks" [CHECK wording with product lead]


def cell(rows, cid, arm, field="error"):
    return examplekit.median([r.get(field) for r in rows if r["circuit_id"] == cid and r["arm"] == arm])


def select(rows):
    """[(circuit_id) shown], [(circuit_id, why) left out]."""
    shown, left = [], []
    for cid in dict.fromkeys(r["circuit_id"] for r in rows):
        g = {a: cell(rows, cid, a, "two_qubit_gates") for a, _ in ARMS}
        e = {a: cell(rows, cid, a) for a, _ in ARMS}
        se = max(cell(rows, cid, a, "error_se") or 0 for a, _ in ARMS)
        if None in g.values() or None in e.values():
            left.append((cid, "an arm has no simulated answer"))
        elif not g["4colors"] < g["qiskit"]:
            left.append((cid, "not fewer two-qubit gates"))
        elif not e["4colors"] < e["qiskit"] - 2 * se:
            left.append((cid, "error not lower by more than 2 standard errors"))
        else:
            shown.append(cid)
    return shown, left


def steps_reached(rows, lattice, arm, threshold=THRESHOLD):
    """The largest Trotter step count s with error <= threshold at every step count up to s (0 if the first fails)."""
    series = {}
    for r in rows:
        m = re.fullmatch(rf"tfim_{lattice}_s(\d+)", r["circuit_id"])
        if m and r["arm"] == arm:
            series.setdefault(int(m.group(1)), []).append(r["error"])
    reached = 0
    for s in sorted(series):
        if examplekit.median(series[s]) > threshold:
            break
        reached = s
    return reached


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--figs", default="figs")
    args = p.parse_args(argv)
    allrows = examplekit.load_rows([os.path.join(x, "sim.jsonl") if os.path.isdir(x) else x for x in args.inputs])
    rows = [r for r in allrows if r.get("kind") == "sim"]
    ideal = {r["circuit_id"]: r for r in allrows if r.get("kind") == "ideal"}
    os.makedirs(args.figs, exist_ok=True)
    plt = examplekit.plot_style()
    shown, left = select(rows)

    print(LABEL)
    print("| circuit | domain | 4colors 2q | Qiskit 2q | 4colors error | Qiskit error | shot SE |")
    print("|---|---|---|---|---|---|---|")
    for cid in shown:
        fam = next(r["family"] for r in rows if r["circuit_id"] == cid)
        se = max(cell(rows, cid, a, "error_se") for a, _ in ARMS)
        print(f"| {cid} | {fam} | {cell(rows, cid, '4colors', 'two_qubit_gates'):g} | "
              f"{cell(rows, cid, 'qiskit', 'two_qubit_gates'):g} | {cell(rows, cid, '4colors'):.4f} | "
              f"{cell(rows, cid, 'qiskit'):.4f} | {se:.4f} |")

    lattices = sorted({m.group(1) for r in rows if (m := re.fullmatch(r"tfim_(\dx\d)_s\d+", r["circuit_id"]))})
    reach = {lat: {a: steps_reached(rows, lat, a) for a, _ in ARMS} for lat in lattices}
    tshown = [lat for lat in lattices if reach[lat]["4colors"] > reach[lat]["qiskit"]]
    for lat in lattices:
        print(f"TFIM {lat}: Trotter steps before error > {THRESHOLD}: 4colors {reach[lat]['4colors']}, "
              f"Qiskit {reach[lat]['qiskit']}{'' if lat in tshown else ' (not shown)'}")

    bars = [c for c in shown if not c.startswith("tfim_")]
    if bars:
        fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(bars) + 2), 3.8))
        w = 0.38
        for k, (arm, label) in enumerate(ARMS):
            ax.bar([i + (k - 0.5) * w for i in range(len(bars))], [cell(rows, c, arm) for c in bars], w * 0.92,
                   color=examplekit.SERIES[k], label=label)
        ax.set_xticks(range(len(bars)), bars, fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("observable error vs ideal (median)")
        ax.set_title("Same circuit, same noise model: error of the answer", loc="left", fontsize=10)
        ax.legend(fontsize=8)
        fig.text(0.01, 0.01, LABEL, fontsize=7)
        fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(args.figs, "fig_observables.png"), dpi=200)
        plt.close(fig)

    if tshown:
        fig, axes = plt.subplots(1, len(tshown), figsize=(4.6 * len(tshown), 3.8), squeeze=False)
        for ax, lat in zip(axes[0], tshown):
            steps = sorted({int(r["circuit_id"].rsplit("_s", 1)[1]) for r in rows
                            if r["circuit_id"].startswith(f"tfim_{lat}_s")})
            ax.plot(steps, [ideal[f"tfim_{lat}_s{s}"]["magnetization"] for s in steps], color="black", ls="--",
                    label="ideal (exact)")
            for k, (arm, label) in enumerate(ARMS):
                ax.plot(steps, [cell(rows, f"tfim_{lat}_s{s}", arm, "magnetization") for s in steps], marker="o",
                        color=examplekit.SERIES[k], label=label)
            ax.set_xlabel("Trotter steps")
            ax.set_ylabel("magnetization <Z> (median)")
            ax.set_title(f"TFIM {lat}", loc="left", fontsize=10)
        axes[0][0].legend(fontsize=8)
        fig.text(0.01, 0.01, LABEL, fontsize=7)
        fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(args.figs, "fig_trotter.png"), dpi=200)
        plt.close(fig)

    if shown:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, (arm, label) in enumerate(ARMS):
            ax.scatter([cell(rows, c, arm, "two_qubit_gates") for c in shown], [cell(rows, c, arm) for c in shown],
                       color=examplekit.SERIES[k], label=label, s=22)
        ax.set_xscale("log")
        ax.set_xlabel("two-qubit gates (median)")
        ax.set_ylabel("observable error vs ideal (median)")
        ax.legend(fontsize=8)
        fig.text(0.01, 0.01, LABEL, fontsize=7)
        fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(args.figs, "fig_gates.png"), dpi=200)
        plt.close(fig)

    total = len(shown) + len(left)
    with open(os.path.join(args.figs, "selection.md"), "w") as f:
        f.write("# Selection\n\n")
        f.write(f"Internal (not for publication): circuits shown {len(shown)} of {total}. Under fig_observables and "
                f"fig_gates: \"Circuits where 4colors used fewer two-qubit gates than Qiskit O3 and its observable "
                f"error under the noise model was lower by more than two shot-noise standard errors (median of the "
                f"seeds). The full record, every cell included, is linked from the run id. {LABEL}\"\n\n")
        f.write(f"Internal (not for publication): TFIM lattices shown {len(tshown)} of {len(lattices)}. Under "
                f"fig_trotter: \"Lattices where the 4colors circuits stay within {THRESHOLD} of the ideal "
                f"magnetization for more Trotter steps than Qiskit O3's. The full record is linked from the run "
                f"id. {LABEL}\"\n\n")
        for c in shown:
            f.write(f"- shown: {c}\n")
        for c, why in left:
            f.write(f"- left out: {c} ({why})\n")
        for lat in lattices:
            f.write(f"- TFIM {lat}: steps reached 4colors {reach[lat]['4colors']}, Qiskit {reach[lat]['qiskit']}"
                    f"{'' if lat in tshown else ' (left out)'}\n")
    print(f"figures in {args.figs}")


if __name__ == "__main__":
    main()
