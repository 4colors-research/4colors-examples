"""Tables and a figure for the verification post, from results.jsonl only.

    python plot.py results/verify --figs figs

Prints every 4colors compile's label, distance and reason (plus the local check for small circuits) and the
background demo; writes fig_labels.png: two-qubit gates per size and mode, each bar labeled with its
verification label.

Rule D2: a Qiskit gate count is drawn and printed only where Qiskit's own distance from the input was measured
(`local_infidelity` on its row: the circuits small enough for the local check). Elsewhere Qiskit's level 3 may have
dropped near-identity gates (the QFT's tiny controlled phases) by an unknown amount, so its count is not comparable
and is left out.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck


def mode(r):
    return (r.get("options") or {}).get("approximation_mode")


MODE_LABEL = {"exact": "exact", "per_block": "per_block: approximate, may drop near-identity gates"}


def qiskit_measured(rows):
    """Qiskit result rows whose distance from the input was measured (D2); the only ones a post may show."""
    return [r for r in ck.results(rows, arm="qiskit") if r["coupler_set"] != "lower_of_both"
            and r.get("local_infidelity") is not None]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--figs", default="figs")
    args = p.parse_args(argv)
    rows = examplekit.load_rows(args.inputs)
    os.makedirs(args.figs, exist_ok=True)
    plt = examplekit.plot_style()
    four = ck.results(rows, arm="4colors")

    print("| circuit | mode | seed | 2q gates | label | distance | reason | approximation spent | local check |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(four, key=lambda r: (r["qubits"], str(mode(r)), r["seed"])):
        local = "" if r.get("local_equivalent") is None else \
            f"{'equivalent' if r['local_equivalent'] else 'differs'} (1 - F = {r['local_infidelity']:.1e})"
        print(f"| {r['circuit_id']} | {mode(r)} | {r['seed']} | {r['two_qubit_gates']} | {r['verification_label']} | "
              f"{r['verification_distance']} | {r['verification_reason'] or ''} | {r.get('approximation_spent')} | "
              f"{local or r.get('refusal_code') or ''} |")
    q = qiskit_measured(rows)
    if q:
        print("\nQiskit level 3, only where its distance from the input was measured (D2):")
        for r in sorted(q, key=lambda r: (r["qubits"], r["seed"])):
            print(f"  {r['circuit_id']} seed {r['seed']}: {r['two_qubit_gates']} two-qubit gates, "
                  f"1 - F = {r['local_infidelity']:.1e}")
    for r in rows:
        if r.get("kind") == "background":
            print(f"\nverify=background on {r['circuit_id']}: '{r['label_at_return']}' at return, "
                  f"'{r['verification_label']}' after {r.get('waited_s')} s of waiting")

    cids = sorted({r["circuit_id"] for r in four}, key=lambda c: int(c.split("_")[1]))
    modes = sorted({mode(r) for r in four}, key=str)
    arms = [("4colors", m) for m in modes] + [("qiskit", None)]
    width = 0.8 / max(1, len(arms))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, (arm, m) in enumerate(arms):
        for i, cid in enumerate(cids):
            rs = ([r for r in q if r["circuit_id"] == cid] if arm == "qiskit" else
                  [r for r in ck.results(rows, circuit_id=cid, arm=arm) if mode(r) == m])
            y = examplekit.median([r.get("two_qubit_gates") for r in rs])
            if y is None:
                continue
            x = i + (k - (len(arms) - 1) / 2) * width
            ax.bar([x], [y], width * 0.92, color=examplekit.SERIES[k],
                   label=(f"4colors, {MODE_LABEL.get(m, m)}" if arm == "4colors" else
                          "Qiskit level 3 (distance measured)"))
            if arm == "4colors":
                labels = Counter(r["verification_label"] for r in rs)
                ax.annotate(", ".join(f"{l}" + (f" x{c}" if c > 1 else "") for l, c in labels.items()),
                            (x, y), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            rotation=90, fontsize=7, color=examplekit.INK)
    ax.set_xticks(range(len(cids)), cids)
    ax.set_yscale("log")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("two-qubit gates (median)")
    ax.set_title("Gates and the label each compile came back with", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    keep = dict(zip(labels, handles))                     # one legend entry per series
    ax.legend(keep.values(), keep.keys(), fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_labels.png"), dpi=200); plt.close(fig)
    print(f"figures in {args.figs}")


if __name__ == "__main__":
    main()
