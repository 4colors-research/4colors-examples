"""Figures, a table and the chart selection for the broken-couplers post, from results.jsonl only.

    python plot.py results/broken --figs figs

fig_failed.png   the hero: the chip_native circuit on the chip, 4colors "not told" (the map alone) above
                 4colors "told" (map + calibration); dead couplers dashed red, gates on them drawn thick
fig_counts.png   gates on failed couplers and two-qubit gates per circuit, "not told" vs "told" (4colors arms only)
fig_esp.png      estimated success probability (a model score from the FakeFez calibration, not a measurement),
                 Qiskit on the full published map vs 4colors told, on the circuits the selection rule keeps;
                 0 is drawn at a marked floor: "0 = a gate on a dead coupler"
selection.json   which circuits fig_esp.png shows and which it leaves out, with the rule

Selection rule (stated under fig_esp.png in the post): a circuit is charted against Qiskit only if 4colors told has
a higher median estimated success probability than Qiskit on the full map at the same budget. Left-out circuits
are listed in selection.json and named in the text. The "not told" arm is never charted against Qiskit, because
Qiskit's arm gets the error rates and "not told" does not.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck

NOT_TOLD = ("4colors", "full", "4colors, not told (map only)")
TOLD = ("4colors", "full+calibration", "4colors, told (map + calibration)")
QISKIT_FULL = ("qiskit", "full", "Qiskit, full published map")
ESP_FLOOR = 1e-12
ESP_LABEL = ("estimated success probability (median)\nmodel estimate from the FakeFez calibration, "
             "not a measurement")


def median_of(rows, circuit_id, budget, arm, key):
    a, c, _ = arm
    return examplekit.median([r.get(key) for r in ck.results(rows, circuit_id=circuit_id, arm=a, coupler_set=c,
                                                           budget_s=budget)])


def select(rows, budget):
    """Apply the selection rule; return (shown, left_out) circuit ids."""
    shown, left = [], []
    for cid in sorted({r["circuit_id"] for r in ck.results(rows)}):
        q = median_of(rows, cid, budget, QISKIT_FULL, "estimated_esp")
        t = median_of(rows, cid, budget, TOLD, "estimated_esp")
        (shown if (q is not None and t is not None and t > q) else left).append(cid)
    return shown, left


def bars(plt, rows, cids, budget, arms, key, ylabel, fname, figs, log=False, floor_note=False, title=""):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(arms)
    for k, arm in enumerate(arms):
        xs, ys, bad = [], [], []
        for i, cid in enumerate(cids):
            v = median_of(rows, cid, budget, arm, key)
            if v is None:
                continue
            xs.append(i + (k - (len(arms) - 1) / 2) * width)
            ys.append(max(v, ESP_FLOOR) if log else v)
            bad.append(median_of(rows, cid, budget, arm, "gates_on_failed_couplers"))
        ax.bar(xs, ys, width * 0.92, color=examplekit.SERIES[k], label=arm[2])
        for x, y, f in zip(xs, ys, bad):
            if f:
                ax.annotate(f"{f:g} on dead", (x, y), ha="center", va="bottom", fontsize=7, color=examplekit.INK,
                            rotation=90, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(cids)), cids)
    if log:
        ax.set_yscale("log")
    if floor_note:
        ax.axhline(ESP_FLOOR, color="#e34948", lw=1, ls="--")
        ax.annotate("0 = a gate on a dead coupler", (0.01, ESP_FLOOR), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", fontsize=8, color="#e34948")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"FakeFez, {budget:g} s per compile", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(figs, fname), dpi=200)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--figs", default="figs")
    args = p.parse_args(argv)
    rows = examplekit.load_rows(args.inputs)
    os.makedirs(args.figs, exist_ok=True)
    plt = examplekit.plot_style()
    res = ck.results(rows)
    print(ck.table(res, ("circuit_id", "arm", "coupler_set", "budget_s"),
                   ("two_qubit_gates", "gates_on_failed_couplers", "estimated_esp")))
    device = next(r for r in rows if r.get("kind") == "device")
    budget = max(r["budget_s"] for r in res)
    cids = sorted({r["circuit_id"] for r in res})

    # Our own two arms: what changes when the compiler is told.
    bars(plt, rows, cids, budget, [NOT_TOLD, TOLD], "gates_on_failed_couplers",
         "two-qubit gates on failed couplers (median)", "fig_counts.png", args.figs,
         title=f"4colors, not told vs told: FakeFez, {budget:g} s per compile")

    # Told vs Qiskit on the full published map, on the circuits the rule keeps.
    shown, left = select(rows, budget)
    with open(os.path.join(args.figs, "selection.json"), "w") as f:
        json.dump({"rule": "chart a circuit against Qiskit only if 4colors told has a higher median estimated "
                           "success probability than Qiskit on the full map at the same budget",
                   "budget_s": budget, "shown": shown, "left_out": left}, f, indent=1)
    print(f"fig_esp.png shows {shown}; left out by the selection rule: {left}")
    bars(plt, rows, shown or cids, budget, [QISKIT_FULL, TOLD], "estimated_esp", ESP_LABEL, "fig_esp.png",
         args.figs, log=True, floor_note=True)

    # The hero: where the chip_native gates land, not told vs told.
    native = "chip_native" if "chip_native" in cids else cids[0]
    seed = min(r["seed"] for r in res)
    panels = [NOT_TOLD, TOLD]
    picks = [ck.results(rows, circuit_id=native, arm=a, coupler_set=c, seed=seed, budget_s=budget)
             for a, c, _ in panels]
    if all(picks):
        fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 4.4 * len(panels)), squeeze=False)
        for ax, (_, _, label), rs in zip(axes[:, 0], panels, picks):
            r = rs[0]
            ck.draw_device(ax, device["n_qubits"], device["couplers"], counts=r.get("coupler_counts"),
                           failed=device["failed"],
                           title=f"{label}: {r['gates_on_failed_couplers']} gates on dead couplers")
        fig.tight_layout()
        fig.savefig(os.path.join(args.figs, "fig_failed.png"), dpi=200)
        plt.close(fig)
    print(f"figures in {args.figs}")


if __name__ == "__main__":
    main()
