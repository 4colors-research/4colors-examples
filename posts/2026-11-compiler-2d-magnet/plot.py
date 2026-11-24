"""Figures for the 2D magnet post, from the run records only.

    python plot.py RECORD [RECORD ...] --figs figs

A record may also be the report card's (compiler-report-card): its tfim2d rows (2 steps) are read as a cross-check.

Cells are (grid, steps). A cell is shown when the 4colors median two-qubit gates per step is more than 2 % below
the lower of Qiskit O3's best-of and once arms (median over seeds), every 4colors compile returned a circuit, and
no Qiskit first trial overran the budget. Writes fig_lattice.png (hero), fig_per_step.png, fig_depth.png,
selection.md and table.md.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit
import circuitkit as ck
import tfim

TIE_BAND = 0.02
NOTE = ("Shown: grids where 4colors beat Qiskit O3 by more than 2 % per Trotter step (rule in the text). "
        "Open boundaries. Gate counts, not a hardware run.")


def from_report_card(rows):
    """The report card's tfim2d rows (compiler-report-card, square open grids, 2 Trotter steps) in this post's
    format, for the cross-check: grid, steps, per-step metrics and the floor. Other rows pass unchanged."""
    out = []
    for r in rows:
        if r.get("post") == "compiler-report-card" and r.get("family") == "tfim2d" and r.get("kind") == "result":
            k = round(r["qubits"] ** 0.5)
            g, d = r.get("two_qubit_gates"), r.get("two_qubit_depth")
            r = {**r, "grid": [k, k], "steps": 2, "logical_floor": tfim.logical_floor(k, k, 2),
                 "two_qubit_gates_per_step": None if g is None else g / 2,
                 "two_qubit_depth_per_step": None if d is None else d / 2, "source": "report card"}
        if r.get("post") == "compiler-report-card" and r.get("family") != "tfim2d":
            continue
        out.append(r)
    return out


def cells(rows, budget=10.0):
    res = [r for r in rows if r.get("kind") == "result" and r.get("budget_s") == budget]
    out = {}
    pairs = {}
    for r in res:
        key = (tuple(r["grid"]), r["steps"])
        c = out.setdefault(key, {"ours": [], "ours_depth": [], "rival": [], "rival_depth": [], "refused": 0,
                                 "over": 0, "labels": {}, "floor": r.get("logical_floor"), "qubits": r["qubits"]})
        if r["arm"] == "4colors":
            if r.get("two_qubit_gates") is None:
                c["refused"] += 1
                continue
            c["ours"].append(r["two_qubit_gates_per_step"]); c["ours_depth"].append(r["two_qubit_depth_per_step"])
            lab = r.get("verification_label") or "none"
            c["labels"][lab] = c["labels"].get(lab, 0) + 1
        else:
            pairs.setdefault((key, r["seed"]), {})[r["arm"]] = r
    for (key, _), arms in pairs.items():
        if len(arms) < 2:
            continue
        low = min(arms.values(), key=lambda r: (r["two_qubit_gates"], r["two_qubit_depth"]))
        c = out[key]
        c["over"] += int(bool(arms["qiskit-o3-best"].get("over_budget")))
        c["rival"].append(low["two_qubit_gates_per_step"]); c["rival_depth"].append(low["two_qubit_depth_per_step"])
    for c in out.values():
        for f in ("ours", "ours_depth", "rival", "rival_depth"):
            c["m_" + f] = examplekit.median(c[f])
        mo, mr = c["m_ours"], c["m_rival"]
        c["pct_fewer"] = None if mo is None or not mr else 100 * (mr - mo) / mr
        c["shown"] = (not c["refused"] and not c["over"] and mo is not None and mr is not None
                      and mo < (1 - TIE_BAND) * mr)
    return out


def caption(by):
    # Owner 23 Nov: no tallies in copy. The caption states the rule only; the count stays in selection.md (internal).
    return ("Open grids and step counts where the 4colors median used more than 2 % fewer two-qubit gates per "
            "Trotter step than the lower of Qiskit O3's best-of-trials and one default call, at the same 10 s, on "
            "FakeFez with its published calibration. The full record, every cell included, is linked from the run id.")


def table(by):
    lines = ["| Grid (open) | Qubits | Steps | Floor per step | Qiskit O3 per step | 4colors per step | Fewer |",
             "|---|---|---|---|---|---|---|"]
    for (g, s), c in sorted(by.items(), key=lambda kv: (kv[1]["qubits"], kv[0][1])):
        if c["shown"]:
            lines.append(f"| {g[0]} x {g[1]} | {c['qubits']} | {s} | {c['floor'] // s} | {c['m_rival']:.0f} | "
                         f"{c['m_ours']:.0f} | {c['pct_fewer']:.1f} % |")
    return "\n".join(lines)


def fig_lattice(path, rows=4, cols=4):
    """Left: the open grid with its 4 matchings colored. Right: the FakeFez map (degree at most 3)."""
    plt = examplekit.plot_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    for k, (name, es) in enumerate(tfim.matchings(rows, cols).items()):
        for u, v in es:
            (r0, c0), (r1, c1) = divmod(u, cols), divmod(v, cols)
            a.plot([c0, c1], [r0, r1], color=examplekit.SERIES[k], lw=3, label=name if (u, v) == es[0] else None)
    for q in range(rows * cols):
        r, c = divmod(q, cols)
        a.plot(c, r, "o", color=examplekit.INK_2, ms=6)
    a.invert_yaxis(); a.set_aspect("equal"); a.axis("off")
    a.set_title(f"{rows} x {cols} grid, open: 4 matchings, degree 4", loc="left", fontsize=10)
    a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4)
    sets, failed = ck.heavy_hex_coupler_sets()
    full = sets["full"]
    ck.draw_device(b, full.num_qubits, ck.device_couplers(full), failed=failed,
                   title="FakeFez heavy-hex: degree at most 3")
    fig.text(0.01, 0.01, "Each matching is one layer of two-qubit gates; a degree-4 site cannot sit on a degree-3 "
             "qubit without moves.", fontsize=7, color=examplekit.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_metric(by, path, ours, rival, ylabel, floor=False):
    plt = examplekit.plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for steps, marker in ((1, "o"), (2, "s")):
        ks = sorted((k for k, c in by.items() if c["shown"] and k[1] == steps), key=lambda k: by[k]["qubits"])
        if not ks:
            continue
        x = [by[k]["qubits"] for k in ks]
        ax.plot(x, [by[k][rival] for k in ks], marker=marker, color=examplekit.SERIES[1],
                label=f"Qiskit O3, {steps} step(s)")
        ax.plot(x, [by[k][ours] for k in ks], marker=marker, color=examplekit.SERIES[0],
                label=f"4colors, {steps} step(s)")
        if floor:
            ax.plot(x, [by[k]["floor"] / steps for k in ks], ls=":", color=examplekit.INK_2,
                    label="no routing (2 per bond)" if steps == 1 else None)
    ax.set_xlabel("qubits (open grid)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.005, NOTE, fontsize=7, color=examplekit.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("records", nargs="+")
    p.add_argument("--figs", required=True)
    p.add_argument("--budget", type=float, default=10.0)
    a = p.parse_args(argv)
    by = cells(from_report_card(examplekit.load_rows(a.records)), a.budget)
    os.makedirs(a.figs, exist_ok=True)
    with open(os.path.join(a.figs, "selection.md"), "w") as f:
        k = sum(c["shown"] for c in by.values())
        f.write("# Selection\n\n" + caption(by) + f"\n\nInternal (not for publication): cells shown {k} of {len(by)}.\n"
                "\n| grid | steps | shown | 4colors | rival | labels |\n"
                "|---|---|---|---|---|---|\n")
        for (g, s), c in sorted(by.items()):
            f.write(f"| {g[0]}x{g[1]} | {s} | {c['shown']} | {c['m_ours']} | {c['m_rival']} | {c['labels']} |\n")
    with open(os.path.join(a.figs, "table.md"), "w") as f:
        f.write(table(by) + "\n\n" + caption(by) + "\n")
    fig_lattice(os.path.join(a.figs, "fig_lattice.png"))
    if any(c["shown"] for c in by.values()):
        fig_metric(by, os.path.join(a.figs, "fig_per_step.png"), "m_ours", "m_rival",
                   "two-qubit gates per Trotter step (median)", floor=True)
        fig_metric(by, os.path.join(a.figs, "fig_depth.png"), "m_ours_depth", "m_rival_depth",
                   "two-qubit depth per Trotter step (median)")
    print(table(by)); print(caption(by))


if __name__ == "__main__":
    main()
