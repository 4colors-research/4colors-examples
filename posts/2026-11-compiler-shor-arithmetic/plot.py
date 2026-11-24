"""Figures and the table of the adder post, from the run records only.

    python plot.py RECORD [RECORD ...] --figs figs

A cell is (adder, bits, budget). It is shown when the 4colors median two-qubit gate count is more than 2 % below
the lower of Qiskit O3's best-of-trials and its first trial (median over seeds), every 4colors compile of the cell
returned a circuit that passed the sum check, and no Qiskit first trial overran the budget.
Writes fig_counts.png (hero), fig_depth.png, table.md and selection.md.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit

TIE_BAND = 0.02
NOTE = ("Shown: sizes where 4colors beat Qiskit O3 by more than 2 % and every 4colors adder returned the right sums. "
        "Gate counts, not a hardware run.")
LABEL = {"cuccaro": "Cuccaro (2n+2 qubits)", "vbe": "VBE (3n+1 qubits)"}


def cells(rows):
    out, pairs = {}, {}
    for r in rows:
        if r.get("kind") != "result":
            continue
        key = (r["family"], r["bits"], r["budget_s"])
        c = out.setdefault(key, {"ours": [], "ours_d": [], "rival": [], "rival_d": [], "bad": 0, "over": 0,
                                 "rival_wrong": 0, "labels": {}, "qubits": r["qubits"],
                                 "floor": r.get("toffoli_counted_input")})
        if r["arm"] == "4colors":
            if r.get("two_qubit_gates") is None or r.get("adder_check") != "returns_sum":
                c["bad"] += 1
                continue
            c["ours"].append(r["two_qubit_gates"]); c["ours_d"].append(r["two_qubit_depth"])
            lab = r.get("verification_label") or "none"
            c["labels"][lab] = c["labels"].get(lab, 0) + 1
        else:
            pairs.setdefault((key, r["seed"]), {})[r["arm"]] = r
    for (key, _), arms in pairs.items():
        if len(arms) < 2:
            continue
        c = out[key]
        c["over"] += int(bool(arms["qiskit-o3-best"].get("over_budget")))
        c["rival_wrong"] += sum(1 for a in arms.values() if a.get("adder_check") == "wrong")
        low = min(arms.values(), key=lambda r: (r["two_qubit_gates"], r["two_qubit_depth"]))
        c["rival"].append(low["two_qubit_gates"]); c["rival_d"].append(low["two_qubit_depth"])
    for c in out.values():
        for f in ("ours", "ours_d", "rival", "rival_d"):
            c["m_" + f] = examplekit.median(c[f])
        mo, mr = c["m_ours"], c["m_rival"]
        c["pct_fewer"] = None if mo is None or not mr else 100 * (mr - mo) / mr
        c["shown"] = not c["bad"] and not c["over"] and mo is not None and mr is not None and mo < (1 - TIE_BAND) * mr
    return out


def caption(by):
    return ("Adder sizes and budgets where the 4colors median used more than 2 % fewer two-qubit gates than the "
            "lower of Qiskit O3's best-of-trials and one default call, at the same budget, on FakeFez with its "
            "published calibration, and every 4colors adder returned the right sum on every input tried. "
            "The full record, every cell included, is linked from the run id.")


def table(by):
    lines = ["| Adder | Bits | Qubits | Budget | Toffoli-counted input | Qiskit O3 | 4colors | Fewer |",
             "|---|---|---|---|---|---|---|---|"]
    for (k, n, b), c in sorted(by.items()):
        if c["shown"]:
            lines.append(f"| {k} | {n} | {c['qubits']} | {b:g} s | {c['floor']} | {c['m_rival']:.0f} | "
                         f"{c['m_ours']:.0f} | {c['pct_fewer']:.1f} % |")
    return "\n".join(lines)


def fig(by, path, ours, rival, ylabel, budget, floor):
    plt = examplekit.plot_style()
    kinds = sorted({k[0] for k in by})
    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 3.8), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        ks = sorted((k for k, c in by.items() if c["shown"] and k[0] == kind and k[2] == budget),
                    key=lambda k: by[k]["qubits"])
        x = [by[k]["qubits"] for k in ks]
        ax.plot(x, [by[k][rival] for k in ks], "s-", color=examplekit.SERIES[1], label="Qiskit O3")
        ax.plot(x, [by[k][ours] for k in ks], "o-", color=examplekit.SERIES[0], label=f"4colors, {budget:g} s")
        if floor:
            ax.plot(x, [by[k]["floor"] for k in ks], ":", color=examplekit.INK_2, label="input, Toffoli = 6, no routing")
        ax.set_title(LABEL.get(kind, kind), loc="left", fontsize=10)
        ax.set_xlabel("qubits")
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
    p.add_argument("--budget", type=float, default=10.0, help="the budget the figures show")
    a = p.parse_args(argv)
    by = cells(examplekit.load_rows(a.records))
    os.makedirs(a.figs, exist_ok=True)
    with open(os.path.join(a.figs, "selection.md"), "w") as f:
        f.write("# Selection\n\n" + caption(by) + "\n\nInternal (not for publication): cells shown "
                f"{sum(c['shown'] for c in by.values())} of {len(by)}.\n\n| adder | bits | budget | shown | 4colors | rival | "
                "bad 4colors rows | wrong rival rows | labels |\n|---|---|---|---|---|---|---|---|---|\n")
        for (k, n, b), c in sorted(by.items()):
            f.write(f"| {k} | {n} | {b:g} | {c['shown']} | {c['m_ours']} | {c['m_rival']} | {c['bad']} | "
                    f"{c['rival_wrong']} | {c['labels']} |\n")
    with open(os.path.join(a.figs, "table.md"), "w") as f:
        f.write(table(by) + "\n\n" + caption(by) + "\n")
    if any(c["shown"] for c in by.values()):
        fig(by, os.path.join(a.figs, "fig_counts.png"), "m_ours", "m_rival", "two-qubit gates (median)", a.budget, True)
        fig(by, os.path.join(a.figs, "fig_depth.png"), "m_ours_d", "m_rival_d", "two-qubit depth (median)", a.budget,
            False)
    print(table(by)); print(caption(by))


if __name__ == "__main__":
    main()
