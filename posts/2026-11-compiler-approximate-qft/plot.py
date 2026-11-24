"""Tables and figures for the approximate-QFT post, from results.jsonl only.

    python plot.py results/aqft --figs figs

Each configuration is (size, arm, input, mode): its median two-qubit gates over seeds and its measured infidelity
from the exact QFT (`infidelity_vs_qft`, the one local check for both compilers). For ours the WORST seed's
infidelity is used, for Qiskit the BEST seed's: any doubt goes against us.

Matching (D2: same approximation): for each Qiskit configuration q, our candidates are the configurations of the same
size whose worst infidelity is at most max(q's best infidelity, FLOOR); the match is the candidate with the fewest
two-qubit gates. FLOOR (1e-9) is the check's numerical floor: two circuits below it are both "exact" to the check.

Selection rule (stated under fig_pairs.png): a pair is shown when the match exists and has fewer two-qubit gates
than q, q's best was found within the budget (`over_budget` reported apart), and both sides were checked. Every
other Qiskit configuration is listed in selection.json with its reason.

Writes fig_pairs.png (hero: the kept pairs), fig_own.png (ours only: gates against measured infidelity, what each
approximation buys), labels.md (ours: verification label, service distance, approximation_spent, our check, and the
[CHECK] flags), selection.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck

FLOOR = 1e-9


def configs(rows):
    """{(qubits, arm, input_id, mode): summary} over seeds."""
    groups = {}
    for r in ck.results(rows):
        groups.setdefault((r["qubits"], r["arm"], r.get("input_id"), r.get("mode")), []).append(r)
    out = {}
    for key, rs in groups.items():
        gates = [r["two_qubit_gates"] for r in rs if r.get("two_qubit_gates") is not None]
        inf = [r["infidelity"] for r in rs if r.get("infidelity") is not None]
        ours = key[1] == "4colors"
        out[key] = {"qubits": key[0], "arm": key[1], "input_id": key[2], "mode": key[3],
                    "two_qubit_gates": examplekit.median(gates) if gates else None,
                    "infidelity": (max(inf) if ours else min(inf)) if inf and len(inf) == len(rs) else None,
                    "over_budget": any(r.get("over_budget") for r in rs), "seeds": len(rs),
                    "refusals": sorted({r["refusal_code"] for r in rs if r.get("refusal_code")})}
    return out


def pairs(rows):
    """Every Qiskit configuration with its match and verdict (see the module docstring)."""
    cfg = configs(rows)
    ours = [c for c in cfg.values() if c["arm"] == "4colors" and c["two_qubit_gates"] is not None
            and c["infidelity"] is not None]
    out = []
    for q in sorted((c for c in cfg.values() if c["arm"] == "qiskit"), key=lambda c: (c["qubits"], c["input_id"],
                                                                                      str(c["mode"]))):
        rec = {"qiskit": q, "match": None, "shown": False}
        if q["infidelity"] is None or q["two_qubit_gates"] is None:
            rec["why"] = "Qiskit circuit not checked"
        elif q["over_budget"]:
            rec["why"] = "Qiskit's first transpile overran the budget: reported apart"
        else:
            cands = [c for c in ours if c["qubits"] == q["qubits"] and c["infidelity"] <= max(q["infidelity"], FLOOR)]
            if not cands:
                rec["why"] = "no 4colors configuration at the same or smaller measured distance"
            else:
                m = min(cands, key=lambda c: (c["two_qubit_gates"], c["infidelity"]))
                rec["match"] = m
                if m["two_qubit_gates"] < q["two_qubit_gates"]:
                    rec["shown"], rec["why"] = True, "4colors fewer two-qubit gates at the same or smaller distance"
                else:
                    rec["why"] = "4colors not fewer at the same distance (tie or behind): named in the text"
        out.append(rec)
    return out


def flags(r):
    """[CHECK] flags on one 4colors row (open product findings from the 30 Oct runs)."""
    f = []
    if r.get("mode") == "per_block" and r.get("approximation_spent") is None and r.get("two_qubit_gates") is not None:
        f.append("approximation_spent missing")
    if r.get("verification_label") == "verified" and (r.get("infidelity") or 0) > FLOOR:
        f.append("verified but the check measures a distance")
    d = r.get("verification_distance")
    if d is not None and r.get("infidelity") is not None and r["infidelity"] > max(FLOOR, d):
        f.append("check far above the service distance (dropped-block phase?)")
    return f


def label_lines(rows):
    lines = ["| circuit | seed | mode | 2q | label | service distance | approximation_spent | our check | flags |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in ck.results(rows, arm="4colors"):
        lines.append(f"| {r['circuit_id']} | {r['seed']} | {r.get('mode')} | {r.get('two_qubit_gates')} | "
                     f"{r.get('verification_label')} | {r.get('verification_distance')} | "
                     f"{r.get('approximation_spent')} | {r.get('infidelity')} | {'; '.join(flags(r))} |")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--figs", default="figs")
    args = p.parse_args(argv)
    rows = examplekit.load_rows(args.inputs)
    os.makedirs(args.figs, exist_ok=True)
    plt = examplekit.plot_style()
    pr = pairs(rows)
    shown = [x for x in pr if x["shown"]]

    def name(c):
        return f"{c['input_id']}, {c['mode']}"

    fig, ax = plt.subplots(figsize=(7.5, 0.6 + 0.5 * max(1, len(shown))))
    ys = list(range(len(shown)))
    ax.barh([y + 0.2 for y in ys], [x["qiskit"]["two_qubit_gates"] for x in shown], height=0.4, color=examplekit.INK_2,
            label="Qiskit level 3")
    ax.barh([y - 0.2 for y in ys], [x["match"]["two_qubit_gates"] for x in shown], height=0.4,
            color=examplekit.SERIES[0], label="4colors")
    for y, x in zip(ys, shown):
        ax.text(x["qiskit"]["two_qubit_gates"], y + 0.2, f"  {x['qiskit']['infidelity']:.1e}", va="center", fontsize=7)
        ax.text(x["match"]["two_qubit_gates"], y - 0.2, f"  {x['match']['infidelity']:.1e}", va="center", fontsize=7)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{x['qiskit']['qubits']} q: Qiskit {name(x['qiskit'])} vs ours {name(x['match'])}"
                        for x in shown], fontsize=7)
    ax.set_xlabel("two-qubit gates (median over seeds); label = measured infidelity from the exact QFT")
    ax.set_title("Same distance or closer, fewer gates (rule in selection.json)", loc="left", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_pairs.png"), dpi=200); plt.close(fig)

    cfg = [c for c in configs(rows).values() if c["arm"] == "4colors" and c["infidelity"] is not None
           and c["two_qubit_gates"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k, n in enumerate(sorted({c["qubits"] for c in cfg})):
        cs = sorted((c for c in cfg if c["qubits"] == n), key=lambda c: c["infidelity"])
        ax.plot([max(c["infidelity"], FLOOR) for c in cs], [c["two_qubit_gates"] for c in cs], marker="o",
                color=examplekit.SERIES[k % len(examplekit.SERIES)], label=f"{n} qubits")
    ax.set_xscale("log")
    ax.set_xlabel(f"measured infidelity from the exact QFT (values below {FLOOR:g} drawn at it)")
    ax.set_ylabel("two-qubit gates")
    ax.set_title("4colors only: exact, per_block and textbook AQFT inputs", loc="left", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.figs, "fig_own.png"), dpi=200); plt.close(fig)

    with open(os.path.join(args.figs, "labels.md"), "w") as fh:
        fh.write("\n".join(label_lines(rows)) + "\n")
    with open(os.path.join(args.figs, "selection.json"), "w") as fh:
        json.dump({"rule": "match: our fewest-gate configuration at worst infidelity <= max(Qiskit best infidelity, "
                           f"{FLOOR:g}); shown when it has fewer two-qubit gates, Qiskit within budget, both checked",
                   "shown": shown, "left_out": [x for x in pr if not x["shown"]]}, fh, indent=1)
    for x in pr:
        m = x["match"]
        print(f"{x['qiskit']['qubits']} q Qiskit {name(x['qiskit'])}: {x['qiskit']['two_qubit_gates']} at "
              f"{x['qiskit']['infidelity']} | ours {name(m) if m else '-'}: {m and m['two_qubit_gates']} at "
              f"{m and m['infidelity']} | {x['why']}")
    print(f"figures in {args.figs}: {len(shown)} pairs shown, {len(pr) - len(shown)} left out")


if __name__ == "__main__":
    main()
