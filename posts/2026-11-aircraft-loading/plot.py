"""Figures and tables of the aircraft loading post, from the run records only.

    python plot.py RECORD [RECORD ...] --figs figs

QUBO cells (hold, time limit) are shown when every 4colors plan is valid (plain-Python check) and the median load
equals HiGHS's proven optimum. Compile cells (circuit, budget) are shown when the 4colors median two-qubit count is
more than 2 % below the lower of Qiskit O3's best-of and once arms, every 4colors compile returned a circuit, and
no Qiskit first trial overran the budget. Writes fig_hold.png (hero), fig_compile.png, tables.md, selection.md.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit

TIE_BAND = 0.02


def qubo_cells(rows):
    refs = {}
    for r in rows:
        if r.get("kind") == "reference":
            refs[(r["hold"], r["hold_seed"])] = r
    by = {}
    for r in rows:
        if r.get("kind") != "solve":
            continue
        c = by.setdefault((r["hold"], r["time_limit_s"]), {"loads": [], "bad": 0, "match": [], "n_vars": r["n_vars"]})
        ref = refs.get((r["hold"], r["hold_seed"]))
        if r.get("error_code") or not r.get("valid"):
            c["bad"] += 1
            continue
        c["loads"].append(r["load_kg"])
        c["match"].append(ref is not None and ref.get("optimal") and r["load_kg"] == ref["reference_value"])
    for c in by.values():
        c["m_load"] = examplekit.median(c["loads"])
        c["shown"] = not c["bad"] and bool(c["match"]) and all(c["match"])
    return by


def compile_cells(rows):
    by, pairs = {}, {}
    for r in rows:
        if r.get("kind") != "result":
            continue
        key = (r["circuit_id"], r["budget_s"])
        c = by.setdefault(key, {"ours": [], "rival": [], "refused": 0, "over": 0, "qubits": r["qubits"],
                                "p": r.get("p"), "input_2q": r.get("input_2q"), "labels": {}})
        if r["arm"] == "4colors":
            if r.get("two_qubit_gates") is None:
                c["refused"] += 1
                continue
            c["ours"].append(r["two_qubit_gates"])
            lab = r.get("verification_label") or "none"
            c["labels"][lab] = c["labels"].get(lab, 0) + 1
        else:
            pairs.setdefault((key, r["seed"]), {})[r["arm"]] = r
    for (key, _), arms in pairs.items():
        if len(arms) == 2:
            by[key]["over"] += int(bool(arms["qiskit-o3-best"].get("over_budget")))
            by[key]["rival"].append(min(a["two_qubit_gates"] for a in arms.values()))
    for c in by.values():
        c["m_ours"], c["m_rival"] = examplekit.median(c["ours"]), examplekit.median(c["rival"])
        mo, mr = c["m_ours"], c["m_rival"]
        c["pct_fewer"] = None if mo is None or not mr else 100 * (mr - mo) / mr
        c["shown"] = not c["refused"] and not c["over"] and mo is not None and mr is not None and mo < (1 - TIE_BAND) * mr
    return by


def tables(qb, cb):
    a = ["| Hold | Variables | Time limit | Proven optimum load | 4colors median load |", "|---|---|---|---|---|"]
    for (h, tl), c in sorted(qb.items(), key=lambda kv: kv[1]["n_vars"]):
        if c["shown"]:
            a.append(f"| {h} | {c['n_vars']} | {tl:g} s | {c['m_load']} kg | {c['m_load']} kg (matches) |")
    b = ["| QAOA circuit | Qubits | p | Input 2q | Qiskit O3 | 4colors | Fewer |", "|---|---|---|---|---|---|---|"]
    for (cid, bud), c in sorted(cb.items(), key=lambda kv: (kv[1]["qubits"], kv[1]["p"] or 0)):
        if c["shown"]:
            b.append(f"| {cid} ({bud:g} s) | {c['qubits']} | {c['p']} | {c['input_2q']} | {c['m_rival']:.0f} | "
                     f"{c['m_ours']:.0f} | {c['pct_fewer']:.1f} % |")
    return "\n".join(a), "\n".join(b)


def fig_hold(rows, path):
    """Hero: one plan on the hold: positions along the fuselage (arm), container weights as bars against each
    position's cap, the center of gravity and the envelope."""
    inst = {(r["hold"], r["hold_seed"]): r for r in rows if r.get("kind") == "instance"}
    sol = next((r for r in rows if r.get("kind") == "solve" and r.get("valid") and r.get("plan")
                and r["hold"] == "grid20x20"), None) or next(
        (r for r in rows if r.get("kind") == "solve" and r.get("valid") and r.get("plan")), None)
    if sol is None:
        return False
    import loading

    hseed = sol["hold_seed"]
    h = (loading.paper_scale(int(sol["hold"][5:]), seed=hseed) if sol["hold"].startswith("paper") else
         loading.generate(sol["containers"], sol["positions"], seed=hseed))
    plt = examplekit.plot_style()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(h.arms, h.caps, width=1.6, color=examplekit.GRID, label="position cap")
    plan = {int(k): v for k, v in sol["plan"].items()}
    ax.bar([h.arms[j] for j in plan.values()], [h.weights[i] for i in plan], width=1.2, color=examplekit.SERIES[0],
           label="container loaded")
    ax.axvspan(h.lo, h.hi, color=examplekit.SERIES[2], alpha=0.15, label="CG envelope")
    ax.axvline(sol["cg"], color=examplekit.SERIES[1], lw=2, label=f"CG {sol['cg']:.2f}")
    ax.set_xlabel("arm (position along the hold, forward negative)")
    ax.set_ylabel("kg")
    ax.set_title(f"{sol['containers_loaded']} of {len(h.weights)} containers, {sol['load_kg']:,} kg of "
                 f"{h.payload:,} kg allowed", loc="left", fontsize=10)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def fig_compile(cb, path):
    plt = examplekit.plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, p in enumerate(sorted({c["p"] for c in cb.values() if c["p"]})):
        ks = sorted((key for key, c in cb.items() if c["shown"] and c["p"] == p), key=lambda key: cb[key]["qubits"])
        if not ks:
            continue
        x = [cb[key]["qubits"] for key in ks]
        ax.plot(x, [cb[key]["m_rival"] for key in ks], "s-", color=examplekit.SERIES[1], alpha=1 - 0.4 * k,
                label=f"Qiskit O3, p = {p}")
        ax.plot(x, [cb[key]["m_ours"] for key in ks], "o-", color=examplekit.SERIES[0], alpha=1 - 0.4 * k,
                label=f"4colors, p = {p}")
        ax.plot(x, [cb[key]["input_2q"] for key in ks], ":", color=examplekit.INK_2, label="input (no routing)" if k == 0 else None)
    ax.set_xlabel("qubits (one per container-position pair)")
    ax.set_ylabel("two-qubit gates on FakeFez (median)")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.005, "Shown: circuits where 4colors beat Qiskit O3 by more than 2 % at the same seconds. "
             "Gate counts, not a hardware run.", fontsize=7, color=examplekit.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("records", nargs="+")
    p.add_argument("--figs", required=True)
    a = p.parse_args(argv)
    rows = examplekit.load_rows(a.records)
    qb, cb = qubo_cells(rows), compile_cells(rows)
    os.makedirs(a.figs, exist_ok=True)
    ta, tb = tables(qb, cb)
    with open(os.path.join(a.figs, "tables.md"), "w") as f:
        f.write(ta + "\n\n" + tb + "\n")
    with open(os.path.join(a.figs, "selection.md"), "w") as f:
        f.write("# Selection\n\n| QUBO cell | shown | bad rows | median load |\n|---|---|---|---|\n")
        for k, c in sorted(qb.items()):
            f.write(f"| {k} | {c['shown']} | {c['bad']} | {c['m_load']} |\n")
        f.write("\n| compile cell | shown | 4colors | rival | labels |\n|---|---|---|---|---|\n")
        for k, c in sorted(cb.items()):
            f.write(f"| {k} | {c['shown']} | {c['m_ours']} | {c['m_rival']} | {c['labels']} |\n")
        f.write(f"\nCells shown: {sum(c['shown'] for c in qb.values())} of {len(qb)} (QUBO), "
                f"{sum(c['shown'] for c in cb.values())} of {len(cb)} (compile).\n")
    fig_hold(rows, os.path.join(a.figs, "fig_hold.png"))
    if any(c["shown"] for c in cb.values()):
        fig_compile(cb, os.path.join(a.figs, "fig_compile.png"))
    print(ta); print(tb)


if __name__ == "__main__":
    main()
