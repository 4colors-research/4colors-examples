"""Figures and tables of the referee post, from the run records only.

    python plot.py RECORD [RECORD ...] --figs figs

Reference side (no contest, every size shown): per instance, the best-known cut is the certified optimum where the
exhaustive search ran, else the best 4colors cut over all time limits (labeled "best found"). For G(n, 1/2) the
table sets it beside Q-score's asymptotic reference and gives the beta a perfect solver would score.
Compile side (a contest, wins only): a circuit is shown when the 4colors median two-qubit count is more than 2 %
below the lower of Qiskit O3's best-of and once arms, every 4colors compile returned a circuit, and no Qiskit first
trial overran the budget. Writes fig_beta.png (hero), fig_compile.png, tables.md and selection.md.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit
import referee as R

TIE_BAND = 0.02


def best_known(rows):
    """{(family, n, seed): {"value", "certified", "agree"}}: `agree` is True when every time limit returned the same
    best cut (a stability signal for uncertified sizes), and, where certified, whether 4colors reached it."""
    inst = {}
    for r in rows:
        k = (r.get("family"), r.get("n"), r.get("instance_seed"))
        if r.get("kind") == "instance":
            inst.setdefault(k, {"cuts": {}, "exact": None, "ref": r.get("qscore_reference")})
    for r in rows:
        k = (r.get("family"), r.get("n"), r.get("instance_seed"))
        if k not in inst:
            continue
        if r.get("kind") == "exact":
            inst[k]["exact"] = r["optimum"]
        if r.get("kind") == "solve" and r.get("cut") is not None:
            tl = r["time_limit_s"]
            inst[k]["cuts"][tl] = max(inst[k]["cuts"].get(tl, -1), r["cut"])
    out = {}
    for k, v in inst.items():
        found = max(v["cuts"].values(), default=None)
        value = v["exact"] if v["exact"] is not None else found
        out[k] = {"value": value, "certified": v["exact"] is not None, "found": found,
                  "reached_certificate": None if v["exact"] is None or found is None else found == v["exact"],
                  "stable": len(set(v["cuts"].values())) <= 1 if v["cuts"] else None, "ref": v["ref"],
                  "first_limit_cut": v["cuts"].get(min(v["cuts"])) if v["cuts"] else None}
    return out


def beta_table(bk):
    lines = ["| n | Q-score reference (n^2/8 + 0.178 n^1.5) | Best-known cut, mean of 3 | Certified? | "
             "beta a perfect solver scores |", "|---|---|---|---|---|"]
    for n in sorted({k[1] for k in bk if k[0] == "er"}):
        ks = [k for k in bk if k[0] == "er" and k[1] == n and bk[k]["value"] is not None]
        if not ks:
            continue
        vals = [bk[k]["value"] for k in ks]
        betas = [R.beta_needed(n, v) for v in vals]
        cert = "yes (exhaustive)" if all(bk[k]["certified"] for k in ks) else "best found"
        lines.append(f"| {n} | {R.expected_optimum(n):.1f} | {sum(vals) / len(vals):.1f} | {cert} | "
                     f"{min(betas):.2f} to {max(betas):.2f} |")
    return "\n".join(lines)


def compile_cells(rows):
    by, pairs = {}, {}
    for r in rows:
        if r.get("kind") != "result":
            continue
        key = (r["circuit_id"], r["budget_s"])
        c = by.setdefault(key, {"ours": [], "rival": [], "refused": 0, "over": 0, "n": r.get("n"), "p": r.get("p"),
                                "graph": r.get("graph"), "input_2q": r.get("input_2q"), "labels": {}})
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
        # how many cost layers fit in the two-qubit budget one layer costs Qiskit
        c["layers_per_rival_budget"] = None if not c["shown"] or not c["p"] else (mr / c["p"]) / (mo / c["p"])
    return by


def compile_table(cb):
    lines = ["| Graph | n | p | Input 2q | Qiskit O3 | 4colors | Fewer | Layers in one Qiskit layer's budget |",
             "|---|---|---|---|---|---|---|---|"]
    for key, c in sorted(cb.items(), key=lambda kv: (kv[1]["graph"] or "", kv[1]["n"] or 0)):
        if c["shown"]:
            lines.append(f"| {c['graph']} | {c['n']} | {c['p']} | {c['input_2q']} | {c['m_rival']:.0f} | "
                         f"{c['m_ours']:.0f} | {c['pct_fewer']:.1f} % | {c['layers_per_rival_budget']:.2f} |")
    return "\n".join(lines)


def fig_beta(bk, path):
    plt = examplekit.plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    pts = sorted((k[1], R.beta_needed(k[1], v["value"]), v["certified"]) for k, v in bk.items()
                 if k[0] == "er" and v["value"] is not None)
    if not pts:
        return False
    ax.scatter([p[0] for p in pts if p[2]], [p[1] for p in pts if p[2]], color=examplekit.SERIES[0],
               label="certified optimum (exhaustive)")
    ax.scatter([p[0] for p in pts if not p[2]], [p[1] for p in pts if not p[2]], facecolors="none",
               edgecolors=examplekit.SERIES[0], label="best found (4colors, not certified)")
    ax.axhline(1.0, ls=":", color=examplekit.INK_2, label="beta = 1: the asymptotic optimum")
    ax.axhline(R.PASS_BETA, ls="--", color=examplekit.SERIES[1], label="pass threshold 0.2")
    ax.set_xlabel("n (G(n, 1/2) instances, 3 per size)")
    ax.set_ylabel("beta of the instance's best-known cut")
    ax.legend(fontsize=8)
    ax.set_title("What a perfect solver would score", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def fig_compile(cb, path):
    plt = examplekit.plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, g in enumerate(("fc", "er")):
        ks = sorted((key for key, c in cb.items() if c["shown"] and c["graph"] == g), key=lambda key: cb[key]["n"])
        if not ks:
            continue
        x = [cb[key]["n"] for key in ks]
        per = lambda key, f: cb[key][f] / cb[key]["p"]
        name = "fully connected, weighted" if g == "fc" else "G(n, 1/2)"
        ax.plot(x, [per(key, "m_rival") for key in ks], "s--" if k else "s-", color=examplekit.SERIES[1],
                label=f"Qiskit O3, {name}")
        ax.plot(x, [per(key, "m_ours") for key in ks], "o--" if k else "o-", color=examplekit.SERIES[0],
                label=f"4colors, {name}")
    ax.set_yscale("log")
    ax.set_xlabel("qubits")
    ax.set_ylabel("two-qubit gates per QAOA layer on FakeFez")
    ax.legend(fontsize=7)
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
    bk, cb = best_known(rows), compile_cells(rows)
    os.makedirs(a.figs, exist_ok=True)
    with open(os.path.join(a.figs, "tables.md"), "w") as f:
        f.write(beta_table(bk) + "\n\n" + compile_table(cb) + "\n")
    with open(os.path.join(a.figs, "selection.md"), "w") as f:
        f.write("# Selection\n\nReference side: every instance is listed (no contest).\n\n"
                "| family | n | seed | value | certified | 4colors reached certificate | stable across limits |\n"
                "|---|---|---|---|---|---|---|\n")
        for k, v in sorted(bk.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0, kv[0][2] or 0)):
            f.write(f"| {k[0]} | {k[1]} | {k[2]} | {v['value']} | {v['certified']} | {v['reached_certificate']} | "
                    f"{v['stable']} |\n")
        f.write("\n| compile cell | shown | 4colors | rival | labels |\n|---|---|---|---|---|\n")
        for k, c in sorted(cb.items()):
            f.write(f"| {k} | {c['shown']} | {c['m_ours']} | {c['m_rival']} | {c['labels']} |\n")
        f.write(f"\nCompile cells shown: {sum(c['shown'] for c in cb.values())} of {len(cb)}.\n")
    fig_beta(bk, os.path.join(a.figs, "fig_beta.png"))
    if any(c["shown"] for c in cb.values()):
        fig_compile(cb, os.path.join(a.figs, "fig_compile.png"))
    print(beta_table(bk)); print(compile_table(cb))


if __name__ == "__main__":
    main()
