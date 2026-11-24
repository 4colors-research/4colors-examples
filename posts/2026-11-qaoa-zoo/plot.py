"""The zoo tables and chart, from OUT/results.jsonl only.

Usage: python plot.py --out runs/zoo [--e2 3e-3]
Writes OUT/table.md (the post's table), OUT/no_price.md (every row with no finite price, and why), OUT/zoo.png,
OUT/selection.md, and OUT/table-full.md (every variant and arm: for the record, not for the post).

Price: "two-qubit gates per point of approximation ratio" = compiled two-qubit gates / (100 x (noisy ratio - random
ratio)): what one percentage point above a zero-gate random guess costs in gates. Computed per row (instance,
variant, arm, seed); the tables show the median over rows with a finite price and say how many rows that is.
RQAOA is priced over all its steps: its gates are the sum over every compiled recursion-step circuit. The noiseless
RQAOA row ("recursion simulated classically") compiles nothing and has no price.

A row has no finite price when its compile was refused, when it was too wide to simulate, when its noisy ratio is at
or below the random-guess ratio, or (RQAOA) when a recursion step was refused. Those rows are never dropped: they
are counted per reason in no_price.md.

Selection rule (owner, 22 Nov 2026): the chart and table.md show the variants where the 4colors-compiled circuit's
median noisy ratio is strictly above Qiskit O3's at the same budget. Others are listed in selection.md.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
from qaoakit import apikit as ak  # noqa: E402

ARMS = ("qiskit-o3", "4colors")
REASONS = ("refused", "too wide to simulate", "noisy ratio at or below random", "recursion step refused")


def price(r):
    """Gates per point of ratio above random, or None (no finite price)."""
    if r.get("two_qubit_gates") is None or r.get("noisy_ratio_mean") is None:
        return None
    gain = r["noisy_ratio_mean"] - r["random_ratio"]
    return r["two_qubit_gates"] / (100 * gain) if gain > 0 else None


def no_price_reason(r):
    """Why a qaoa / rqaoa_noisy row has no finite price (None if it has one)."""
    if r["kind"] == "rqaoa_noisy" and r.get("noisy_ratio_mean") is None:
        return "recursion step refused"
    if r.get("refusal_code") is not None or (r["kind"] == "qaoa" and r.get("two_qubit_gates") is None):
        return "refused"
    if r.get("simulated") is False:
        return "too wide to simulate"
    if price(r) is None:
        return "noisy ratio at or below random"
    return None


def scored_rows(rows, e2):
    """The rows the tables price at noise level e2: qaoa rows at e2, plus refused and unsimulated qaoa rows (which
    have no e2), plus the rqaoa_noisy rows at e2."""
    out = []
    for r in rows:
        if r["kind"] == "qaoa" and (r.get("e2") == e2 or "e2" not in r):
            out.append(r)
        elif r["kind"] == "rqaoa_noisy" and r.get("e2") == e2:
            out.append(r)
    return out


def cells(rows, e2):
    """{(variant, arm): summary} over the scored rows."""
    s = scored_rows(rows, e2)
    ideal = {}
    for r in rows:
        if r["kind"] == "variant":
            ideal.setdefault(r["variant"], []).append(r["ideal_ratio"])
    out = {}
    for v in dict.fromkeys(r["variant"] for r in s):
        for arm in ARMS:
            mine = [r for r in s if r["variant"] == v and r["arm"] == arm]
            if not mine:
                continue
            prices = [price(r) for r in mine]
            finite = [p for p in prices if p is not None]
            out[(v, arm)] = dict(
                rows=len(mine), gates=ak.median([r.get("two_qubit_gates") for r in mine]),
                ideal=ak.median(ideal.get(v, [])), noisy=ak.median([r.get("noisy_ratio_mean") for r in mine]),
                price=ak.median(finite), priced=len(finite),
                shots=ak.median([r.get("shots_total") for r in mine]),
                no_price={k: sum(no_price_reason(r) == k for r in mine) for k in REASONS})
    return out


def select_variants(c):
    """Variants where 4colors' median noisy ratio is strictly above Qiskit O3's (both present)."""
    vs = dict.fromkeys(v for v, _ in c)
    return [v for v in vs if (v, "4colors") in c and (v, "qiskit-o3") in c
            and c[(v, "4colors")]["noisy"] is not None and c[(v, "qiskit-o3")]["noisy"] is not None
            and c[(v, "4colors")]["noisy"] > c[(v, "qiskit-o3")]["noisy"]]


def fmt(x, nd=4):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def table(c, variants, head):
    lines = head + ["| variant | compiled by | rows | median 2q gates | median ideal ratio | median noisy ratio | "
                    "median gates per point | rows with a finite price |", "|---|---|---|---|---|---|---|---|"]
    for v in variants:
        for arm in ARMS:
            if (v, arm) not in c:
                continue
            x = c[(v, arm)]
            label = v + (" (all steps)" if v == "rqaoa-p1" else "")
            lines.append(f"| {label} | {ak.ARM_LABELS[arm]} | {x['rows']} | {fmt(x['gates'])} | {fmt(x['ideal'])} | "
                         f"{fmt(x['noisy'])} | {fmt(x['price'], 1)} | {x['priced']} |")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--e2", type=float, default=None, help="noise level for the tables (default: the first one)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows = ak.read_jsonl(out / "results.jsonl")
    e2s = [r["e2"] for r in rows if r["kind"] in ("qaoa", "rqaoa_noisy") and r.get("e2") is not None]
    e2 = a.e2 if a.e2 is not None else (e2s[0] if e2s else None)
    c = cells(rows, e2)
    shown = select_variants(c)
    every = list(dict.fromkeys(v for v, _ in c))
    note = f"Noise: e2 = {e2}, readout 1%. Medians over instances and seeds. Simulated, noise model as stated."
    rule = ("Variants where the 4colors-compiled circuit's median noisy ratio is above Qiskit O3's at the same "
            "budget. The full record, every variant included, is linked.")
    (out / "table.md").write_text("\n".join(table(c, shown, [note, rule, ""])) + "\n")
    (out / "table-full.md").write_text("\n".join(table(c, every, ["For the record, not for the post. " + note, ""]))
                                       + "\n")

    # rows with no finite price, per reason; the classically simulated recursion is listed as such
    nl = [f"Rows with no finite price. {note}", "",
          "| variant | compiled by | rows | " + " | ".join(REASONS) + " |", "|---|---|---|" + "---|" * len(REASONS)]
    for v in every:
        for arm in ARMS:
            if (v, arm) in c and sum(c[(v, arm)]["no_price"].values()):
                x = c[(v, arm)]
                nl.append(f"| {v} | {ak.ARM_LABELS[arm]} | {x['rows']} | "
                          + " | ".join(str(x["no_price"][k]) for k in REASONS) + " |")
    classical = [r for r in rows if r["kind"] == "variant" and r.get("pricing") == "recursion simulated classically"]
    if classical:
        nl += ["", f"rqaoa-p1, recursion simulated classically (exact p = 1 correlations, no circuit compiled): "
                   f"{len(classical)} instances, median ratio {fmt(ak.median([r['ideal_ratio'] for r in classical]))}; "
                   "no gate price."]
    (out / "no_price.md").write_text("\n".join(nl) + "\n")

    sel = ["# Selection", "", rule, "", f"Internal count (not for publication): {len(shown)} of {len(every)}",
           f"Shown: {shown}", f"Left out (4colors not strictly ahead, or an arm missing): "
           f"{[v for v in every if v not in shown]}"]
    (out / "selection.md").write_text("\n".join(sel) + "\n")

    cl = [r for r in rows if r["kind"] == "classical" and r.get("classical_ratio") is not None]
    ql = ["| 4colors Optimizer for QUBO | time limit s | runs | median ratio | worst ratio |", "|---|---|---|---|---|"]
    for lim in sorted({r["time_limit_s"] for r in cl}):
        mine = [r["classical_ratio"] for r in cl if r["time_limit_s"] == lim]
        ql.append(f"| | {lim:g} | {len(mine)} | {fmt(ak.median(mine))} | {fmt(min(mine))} |")
    with open(out / "table.md", "a") as f:
        f.write("\n" + "\n".join(ql) + "\n")
    print((out / "table.md").read_text())
    print((out / "no_price.md").read_text())

    fig, ax = ak.figure(figsize=(7.5, 3.6))
    ak.axes_style(ax)
    for k, arm in enumerate(ARMS):
        xs, ys = [], []
        for i, v in enumerate(shown):
            m = c[(v, arm)]["noisy"]
            if m is not None:
                xs.append(i + (k - 0.5) * 0.25)
                ys.append(m)
        if xs:
            ax.plot(xs, ys, "o", markersize=8, color=ak.ARM_COLORS[arm], label=f"QAOA, {ak.ARM_LABELS[arm]}, noisy")
    for lim in sorted({r["time_limit_s"] for r in cl}):
        m = ak.median([r["classical_ratio"] for r in cl if r["time_limit_s"] == lim])
        ax.axhline(m, color=ak.ARM_COLORS["4colors-qubo"], linewidth=1.5, linestyle="-" if lim >= 1 else "--")
        ax.text(max(len(shown), 1) - 0.5, m, f" QUBO optimizer, {lim:g} s", fontsize=7, color=ak.INK, va="bottom",
                ha="right")
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([v + (" (all steps)" if v == "rqaoa-p1" else "") for v in shown], fontsize=7, rotation=30)
    ax.set_ylabel("approximation ratio (vs exact optimum)", fontsize=9)
    if shown:
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "zoo.png", dpi=200)


if __name__ == "__main__":
    main()
