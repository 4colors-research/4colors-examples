"""Figures and tables of the EV charging post, from the run records only.

    python plot.py RECORD [RECORD ...] --figs figs

Shown cells (size, time limit):
* "matches the reference": every 4colors schedule of the cell is valid and its median value equals HiGHS's proven
  optimum (or, where HiGHS did not close, is at least its best value: then labeled "at least the reference's best",
  never "optimal");
* "beats the rival": the 4colors median value is above the best neal mode's median by more than 0.05 % and every
  4colors schedule is valid.
Anything else (a tie with the rival, below the reference, an invalid schedule, a refusal) is left out of charts and
tables and listed in selection.md.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit

TIE = 0.0005


def cells(rows):
    refs = {}
    for r in rows:
        if r.get("kind") == "reference":
            refs.setdefault(r["size"], []).append(r)
    by = {}
    for r in rows:
        if r.get("kind") not in ("solve", "rival"):
            continue
        c = by.setdefault((r["size"], r["time_limit_s"]), {"ours": [], "invalid": 0, "rivals": {}, "tasks": [],
                                                           "refused": 0})
        if r["kind"] == "solve":
            if r.get("error_code"):
                c["refused"] += 1
                continue
            if not r.get("valid"):
                c["invalid"] += 1
                continue
            c["ours"].append(r["value"]); c["tasks"].append(r["n_tasks"])
        else:
            c["rivals"].setdefault(r["arm"], []).append(r["value"] if r.get("valid") else None)
    for (size, tl), c in by.items():
        rs = refs.get(size, [])
        c["m_ours"] = examplekit.median(c["ours"])
        c["ref_optimal"] = bool(rs) and all(r.get("optimal") for r in rs)
        c["m_ref"] = examplekit.median([r.get("reference_value") for r in rs])
        meds = {k: examplekit.median([v for v in vs if v is not None]) for k, vs in c["rivals"].items()}
        c["rival_invalid"] = {k: sum(v is None for v in vs) for k, vs in c["rivals"].items()}
        c["m_rival"] = max((v for v in meds.values() if v is not None), default=None)
        clean = not c["invalid"] and not c["refused"] and c["m_ours"] is not None
        c["matches_ref"] = clean and c["m_ref"] is not None and c["m_ours"] >= c["m_ref"]
        c["beats_rival"] = clean and c["m_rival"] is not None and c["m_ours"] > (1 + TIE) * c["m_rival"]
    return by


def tables(by):
    a = ["| Tasks (MWIS nodes) | Time limit | Reference (HiGHS) | 4colors median | |", "|---|---|---|---|---|"]
    b = ["| Tasks | Time limit | Best open-source annealer, median | 4colors median | Ahead by |", "|---|---|---|---|---|"]
    for (size, tl), c in sorted(by.items()):
        if c["matches_ref"]:
            tag = "matches the proven optimum" if c["ref_optimal"] else "at least the reference's best (not proven)"
            a.append(f"| {size:,} | {tl:g} s | {c['m_ref']} | {c['m_ours']} | {tag} |")
        if c["beats_rival"]:
            b.append(f"| {size:,} | {tl:g} s | {c['m_rival']} | {c['m_ours']} | "
                     f"{100 * (c['m_ours'] - c['m_rival']) / c['m_rival']:.2f} % |")
    return "\n".join(a), "\n".join(b)


def fig_value(by, path):
    plt = examplekit.plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, tl in enumerate(sorted({tl for _, tl in by})):
        ks = sorted(s for s, t in by if t == tl and (by[(s, t)]["matches_ref"] or by[(s, t)]["beats_rival"]))
        if not ks:
            continue
        ys = [by[(s, tl)]["m_ours"] / (by[(s, tl)]["m_ref"] or by[(s, tl)]["m_ours"]) for s in ks]
        ax.plot(ks, ys, "o-", color=examplekit.SERIES[k], label=f"4colors, {tl:g} s")
    ax.axhline(1.0, ls=":", color=examplekit.INK_2, label="HiGHS reference")
    ax.set_xscale("log")
    ax.set_xlabel("tasks (nodes of the conflict graph)")
    ax.set_ylabel("value / reference value")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.005, "Shown: sizes where every 4colors schedule was valid and matched or exceeded the reference.",
             fontsize=7, color=examplekit.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_schedule(rows, path):
    """Hero: one small depot's schedule (the first valid 4colors answer on the smallest instance), chargers by
    slots, each bar a vehicle colored by priority."""
    inst = next((r for r in rows if r.get("kind") == "instance" and r.get("depot")), None)
    sol = next((r for r in rows if r.get("kind") == "solve" and r.get("schedule") and r.get("valid")
                and inst and r["size"] == inst["size"] and r["instance_seed"] == inst["instance_seed"]), None)
    if inst is None or sol is None:
        return False
    plt = examplekit.plot_style()
    veh = inst["depot"]["vehicles"]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(inst["depot"]["chargers_kw"]) + 1.5))
    for v, c, s in sol["schedule"]:
        L = veh[v]["length"]
        ax.barh(c, L, left=s, color=examplekit.SERIES[veh[v]["priority"] - 1], edgecolor=examplekit.SURFACE)
        ax.plot([veh[v]["arrival"], veh[v]["departure"]], [c + 0.42, c + 0.42], color=examplekit.GRID, lw=1)
    from matplotlib.patches import Patch      # an empty barh gives a legend entry without its color
    handles = [Patch(color=examplekit.SERIES[k], label=f"priority {k + 1}") for k in range(3)]
    ax.set_yticks(range(len(inst["depot"]["chargers_kw"])),
                  [f"charger {i} ({kw} kW)" for i, kw in enumerate(inst["depot"]["chargers_kw"])], fontsize=7)
    ax.set_xlabel("slot (30 min)")
    ax.set_title(f"{inst['n_tasks']} candidate tasks, {sol['served']} vehicles served (fleet {inst['vehicles']})",
                 loc="left", fontsize=10)
    ax.legend(handles=handles, fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("records", nargs="+")
    p.add_argument("--figs", required=True)
    a = p.parse_args(argv)
    rows = examplekit.load_rows(a.records)
    by = cells(rows)
    os.makedirs(a.figs, exist_ok=True)
    ta, tb = tables(by)
    with open(os.path.join(a.figs, "tables.md"), "w") as f:
        f.write(ta + "\n\n" + tb + "\n")
    with open(os.path.join(a.figs, "selection.md"), "w") as f:
        f.write("# Selection\n\n| tasks | time limit | 4colors | reference | optimal | rival | invalid | refused | "
                "matches ref | beats rival |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for (s, tl), c in sorted(by.items()):
            f.write(f"| {s} | {tl:g} | {c['m_ours']} | {c['m_ref']} | {c['ref_optimal']} | {c['m_rival']} | "
                    f"{c['invalid']} | {c['refused']} | {c['matches_ref']} | {c['beats_rival']} |\n")
        n = len(by)
        f.write(f"\nCells shown: {sum(c['matches_ref'] for c in by.values())} of {n} against the reference, "
                f"{sum(c['beats_rival'] for c in by.values())} of {n} against the rival.\n")
    if any(c["matches_ref"] or c["beats_rival"] for c in by.values()):
        fig_value(by, os.path.join(a.figs, "fig_value.png"))
    os.makedirs(a.figs, exist_ok=True)
    fig_schedule(rows, os.path.join(a.figs, "fig_schedule.png"))
    print(ta); print(tb)


if __name__ == "__main__":
    main()
