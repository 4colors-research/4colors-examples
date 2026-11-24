"""Gap and pixel error by budget, from results.jsonl only.

    python plot.py results/*/results.jsonl --out figures

Reference per (instance, region): the exact min-cut energy where there is one (submodular variant),
else the lowest energy found by any row ("best found", never "optimum"). Gap = (E - E_ref) / |E_ref|.
Writes figures/denoise_gap.png and prints a table.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import examplekit as bk


def references(rows):
    """{(instance, region): (energy, kind)}: exact when an exact row exists, else best found."""
    ref = {}
    for r in rows:
        if r.get("energy") is None:
            continue
        key = (r["instance"], r["region"])
        if r["event"] == "exact":
            ref[key] = (r["energy"], "exact")
        elif ref.get(key, (None, ""))[1] != "exact":
            if key not in ref or r["energy"] < ref[key][0]:
                ref[key] = (r["energy"], "best found")
    return ref


def table(rows):
    ref = references(rows)
    groups = defaultdict(list)
    for r in rows:
        if r.get("event") in ("solve", "icm", "noisy") and r.get("energy") is not None:
            groups[(r["instance"], r["region"], r.get("solver", r["event"]), r.get("budget_s"))].append(r)
    out = {}
    for (inst, reg, solver, T), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        e_ref, kind = ref[(inst, reg)]
        out[(inst, reg, solver, T)] = {
            "runs": len(rs), "reference": kind,
            "median_gap": statistics.median((r["energy"] - e_ref) / abs(e_ref) for r in rs),
            "median_pixel_error": statistics.median(r["pixel_error"] for r in rs),
            "all_agree": all(r.get("agrees", True) for r in rs)}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--out", default="figures")
    a = p.parse_args(argv)
    rows = [r for path in a.jsonl for r in bk.Results(path).rows()]
    t = table(rows)
    print("instance, region, solver, budget_s, runs, reference, median gap, median pixel error, re-checks agree")
    for (inst, reg, solver, T), s in t.items():
        print(f"{inst}, {reg}, {solver}, {T}, {s['runs']}, {s['reference']}, {100 * s['median_gap']:.4f}%, "
              f"{100 * s['median_pixel_error']:.3f}%, {s['all_agree']}")
    ours = {k: v for k, v in t.items() if k[2] == "4colors"}
    if not ours:
        return t
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    series = sorted({(k[0], k[1]) for k in ours})
    for i, (inst, reg) in enumerate(series):
        pts = sorted((k[3], 100 * v["median_gap"]) for k, v in ours.items() if (k[0], k[1]) == (inst, reg))
        label = inst if reg == "full" else f"{inst} ({reg.split(' at')[0]})"
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=bk.SERIES[i % 8], marker=bk.MARKERS[i % 8],
                linewidth=2, markersize=8, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("time budget (s)")
    ax.set_ylabel("energy gap to reference (%)")
    ax.set_title("Image denoising: gap to the exact or best-found energy", loc="left", fontsize=11)
    bk.style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(a.out) / "denoise_gap.png", dpi=150, bbox_inches="tight")
    return t


if __name__ == "__main__":
    main()
