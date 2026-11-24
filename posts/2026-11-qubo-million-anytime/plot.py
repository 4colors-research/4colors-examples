"""Figures of the anytime study, from analyze.py's analysis.json only (never from raw rows by hand).

    python plot.py $RUN/analysis/analysis.json --out $RUN/figures

gap_by_budget_<family>.png  ours and the best rival setting per budget, only the cells the selection rule admits
time_to_target.png          per rival setting, the budget bracket in which it reaches our 10 s gap
first_answer.png            time to the first answer at the 1 s budget, ours and each rival that answered
Every chart carries its selection rule in the caption line under the axes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys as _sys                           # noqa: E402
from pathlib import Path as _Path             # noqa: E402
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import examplekit as bk

RULE = ("Shown: cells where 4colors' median gap beats the best rival setting at the same budget by more than "
        "0.05 points (10 seeds). The full record, every cell included, is linked from the run id.")
NAMES = {"ea3d": "3D spin glass, 1M spins", "torus2d": "2D spin glass (torus), 1M spins",
         "regular": "random 3-regular Max-Cut, 1M nodes"}


def gap_by_budget(rep, fam, out, plt):
    rows = [t for t in rep["table"] if t["family"] == fam and t["win"]]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4))
    b = [t["budget_s"] for t in rows]
    ax.plot(b, [100 * t["ours"] for t in rows], marker="o", color=bk.SERIES[0], label="4colors (API)")
    ax.plot(b, [100 * t["rival"] for t in rows], marker="s", color=bk.SERIES[1], label="best other solver per budget")
    for t in rows:
        ax.annotate(t["best_rival"], (t["budget_s"], 100 * t["rival"]), textcoords="offset points", xytext=(4, 4),
                    fontsize=8, color=bk.INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("time budget (s)")
    ax.set_ylabel("gap to the reference (%)")
    ax.set_title(NAMES.get(fam, fam), loc="left")
    ax.legend()
    n_ran = sum(1 for t in rep["table"])
    fig.text(0.01, 0.01, RULE.format(k=len(rep["shown"]), n=n_ran), fontsize=7, color=bk.INK_2, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = out / f"gap_by_budget_{fam}.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def time_to_target(rep, out, plt):
    rows = [t for t in rep["ttt"] if t["factor_high"] is not None or t["never_reached"]]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 0.5 + 0.45 * len(rows)))
    for k, t in enumerate(rows):
        label = f"{NAMES.get(t['family'], t['family']).split(',')[0]}: {t['setting']}"
        if t["never_reached"]:
            ax.barh(k, t["largest_budget_tried"] / t["target_budget_s"], color=bk.GRID)
            ax.text(t["largest_budget_tried"] / t["target_budget_s"], k, "  not reached", va="center", fontsize=8)
        else:
            lo, hi = t["factor_low"], t["factor_high"]
            ax.barh(k, hi - lo, left=lo, color=bk.SERIES[1])
        ax.text(-0.02, k, label, ha="right", va="center", fontsize=8, transform=ax.get_yaxis_transform())
    ax.axvline(1, color=bk.SERIES[0], linewidth=2)
    ax.set_yticks([])
    ax.set_xlabel("time the other solver needs, as a multiple of 4colors' 10 s (bracket)")
    fig.tight_layout()
    p = out / "time_to_target.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def first_answer(rep, out, plt):
    rows = [t for t in rep["first_answer"] if t["median_s"] is not None]
    ours = {t["family"]: t["median_s"] for t in rows if t["setting"] == "api" and not t["no_answer"]}
    # selection rule: a family is shown only where our first answer is the fastest of every arm that answered
    rows = [t for t in rows if t["family"] in ours and all(ours[t["family"]] < u["median_s"] for u in rows
                                                            if u["family"] == t["family"] and u["setting"] != "api")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 0.5 + 0.4 * len(rows)))
    for k, t in enumerate(rows):
        ax.barh(k, t["median_s"], color=bk.SERIES[0] if t["setting"] == "api" else bk.SERIES[1])
        ax.text(t["median_s"], k, f"  {t['median_s']:.1f} s" + (f" ({t['no_answer']} runs with no answer)"
                                                               if t["no_answer"] else ""), va="center", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{t['family']}: {'4colors (API)' if t['setting'] == 'api' else t['setting']}" for t in rows],
                       fontsize=8)
    ax.axvline(1, color=bk.INK_2, linewidth=1, linestyle="--")
    ax.set_xlabel("seconds to the first answer at a 1 s budget (median)")
    fig.tight_layout()
    p = out / "first_answer.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("analysis")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = json.loads(Path(a.analysis).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt = bk.plot_style()
    made = [gap_by_budget(rep, f, out, plt) for f in sorted({t["family"] for t in rep["table"]})]
    made += [time_to_target(rep, out, plt), first_answer(rep, out, plt)]
    made = [str(p) for p in made if p is not None]
    for p in made:
        print(p)
    return made


if __name__ == "__main__":
    main()
