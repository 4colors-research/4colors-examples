"""Charts for the stop-and-stream post, from results.jsonl only.

    python plot.py runs/watch runs/stop5 runs/stop15 --out figures

stream_progress.png: cut against the service's search time (each final result's trajectory), one line per arm and
seed, with a marker where each job ended.
billed.png: billed seconds per job, by arm, beside the time limit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
import examplekit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    finals = [r for r in examplekit.load_rows(a.inputs) if r.get("kind") == "final"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = sorted({r["arm"] for r in finals}, key=lambda s: (s != "watch", s))
    color = {arm: examplekit.SERIES[k % len(examplekit.SERIES)] for k, arm in enumerate(arms)}

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labelled = set()
    for r in finals:
        pts = [(t, -obj) for t, obj in r["trajectory"] if t is not None and obj is not None]
        if not pts:
            continue
        lab = None if r["arm"] in labelled else r["arm"]
        labelled.add(r["arm"])
        ax.step(*zip(*pts), where="post", color=color[r["arm"]], label=lab)
        if r.get("billed_s") is not None:
            ax.scatter([r["billed_s"]], [pts[-1][1]], color=color[r["arm"]], edgecolor=examplekit.SURFACE, zorder=3)
    ax.set_xlabel("search time (s)")
    ax.set_ylabel("cut")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "stream_progress.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for k, arm in enumerate(arms):
        rs = [r for r in finals if r["arm"] == arm]
        xs = [k + 0.15 * (j - (len(rs) - 1) / 2) for j in range(len(rs))]
        ax.bar(xs, [r["billed_s"] or 0 for r in rs], 0.14, color=color[arm])
    if finals:
        ax.axhline(max(r["time_limit_s"] for r in finals), color=examplekit.INK_2, ls="--", lw=1)
        ax.text(len(arms) - 0.5, max(r["time_limit_s"] for r in finals), "time limit", va="bottom", ha="right",
                fontsize=8, color=examplekit.INK_2)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms)
    ax.set_ylabel("billed seconds per job")
    fig.tight_layout()
    fig.savefig(out / "billed.png", dpi=150)
    plt.close(fig)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
