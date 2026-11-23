"""The efficient frontier chart, from results.jsonl only.

    python plot.py runs/n30 --out figures

frontier.png: risk (annual volatility) on x, return on y. The line is the reference portfolio for each q (exact
when the run enumerated); the dots are the solver's answers, one per seed and q.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))  # examplekit
import examplekit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    rows = examplekit.load_rows(a.inputs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in sorted({(r["n"], r["k"], r["data"]) for r in rows if r.get("kind") == "reference"}):
        n, k, data = key
        sel = [r for r in rows if (r.get("n"), r.get("k"), r.get("data")) == key]
        ref = sorted((r for r in sel if r["kind"] == "reference"), key=lambda r: r["q"])
        sol = [r for r in sel if r["kind"] == "solve" and r.get("feasible")]
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot([100 * r["risk"] for r in ref], [100 * r["ret"] for r in ref], color=examplekit.SERIES[0],
                label=f"reference ({ref[0]['method']})")
        ax.scatter([100 * r["risk"] for r in sol], [100 * r["ret"] for r in sol], color=examplekit.SERIES[1],
                   edgecolor=examplekit.SURFACE, linewidth=2, zorder=3, label="4colors answers")
        for r in ref:
            ax.annotate(f"q={r['q']:g}", (100 * r["risk"], 100 * r["ret"]), textcoords="offset points",
                        xytext=(6, -10), fontsize=7, color=examplekit.INK_2)
        ax.set_xlabel("risk: annual volatility (%)")
        ax.set_ylabel("expected annual return (%)")
        ax.set_title(f"pick {k} of {n} assets ({data})", color=examplekit.INK, fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        name = f"frontier_n{n}_k{k}.png"
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
