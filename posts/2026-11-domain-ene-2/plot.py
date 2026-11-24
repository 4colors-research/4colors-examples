"""Charts for the grid-partitioning post, from results.jsonl only.

    python plot.py runs/grid runs/grid-sa --out figures --draw pglib_opf_case118_ieee:4 --case-file data/pglib_opf_case118_ieee.m

1. grid_partition.png (hero): one grid drawn with a seeded spring layout, buses colored by part, cut branches thick,
   from the best balanced answer of the chosen (case, parts). Needs the case file to redraw the graph.
2. grid_gap.png: on grids with a proven MILP optimum, our median cut over seeds as a gap to it, per time limit.
3. grid_vs_sa.png: bisection cells, our median cut against the annealer's median best feasible cut at the same time
   limit. Selection rule: only (case, time limit) cells where our median is strictly lower are drawn, and only from
   rival rows of a timed run; the title says how many of all cells that is.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
import examplekit
import grid as gr


def ours(rows):
    """{(case, parts, time limit): [cut of balanced, feasible answers]}."""
    out = defaultdict(list)
    for r in rows:
        if r.get("kind") == "solve" and r.get("balanced") and r.get("model_feasible") and r.get("cut") is not None:
            out[(r["case"], r["parts"], r["time_limit_s"])].append(r["cut"])
    return dict(out)


def optima(rows):
    return {(r["case"], r["parts"]): r["cut"] for r in rows
            if r.get("kind") == "reference" and r.get("status") == "optimal"}


def gaps(rows):
    """[(case, parts, time limit, median gap in %)] where a proven optimum exists."""
    opt = optima(rows)
    out = []
    for (case, parts, t), cuts in sorted(ours(rows).items()):
        if (case, parts) in opt and opt[(case, parts)]:
            out.append((case, parts, t, 100 * (examplekit.median(cuts) - opt[(case, parts)]) / opt[(case, parts)]))
    return out


def vs_rival(rows):
    """(all cells, winning cells): [(case, time limit, our median, rival median)] for bisection."""
    riv = defaultdict(list)
    for r in rows:
        if r.get("kind") == "rival" and not r.get("tuning") and r.get("cut") is not None:
            riv[(r["case"], r["time_limit_s"])].append(r["cut"])
    cells = []
    for (case, parts, t), cuts in sorted(ours(rows).items()):
        if parts == 2 and (case, t) in riv:
            cells.append((case, t, examplekit.median(cuts), examplekit.median(riv[(case, t)])))
    return cells, [c for c in cells if c[2] < c[3]]


def best_part(rows, case, parts):
    best = None
    for r in rows:
        if (r.get("kind") == "solve" and r.get("case") == case and r.get("parts") == parts and r.get("balanced")
                and r.get("model_feasible") and (best is None or r["cut"] < best["cut"])):
            best = r
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--draw", help="case:parts for the hero figure")
    p.add_argument("--case-file", help="the MATPOWER file of the --draw case")
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    rows = examplekit.load_rows(a.inputs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.draw and a.case_file:
        import networkx as nx

        case, parts = a.draw.rsplit(":", 1)
        parts = int(parts)
        g = gr.to_graph(gr.parse_matpower(Path(a.case_file).read_text()), "unit")
        row = best_part(rows, case, parts)
        if row is not None and row.get("part") is not None:
            part = row["part"]
            G = nx.Graph()
            G.add_nodes_from(range(g["n"]))
            G.add_edges_from(g["edges"])
            pos = nx.spring_layout(G, seed=1)
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            cut = [e for e in g["edges"] if part[e[0]] != part[e[1]]]
            keep = [e for e in g["edges"] if part[e[0]] == part[e[1]]]
            nx.draw_networkx_edges(G, pos, edgelist=keep, ax=ax, edge_color=examplekit.GRID, width=0.8)
            nx.draw_networkx_edges(G, pos, edgelist=cut, ax=ax, edge_color=examplekit.INK, width=2.2)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=14 if g["n"] > 300 else 30,
                                   node_color=[examplekit.SERIES[q % 8] for q in part])
            ax.set_axis_off()
            ax.set_title(f"{case}: {parts} parts, {len(cut)} branches cut", color=examplekit.INK, fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "grid_partition.png", dpi=150)
            plt.close(fig)

    gp = gaps(rows)
    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.3 * len(gp) + 1.2)))
    ax.barh(range(len(gp)), [x[3] for x in gp], color=examplekit.SERIES[0])
    ax.set_yticks(range(len(gp)))
    ax.set_yticklabels([f"{c}  {k} parts  {t:g} s" for c, k, t, _ in gp], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median gap to the proven optimum (%)")
    fig.tight_layout()
    fig.savefig(out / "grid_gap.png", dpi=150)
    plt.close(fig)

    cells, won = vs_rival(rows)
    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.4 * len(won) + 1.2)))
    ys = list(range(len(won)))
    ax.scatter([c[3] for c in won], ys, color=examplekit.SERIES[1], label="annealer (penalty form)")
    ax.scatter([c[2] for c in won], ys, color=examplekit.SERIES[0], label="4colors (exact row)")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c[0]}  {c[1]:g} s" for c in won], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median cut (branches), lower is better")
    ax.legend()
    ax.set_title("bisection cells where we cut fewer branches (selection rule; full record linked)", color=examplekit.INK,
                 fontsize=10)
    print(f"grid_vs_sa: {len(won)} of {len(cells)} cells drawn (internal)")
    fig.tight_layout()
    fig.savefig(out / "grid_vs_sa.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
