"""Figures for the qLDPC post, from the record only (the Tanner-graph drawing is built from the code itself).

    python plot.py RUN/out --figs RUN/figs

Selection rule (owner 22 Nov: wins only, rule stated under each chart):
* distance.png: codes whose least certified weight equals the distance the code's paper reports, at the shortest
  budget that reached it;
* heavy_hex.png: codes where the 4colors round on heavy-hex has fewer two-qubit gates than Qiskit's (median over
  seeds, same budget) and every shown circuit of both arms passed the syndrome check;
* tanner.png: the code's own chip (no comparison).
Everything else is listed in selection.md; nothing is dropped from the record.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit
import codes as cd


def select(rows):
    dist, hh, notes = [], [], []
    bounds = {}
    for r in rows:
        if r.get("kind") == "distance_bound":
            bounds.setdefault(r["code"], []).append(r)
    for code, rs in sorted(bounds.items()):
        hit = [r for r in sorted(rs, key=lambda r: r["time_limit_s"])
               if r["upper_bound"] is not None and r["upper_bound"] == r["reported_distance"]]
        if hit:
            dist.append((code, hit[0]["upper_bound"], hit[0]["time_limit_s"]))
        else:
            notes.append(f"{code}: no certified weight equal to the reported distance "
                         f"({[(r['time_limit_s'], r['upper_bound']) for r in rs]})")
    res = [r for r in rows if r.get("kind") == "result"]
    for code in sorted({r["code"] for r in res}):
        ours = [r for r in res if r["code"] == code and r["chip"] == "heavy_hex" and r["arm"] == "4colors"]
        theirs = [r for r in res if r["code"] == code and r["chip"] == "heavy_hex" and r["arm"] == "qiskit"]
        tan = [r for r in res if r["code"] == code and r["chip"] == "tanner" and r["arm"] == "4colors"]
        checks = all(r.get("syndrome_check") == "pass" for r in ours + theirs)
        mo = examplekit.median([r.get("two_qubit_gates") for r in ours])
        mt = examplekit.median([r.get("two_qubit_gates") for r in theirs])
        if mo is not None and mt is not None and mo < mt and checks:
            hh.append({"code": code, "input": ours[0]["input_2q"], "ours": mo, "qiskit": mt,
                       "tanner": examplekit.median([r.get("two_qubit_gates") for r in tan])})
        else:
            notes.append(f"{code}: heavy-hex not shown (ours {mo}, Qiskit {mt}, checks passed {checks})")
    return dist, hh, notes


def tanner_positions(l: int, m: int):
    """Torus layout of a BB code: index i = a * m + b; data L at (2a, 2b), data R at (2a+1, 2b+1), X check at
    (2a+1, 2b), Z check at (2a, 2b+1)."""
    n = l * m
    pos = {}
    for a in range(l):
        for b in range(m):
            i = a * m + b
            pos[i] = (2 * a, 2 * b)
            pos[n + i] = (2 * a + 1, 2 * b + 1)
            pos[2 * n + i] = (2 * a + 1, 2 * b)
            pos[3 * n + i] = (2 * a, 2 * b + 1)
    return pos


def draw_tanner(ax, name: str):
    l, m = cd.BB[name][0], cd.BB[name][1]
    hx, hz, _ = cd.code(name)
    pos = tanner_positions(l, m)
    for a, b in cd.tanner_couplers(hx, hz):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        far = abs(xa - xb) > 1 or abs(ya - yb) > 1
        ax.plot([xa, xb], [ya, yb], lw=0.6 if far else 1.2, alpha=0.35 if far else 0.9,
                color=examplekit.SERIES[1] if far else examplekit.SERIES[0])
    roles = cd.qubit_roles(hx, hz)
    for key, marker in (("data", "o"), ("x_checks", "s"), ("z_checks", "D")):
        xs = [pos[q][0] for q in roles[key]]
        ys = [pos[q][1] for q in roles[key]]
        ax.scatter(xs, ys, s=14, marker=marker, label=key.replace("_", " "), zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=7)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", nargs="+")
    p.add_argument("--figs", required=True)
    p.add_argument("--tanner-code", default="bb_72_12_6")
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    rows = examplekit.load_rows(a.run)
    dist, hh, notes = select(rows)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(range(len(dist)), [d for _, d, _ in dist], 0.5, label="least certified logical weight")
    ax.scatter(range(len(dist)), [cd.BB.get(c, (0, 0, 0, 0, None))[4] or d for c, d, _ in dist], color=examplekit.INK,
               marker="_", s=400, label="distance reported by the code's paper", zorder=3)
    ax.set_xticks(range(len(dist)), [f"{c}\n{t:g} s" for c, _, t in dist])
    ax.set_ylabel("weight")
    ax.legend(fontsize=8)
    ax.set_title("A certified logical at the reported distance", loc="left")
    fig.text(0.01, 0.01, "Upper bounds, each certified in plain Python. Shown: codes where the bound reached the "
                         "reported distance.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figs / "distance.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    labels = ["input round", "own chip (4colors)", "heavy-hex, Qiskit O3", "heavy-hex, 4colors"]
    for i, h in enumerate(hh):
        vals = [h["input"], h["tanner"], h["qiskit"], h["ours"]]
        ax.bar([i * 5 + j for j in range(4)], vals, 0.8, color=examplekit.SERIES[:4],
               label=labels if i == 0 else None)
    if hh:
        ax.legend(ax.patches[:4], labels, fontsize=8)
    ax.set_xticks([i * 5 + 1.5 for i in range(len(hh))], [h["code"] for h in hh])
    ax.set_ylabel("two-qubit gates, one syndrome round")
    ax.set_title("The routing price of not having the code's couplers", loc="left")
    fig.text(0.01, 0.01, "Medians over seeds, same budget, every circuit syndrome-checked. Shown: codes where 4colors "
                         "has fewer two-qubit gates on heavy-hex.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figs / "heavy_hex.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_tanner(ax, a.tanner_code)
    ax.set_title(f"{a.tanner_code}: the chip the code wants", loc="left")
    fig.tight_layout()
    fig.savefig(figs / "tanner.png", dpi=150)

    (figs / "selection.md").write_text("# Selection\n\n" + f"distance.png: {dist}\n\nheavy_hex.png: "
                                       f"{[h['code'] for h in hh]}\n\nNot shown:\n\n"
                                       + "\n".join(f"- {x}" for x in notes) + "\n")
    print(f"figures in {figs}")


if __name__ == "__main__":
    main()
