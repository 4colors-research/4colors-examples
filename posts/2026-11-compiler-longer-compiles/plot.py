"""Figures, tables and the selection note for the longer-compiles post.

    python plot.py RUN_4COLORS RUN_QISKIT_LONG RUN_QISKIT_RESEED --figs figs

Reads the results.jsonl of each record and writes:
  fig_plateau.png  (hero) when the 4colors compile stopped improving, per shown circuit, against the budget;
  fig_time.png     two-qubit gates against compile time, one panel per shown circuit;
  fig_esp.png      estimated success probability against compile time (a model, not a measurement);
  fig_qiskit_time.png  where Qiskit-long's core-seconds went, per member and budget;
  table.md, selection.md (the cells shown and left out, and the line under each chart).

Selection rule (fixed before any run): a (circuit, budget) cell is shown on a metric's chart
when the 4colors median over seeds beats the Qiskit reference median by more than the tie band (2 % on two-qubit
gates; on ESP, a higher median by more than 2 % relative), where the Qiskit reference is the better of the
Qiskit-long portfolio and its best single member in hindsight, Qiskit produced an answer within the budget on every
seed, every 4colors answer passed the service's check (verified, or approximate within the matched
approximation), no 4colors compile was refused or empty, and the plan floor did not give 4colors more time than Qiskit used.
It is a strict win when the 4colors worst seed also beats the Qiskit reference's best seed.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit  # noqa: E402
import plateau  # noqa: E402

TIE_BAND = 0.02
METRICS = {"2q": ("two_qubit_gates", "best_member_2q", -1), "esp": ("estimated_esp", "best_member_esp", +1)}


def _res(rows, **match):
    return [r for r in rows if r.get("kind") == "result" and all(r.get(k) == v for k, v in match.items())]


def verified(r: dict) -> bool:
    """The service's check passed: "verified", or "approximate" within the cell's matched approximation."""
    lab = r.get("verification_label")
    if lab == "verified":
        return True
    return lab == "approximate" and r.get("d_cell") is not None and (r.get("verification_distance") or 0) <= r["d_cell"]


def qiskit_reference(rows: List[dict], cid: str, budget: float, metric: str) -> Tuple[List[float], bool]:
    """Per seed, the better of the portfolio's answer and its best single member (by the chart's metric), and
    whether every seed produced an answer within the budget."""
    field, hind, sign = METRICS[metric]
    vals, all_ok = [], True
    q = _res(rows, arm="qiskit-long", circuit_id=cid, budget_s=budget, select=metric)
    for r in q:
        if r.get("status") != "finished" or r.get(field) is None:
            all_ok = False
            continue
        cands = [r[field]] + ([r[hind]] if r.get(hind) is not None else [])
        vals.append(max(cands) if sign > 0 else min(cands))
    return vals, all_ok and bool(q)


def cell(rows: List[dict], cid: str, budget: float, metric: str, tie_band: float = TIE_BAND) -> dict:
    field, _, sign = METRICS[metric]
    ours_rows = _res(rows, arm="4colors", circuit_id=cid, budget_s=budget)
    ours = [r.get(field) for r in ours_rows]
    qv, q_ok = qiskit_reference(rows, cid, budget, metric)
    c = {"circuit_id": cid, "budget_s": budget, "metric": metric, "ours": ours, "qiskit": qv,
         "ours_median": examplekit.median(ours), "qiskit_median": examplekit.median(qv), "win": False, "strict": False}
    if not ours_rows or any(r.get("refusal_code") or r.get(field) is None for r in ours_rows):
        return {**c, "why": "4colors refused or returned no circuit"}
    if not all(verified(r) for r in ours_rows):
        return {**c, "why": "a 4colors answer was not verified within the matched approximation"}
    if any(r.get("budget_floor_applied") for r in ours_rows):
        return {**c, "why": "plan floor: 4colors would get more time than Qiskit used"}
    if not q_ok:
        return {**c, "why": "Qiskit had no answer within the budget on some seed (over_budget, reported apart)"}
    om, km = c["ours_median"], c["qiskit_median"]
    if sign < 0:
        win = om < km * (1 - tie_band)
        strict = win and max(ours) < min(qv)
    else:
        win = om > km * (1 + tie_band)
        strict = win and min(ours) > max(qv)
    return {**c, "win": win, "strict": strict, "why": "win" if win else "tie or loss (kept in the record)"}


def select_cells(rows: List[dict], metric: str, tie_band: float = TIE_BAND) -> List[dict]:
    keys = sorted({(r["circuit_id"], r["budget_s"]) for r in _res(rows, arm="4colors")}, key=lambda k: (k[0], k[1]))
    return [cell(rows, cid, b, metric, tie_band) for cid, b in keys]


def stream_points(rows: List[dict], cid: str) -> List[Tuple[float, float]]:
    return [(r["t_client_s"], r["two_qubit_gates"]) for r in rows
            if r.get("kind") == "incumbent" and r.get("arm") == "4colors" and r.get("circuit_id") == cid
            and r.get("two_qubit_gates") is not None]


def qiskit_curve(rows: List[dict], cid: str, budget: float, key: str = "2q", seed: int = 1):
    return [(r["t_s"], r["two_qubit_gates"] if key == "2q" else r["estimated_esp"]) for r in rows
            if r.get("kind") == "incumbent" and r.get("arm") == "qiskit-long" and r.get("circuit_id") == cid
            and r.get("budget_s") == budget and r.get("select") == key and r.get("seed") == seed]


def plateau_table(rows: List[dict], shown: List[str]) -> List[dict]:
    out = []
    for cid in shown:
        pts = stream_points(rows, cid)
        top = max((r.get("t_cell") or 0 for r in rows if r.get("kind") == "incumbent" and r.get("circuit_id") == cid),
                  default=None)
        out.append({"circuit_id": cid, "last_improvement_s": plateau.last_improvement_s(pts),
                    "within_1pct_s": plateau.time_to_within(pts, 0.01), "budget_s": top,
                    "advice": plateau.advice(pts, top) if top else None})
    return out


def selection_note(cells: Dict[str, List[dict]]) -> str:
    lines = ["# Selection (see the post)", ""]
    for metric, cs in cells.items():
        won = [c for c in cs if c["win"]]
        lines += [f"## {metric}", "",
                  f"Under the chart: \"Circuits and budgets where 4colors' median beat the better of Qiskit-long and "
                  f"its best single member by more than {TIE_BAND:.0%}, at the same wall time and approximation, on "
                  "FakeFez's published calibration. The full record, every cell included, is linked from the run id.\"",
                  "", f"Internal (not for publication): cells shown {len(won)} of {len(cs)}.", "",
                  "| circuit | budget s | shown | strict | 4colors median | Qiskit median | why |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {c['circuit_id']} | {c['budget_s']:g} | {c['win']} | {c['strict']} | {c['ours_median']} | "
                  f"{c['qiskit_median']} | {c['why']} |" for c in cs]
        lines.append("")
    return "\n".join(lines)


def table_md(rows: List[dict], won: List[dict]) -> str:
    head = ("| circuit | qubits | T (s) | 4colors 2q | Qiskit-long 2q | Qiskit reseeded 2q | O3 once 2q | "
            "4colors ESP | Qiskit-long ESP |")
    lines = [head, "|" + "---|" * 9]
    for c in won:
        cid, b = c["circuit_id"], c["budget_s"]
        med = lambda arm, f, sel="2q": examplekit.median([r.get(f) for r in _res(rows, arm=arm, circuit_id=cid,  # noqa
                                                                                budget_s=b, select=sel)])
        once = examplekit.median([r.get("two_qubit_gates") for r in _res(rows, arm="qiskit-default", circuit_id=cid)])
        q = next((r["qubits"] for r in _res(rows, circuit_id=cid)), None)
        ours = _res(rows, arm="4colors", circuit_id=cid, budget_s=b)
        lines.append(f"| {cid} | {q} | {b:g} | {examplekit.median([r.get('two_qubit_gates') for r in ours])} | "
                     f"{med('qiskit-long', 'two_qubit_gates')} | {med('qiskit-reseed', 'two_qubit_gates')} | {once} | "
                     f"{examplekit.median([r.get('estimated_esp') for r in ours])} | "
                     f"{med('qiskit-long', 'estimated_esp', 'esp')} |")
    return "\n".join(lines)


def draw(rows: List[dict], cells: Dict[str, List[dict]], figs: str):
    plt = examplekit.plot_style()
    os.makedirs(figs, exist_ok=True)
    shown = sorted({c["circuit_id"] for c in cells["2q"] if c["win"]})
    if not shown:
        return []
    written = []
    # hero: when did the 4colors stream stop improving, against the budget it had
    tab = plateau_table(rows, shown)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(tab) + 1.2))
    for i, t in enumerate(tab):
        ax.plot([0.1, t["budget_s"] or 0.1], [i, i], color=examplekit.GRID, lw=6, solid_capstyle="round")
        if t["last_improvement_s"]:
            ax.plot([t["last_improvement_s"]], [i], "o", color=examplekit.SERIES[0])
        if t["within_1pct_s"]:
            ax.plot([t["within_1pct_s"]], [i], "|", ms=14, color=examplekit.SERIES[1])
    ax.set_xscale("log")
    ax.set_yticks(range(len(tab)), [t["circuit_id"] for t in tab])
    ax.set_xlabel("seconds into the compile (log)")
    ax.set_title("Within 1 % of the final count (|) and the last improvement (o), against the budget", loc="left",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "fig_plateau.png"), dpi=160)
    written.append("fig_plateau.png")
    plt.close(fig)
    for metric, name, ylab in (("2q", "fig_time.png", "two-qubit gates"),
                               ("esp", "fig_esp.png", "estimated success probability (model)")):
        won = [c for c in cells[metric] if c["win"]]
        ids = sorted({c["circuit_id"] for c in won})
        if not ids:
            continue
        fig, axes = plt.subplots(1, len(ids), figsize=(3.2 * len(ids), 3), squeeze=False)
        for ax, cid in zip(axes[0], ids):
            cs = [c for c in cells[metric] if c["circuit_id"] == cid]
            ax.plot([c["budget_s"] for c in cs], [c["ours_median"] for c in cs], "o-", label="4colors")
            ax.plot([c["budget_s"] for c in cs], [c["qiskit_median"] for c in cs], "s--", label="Qiskit-long")
            for c in cs:
                if c["strict"]:
                    ax.plot([c["budget_s"]], [c["ours_median"]], "*", ms=12, color=examplekit.SERIES[0])
            if metric == "2q":
                pts = stream_points(rows, cid)
                if pts:
                    steps = plateau.best_so_far(pts)
                    ax.step([t for t, _ in steps], [v for _, v in steps], where="post", lw=1, alpha=0.6)
            ax.set_xscale("log")
            if metric == "esp":
                ax.set_yscale("log")
            ax.set_title(cid, loc="left", fontsize=9)
            ax.set_xlabel("budget (s, log)")
        axes[0][0].set_ylabel(ylab)
        axes[0][0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figs, name), dpi=160)
        written.append(name)
        plt.close(fig)
    shares = [r for r in rows if r.get("kind") == "time_share" and r.get("circuit_id") in shown and r.get("seed") == 1]
    if shares:
        budgets = sorted({r["budget_s"] for r in shares})
        members = sorted({m for r in shares for m in r["core_s"]})
        fig, axes = plt.subplots(1, len(budgets), figsize=(3.2 * len(budgets), 0.4 * len(shown) + 1.5), squeeze=False)
        for ax, b in zip(axes[0], budgets):
            for i, cid in enumerate(shown):
                r = next((x for x in shares if x["circuit_id"] == cid and x["budget_s"] == b), None)
                if r is None:
                    continue
                tot = sum(r["core_s"].values()) or 1.0
                left = 0.0
                for k, m in enumerate(members):
                    w = r["core_s"].get(m, 0.0) / tot
                    ax.barh(i, w, left=left, color=examplekit.SERIES[k % len(examplekit.SERIES)], label=m if i == 0 else None)
                    left += w
            ax.set_yticks(range(len(shown)), shown if b == budgets[0] else [""] * len(shown))
            ax.set_title(f"{b:g} s", loc="left", fontsize=9)
            ax.set_xlim(0, 1)
        axes[0][-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(figs, "fig_qiskit_time.png"), dpi=160)
        written.append("fig_qiskit_time.png")
        plt.close(fig)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("records", nargs="+", help="run dirs (or results.jsonl files): 4colors, qiskit-long, qiskit-reseed")
    p.add_argument("--figs", required=True)
    p.add_argument("--tie-band", type=float, default=TIE_BAND)
    a = p.parse_args(argv)
    rows = examplekit.load_rows(a.records)
    cells = {m: select_cells(rows, m, a.tie_band) for m in METRICS}
    os.makedirs(a.figs, exist_ok=True)
    with open(os.path.join(a.figs, "selection.md"), "w") as fh:
        fh.write(selection_note(cells))
    with open(os.path.join(a.figs, "table.md"), "w") as fh:
        fh.write(table_md(rows, [c for c in cells["2q"] if c["win"]]))
    for name in draw(rows, cells, a.figs):
        print("wrote", os.path.join(a.figs, name))


if __name__ == "__main__":
    main()
