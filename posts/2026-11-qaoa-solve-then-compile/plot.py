"""Figures and tables for "solve it, then compile it", from OUT/results.jsonl only.

Usage: python plot.py --out runs/solve [--e2 3e-3]
Writes optimizer.png, ratio_vs_p.png, breakeven.png + breakeven.md, cost.md, selection.md; and, only if the
results hold rows of the later panels, density.png + density.md and warm.md (not part of this post).

Selection rules (owner, 22 Nov 2026: show the cells where ours is ahead, state the rule under the chart):
  ratio_vs_p   the compiler lines show the depths p where the 4colors circuit's noisy ratio (median over compile
               seeds, mean over graphs, optimized angles, at the chart's e2) is strictly above Qiskit O3's. The
               ideal line, the optimizer's line and the random line are not selected: they are not a comparison
               of compilers.
  breakeven    a (p, angles, S) cell is shown when the 4colors threshold is strictly above Qiskit O3's (a cell
               where Qiskit never reaches 1/2 and 4colors does counts as ahead).
Every left-out cell is listed in selection.md with the reason.

Break-even threshold (model-derived, never a hardware claim): the largest e2 of the sweep at which S shots of the
compiled circuit reach the optimizer's answer (at its reference time limit) with probability at least 1/2, read off
by log-linear interpolation between sweep points; next to it the same probability for S uniformly random bit strings.
"""
import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
from qaoakit import apikit as ak  # noqa: E402

med = ak.median


def fmt(x, nd=4):
    return "" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))


def threshold(points, level=0.5):
    """Largest e2 with success >= level, interpolating log(e2) between neighbors; None if never reached."""
    pts = sorted(points)
    best = None
    for (e_a, s_a), (e_b, s_b) in zip(pts, pts[1:]):
        if s_a >= level > s_b:
            f = (s_a - level) / (s_a - s_b)
            best = math.exp(math.log(e_a) + f * (math.log(e_b) - math.log(e_a)))
    if pts and pts[-1][1] >= level:
        best = pts[-1][0]
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--e2", type=float, default=None)
    ap.add_argument("--timed", action="store_true", help="the record is timed: also tabulate angle-optimizer CPU s")
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows = ak.read_jsonl(out / "results.jsonl")
    note = "Simulated, noise model as stated; no hardware."
    sel = ["# Selection", ""]

    # optimizer: ratio against time limit
    opt_rows = [r for r in rows if r["kind"] == "optimizer" and r.get("cut") is not None]
    best_found = {}
    for r in opt_rows:
        best_found[r["instance"]] = max(best_found.get(r["instance"], -1e300), r["cut"])
    fig, ax = ak.figure(figsize=(6, 3.4))
    ak.axes_style(ax)
    for n in sorted({r["n"] for r in opt_rows}):
        lims = sorted({r["time_limit_s"] for r in opt_rows if r["n"] == n})
        ys = [med([r["cut"] / (r["exact_optimum"] or best_found[r["instance"]]) for r in opt_rows
                   if r["n"] == n and r["time_limit_s"] == t]) for t in lims]
        ax.plot(lims, ys, "-o", linewidth=2, markersize=5, color=ak.ARM_COLORS["4colors-qubo"],
                alpha=0.4 + 0.6 * n / max(r["n"] for r in opt_rows),
                label=f"n = {n}" + ("" if any(r["n"] == n and r["exact_optimum"] for r in opt_rows) else " (best found)"))
    ax.set_xscale("log")
    ax.set_xlabel("time limit, seconds", fontsize=9)
    ax.set_ylabel("cut / optimum (median)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "optimizer.png", dpi=200)

    q = [r for r in rows if r["kind"] == "qaoa" and r.get("simulated")]
    e2s = sorted({r["e2"] for r in q})
    e2 = a.e2 if a.e2 is not None else (e2s[len(e2s) // 2] if e2s else None)

    # ratio against depth p (the J1 panel): ideal rises with p, noisy turns down, a better compile moves the turn
    by_arm = {arm: ratio_by_p(q, arm, e2) for arm in ARMS}
    shown = select_ps(by_arm)
    sel += [f"ratio_vs_p (e2 = {e2:g}): depths shown {shown}; left out (4colors not strictly ahead): "
            f"{sorted(set(by_arm['4colors']) - set(shown))}", ""] if e2 is not None else []
    fig, ax = ak.figure(figsize=(6.4, 4.0))
    ak.axes_style(ax)
    ideal = ideal_by_p(q)
    if ideal:
        ax.plot(list(ideal), list(ideal.values()), "-o", linewidth=2, color=ak.ARM_COLORS["ideal"],
                label="QAOA, ideal (noiseless simulation)")
    for arm in ARMS:
        pts = [(p, by_arm[arm][p]) for p in shown]
        if pts:
            ax.plot(*zip(*pts), "-o", linewidth=2, color=ak.ARM_COLORS[arm],
                    label=f"QAOA + noise, compiled by {ak.ARM_LABELS[arm]}")
    cls = [r["classical_ratio"] for r in q if r.get("classical_ratio") is not None]
    if cls:
        ax.axhline(med(cls), color=ak.ARM_COLORS["4colors-qubo"], linewidth=1.5)
        ax.text(max(ideal or [1]), med(cls), "4colors Optimizer for QUBO ", fontsize=7, ha="right", va="bottom")
    rnd = [r["random_ratio"] for r in q if r.get("random_ratio") is not None]
    if rnd:
        ax.axhline(med(rnd), color=ak.MUTED, linestyle=":", linewidth=1.2)
        ax.text(max(ideal or [1]), med(rnd), "random cut ", fontsize=7, ha="right", va="bottom")
    ax.set_xlabel("QAOA depth p", fontsize=9)
    ax.set_ylabel(f"E[cut] / max cut (noise: e2 = {e2:g})" if e2 else "E[cut] / max cut", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "ratio_vs_p.png", dpi=200)

    # break-even, both shot counts
    shot_list = sorted({int(k.split("_")[1]) for r in q for k in r if k.startswith("success_")})
    lines = [f"Break-even (model-derived): the largest e2 at which S shots reach the optimizer's answer (a cut at "
             f"least as large as the optimizer's at its reference limit) with probability >= 1/2. {note}", "",
             "| p | angles | S | arm | median 2q gates | e2 threshold | S random bit strings |", "|---|---|---|---|---|---|---|"]
    cells = breakeven_cells(q, shot_list)
    fig, axes = ak.figure(1, max(1, len(shot_list)), figsize=(4.2 * max(1, len(shot_list)), 3.4), squeeze=False)
    for c in cells:
        if not c["shown"]:
            sel.append(f"breakeven p={c['p']} {c['angles']} S={c['S']}: left out (4colors {fmt(c['4colors'])}, "
                       f"Qiskit O3 {fmt(c['qiskit-o3'])})")
            continue
        for arm in ARMS:
            lines.append(f"| {c['p']} | {c['angles']} | {c['S']} | {ak.ARM_LABELS[arm]} | {c['gates'][arm]} | "
                         f"{fmt(c[arm])} | {fmt(c['random'])} |")
    for k, S in enumerate(shot_list):
        ax = axes[0][k]
        ak.axes_style(ax)
        show_p1 = any(c["shown"] and c["S"] == S and c["p"] == 1 and c["angles"] == "optimized" for c in cells)
        for arm in ARMS if show_p1 else ():
            mine = [r for r in q if r["arm"] == arm and r["p"] == 1 and r["angles"] == "optimized"
                    and r.get(f"success_{S}") is not None]
            es = sorted({r["e2"] for r in mine})
            ax.plot(es, [med([r[f"success_{S}"] for r in mine if r["e2"] == e]) for e in es], "-o", linewidth=2,
                    color=ak.ARM_COLORS[arm], label=f"QAOA p = 1, {ak.ARM_LABELS[arm]}")
        rs = [r.get(f"random_success_{S}") for r in q if r.get(f"random_success_{S}") is not None]
        if rs:
            ax.axhline(med(rs), color=ak.MUTED, linestyle=":", linewidth=1.5, label=f"{S} random bit strings")
        ax.axhline(0.5, color=ak.GRID, linewidth=1)
        ax.set_xscale("log")
        ax.set_title(f"S = {S} shots", fontsize=9, loc="left")
        ax.set_xlabel("two-qubit error e2 (model)", fontsize=9)
        ax.set_ylabel("P(S shots reach the optimizer's answer)", fontsize=7)
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "breakeven.png", dpi=200)
    (out / "breakeven.md").write_text("\n".join(lines) + "\n")

    # cost, in units that do not mix: angle-optimizer evaluations, billed compile seconds, Qiskit seeds tried
    cl = ["Classical cost per QAOA point, kept in separate units (service seconds are billed seconds; local CPU "
          "seconds only from a timed record).", "",
          "| p | angles | median angle-optimizer evaluations | 4colors median billed compile s | Qiskit O3 median "
          "seeds tried | optimizer billed s at the reference limit |" + (" median angle-optimizer CPU s |" if a.timed else ""),
          "|---|---|---|---|---|---|" + ("---|" if a.timed else "")]
    for p in sorted({r["p"] for r in q}):
        for ang in ("optimized", "fixed"):
            mine = [r for r in q if r["p"] == p and r["angles"] == ang]
            if not mine:
                continue
            cl.append(f"| {p} | {ang} | {med([r['tune_evals'] for r in mine])} | "
                      f"{fmt(med([r.get('billed_s') for r in mine if r['arm'] == '4colors']))} | "
                      f"{med([r.get('tries') for r in mine if r['arm'] == 'qiskit-o3'])} | "
                      f"{fmt(med([r.get('classical_billed_s') for r in mine]))} |"
                      + (f" {fmt(med([r['tune_cpu_s'] for r in mine]))} |" if a.timed else ""))
    (out / "cost.md").write_text("\n".join(cl) + "\n")
    (out / "selection.md").write_text("\n".join(sel) + "\n")

    # ---- later panels (not part of this post): only when their rows exist
    dc = [r for r in rows if r["kind"] == "density_compile"]
    dq = [r for r in rows if r["kind"] == "density_qubo"]
    if dc:
        density_panel(out, dc, dq)
    wq = [r for r in rows if r["kind"] == "warm_qaoa" and r.get("simulated")]
    if wq:
        warm_panel(out, rows, wq, note)


ARMS = ("qiskit-o3", "4colors")


def ideal_by_p(q):
    """{p: mean over graphs of the ideal ratio}, optimized angles."""
    out = {}
    for p in sorted({r["p"] for r in q if r["angles"] == "optimized"}):
        per = {}
        for r in q:
            if r["p"] == p and r["angles"] == "optimized":
                per[r["instance"]] = r["ideal_ratio"]
        out[p] = sum(per.values()) / len(per)
    return out


def ratio_by_p(q, arm, e2):
    """{p: mean over graphs of the median over compile seeds of the noisy ratio}, optimized angles, at e2."""
    out = {}
    for p in sorted({r["p"] for r in q}):
        per = {}
        for r in q:
            if r["arm"] == arm and r["p"] == p and r["angles"] == "optimized" and r["e2"] == e2:
                per.setdefault(r["instance"], []).append(r["noisy_ratio_mean"])
        if per:
            out[p] = sum(med(v) for v in per.values()) / len(per)
    return out


def select_ps(by_arm):
    """Depths where 4colors' noisy ratio is strictly above Qiskit O3's (both present)."""
    f, k = by_arm.get("4colors", {}), by_arm.get("qiskit-o3", {})
    return [p for p in sorted(f) if p in k and f[p] > k[p]]


def breakeven_cells(q, shot_list):
    """One cell per (p, angles, S): each arm's threshold from its median success curve over graphs and seeds,
    the median two-qubit gates, the random sampler, and `shown` (4colors strictly ahead; None = never reached)."""
    cells = []
    for p in sorted({r["p"] for r in q}):
        for ang in ("optimized", "fixed"):
            for S in shot_list:
                key = f"success_{S}"
                c = dict(p=p, angles=ang, S=S, gates={}, random=med([r.get("random_" + key) for r in q
                                                                     if r["p"] == p and r["angles"] == ang]))
                for arm in ARMS:
                    mine = [r for r in q if r["p"] == p and r["angles"] == ang and r["arm"] == arm
                            and r.get(key) is not None]
                    pts = [(e, med([r[key] for r in mine if r["e2"] == e])) for e in sorted({r["e2"] for r in mine})]
                    c[arm] = threshold(pts) if pts else None
                    c["gates"][arm] = med([r["two_qubit_gates"] for r in mine]) if mine else None
                if c["gates"]["4colors"] is None and c["gates"]["qiskit-o3"] is None:
                    continue
                f, k = c["4colors"], c["qiskit-o3"]
                c["shown"] = f is not None and (k is None or f > k)
                cells.append(c)
    return cells


def density_panel(out, dc, dq):
    fig, ax = ak.figure(figsize=(6, 3.4))
    ak.axes_style(ax)
    dl = ["| n | p | edges | arm | median 2q gates | refused | SWAP-network reference | optimizer median time to best s |",
          "|---|---|---|---|---|---|---|---|"]
    for n in sorted({r["n"] for r in dc}):
        for arm in ("qiskit-o3", "4colors"):
            pts = []
            for dens in sorted({r["density"] for r in dc if r["n"] == n}):
                mine = [r for r in dc if r["n"] == n and r["density"] == dens and r["arm"] == arm]
                ok = [r["two_qubit_gates"] for r in mine if r.get("two_qubit_gates") is not None]
                ttb = med([r.get("time_to_best_s") for r in dq if r["n"] == n and r["density"] == dens])
                dl.append(f"| {n} | {dens:g} | {mine[0]['n_edges']} | {arm} | {med(ok)} | {len(mine) - len(ok)} | "
                          f"{mine[0]['swap_network_two_qubit_gates']} | {fmt(ttb)} |")
                if ok:
                    pts.append((mine[0]["n_edges"], med(ok)))
            if pts:
                ax.plot(*zip(*pts), "-o", linewidth=2, markersize=5, color=ak.ARM_COLORS[arm],
                        label=f"{ak.ARM_LABELS[arm]}, n = {n}")
    ax.set_xlabel("edges", fontsize=9)
    ax.set_ylabel("compiled two-qubit gates, QAOA p = 1", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "density.png", dpi=200)
    (out / "density.md").write_text("\n".join(dl) + "\n")


def warm_panel(out, rows, wq, note):
    wc = [r for r in rows if r["kind"] == "warm_classical"]
    wl = [f"Warm-start QAOA against its own seed and against more classical search at matched compute. {note}", "",
          "| instance | seed source | seed optimal | seed ratio | arm | QAOA best-of-S ratio | P(S shots beat the seed) | "
          "optimizer at matched seconds, ratio |", "|---|---|---|---|---|---|---|---|"]
    for inst in dict.fromkeys(r["instance"] for r in wq):
        for src in dict.fromkeys(r["seed_source"] for r in wq if r["instance"] == inst):
            for arm in ("qiskit-o3", "4colors"):
                mine = [r for r in wq if r["instance"] == inst and r["seed_source"] == src and r["arm"] == arm]
                if not mine:
                    continue
                bk_ = next(k for k in mine[0] if k.startswith("best_of_") and k.endswith("_ratio"))
                sk = next(k for k in mine[0] if k.startswith("success_"))
                cl = [r["classical_ratio"] for r in wc if r["instance"] == inst and r["seed_source"] == src
                      and r["matched_to_arm"] == arm and r.get("classical_ratio") is not None]
                wl.append(f"| {inst} | {src} | {mine[0]['seed_is_optimal']} | {fmt(mine[0]['seed_ratio'])} | {arm} | "
                          f"{fmt(med([r[bk_] for r in mine]))} | {fmt(med([r[sk] for r in mine]))} | {fmt(med(cl))} |")
    (out / "warm.md").write_text("\n".join(wl) + "\n")


if __name__ == "__main__":
    main()
