"""The MIMO post's charts, from results.jsonl only (plus one generated instance for the coupling picture).

    python plot.py runs/q32 runs/h16 --out figures [--budget 1]

- ber_<mod>_<nt>x<nr>.png: bit error rate against SNR. MMSE, the exact ML reference (sphere decoder, only at SNRs
  where it finished on every frame) and 4colors at one budget. Selection rule: a 4colors point is drawn only where
  its 95% Wilson interval lies wholly below MMSE's; a chart with no such point is not written.
- time_to_ml_<mod>_snr<s>.png: milliseconds to an ML-quality answer against the number of users. Sphere decoder:
  median local wall time, from TIMED rows only. 4colors: median search time until its best reached the ML residual
  (service trajectory). Drawn only at sizes where the 4colors median is lower; no chart without timed rows.
- sphere_nodes.png: median sphere-decoder nodes against users, per SNR (machine-independent; explains the above).
- couplings.png: the dense coupling matrix of one generated 16-QAM channel (illustration, not a result).
selection.json lists every cell considered and whether it was drawn, for the internal record (no counts in the post).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import examplekit
import mimo


def ber_cells(rows, budget):
    """{(mod, nt, nr): {snr: {arm: (errors, bits)}}} for mmse, sphere (complete frames only) and api at `budget`."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    sphere_ok = defaultdict(lambda: True)
    for r in rows:
        if "bit_errors" not in r:
            continue
        key, snr = (r["mod"], r["nt"], r["nr"]), r["snr_db"]
        arm = r["arm"]
        if arm == "api" and r.get("time_limit_s") != budget:
            continue
        if arm == "sphere" and not r.get("complete"):
            sphere_ok[(key, snr)] = False
        if arm in ("mmse", "sphere", "api"):
            acc[key][snr][arm][0] += r["bit_errors"]
            acc[key][snr][arm][1] += r["n_bits"]
    for (key, snr), ok in sphere_ok.items():
        if not ok:
            acc[key][snr].pop("sphere", None)
    return acc


def api_wins(cell) -> bool:
    if "api" not in cell or "mmse" not in cell:
        return False
    lo_m, _ = mimo.wilson(*cell["mmse"])
    _, hi_a = mimo.wilson(*cell["api"])
    return hi_a < lo_m


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=float, default=1.0, help="the 4colors time limit charted (seconds)")
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    rows = examplekit.load_rows(a.inputs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = {"budget": a.budget, "ber": [], "time_to_ml": []}
    written = []

    for key, by_snr in sorted(ber_cells(rows, a.budget).items()):
        mod, nt, nr = key
        snrs = sorted(by_snr)
        wins = [s for s in snrs if api_wins(by_snr[s])]
        for s in snrs:
            sel["ber"].append({"mod": mod, "nt": nt, "nr": nr, "snr_db": s, "drawn": s in wins})
        if not wins:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        rate = lambda c: max(c[0], 0.5) / c[1]          # a zero-error point is drawn at half an error
        ax.plot(snrs, [rate(by_snr[s]["mmse"]) for s in snrs], marker="s", color=examplekit.SERIES[1], label="MMSE")
        ml = [s for s in snrs if "sphere" in by_snr[s]]
        if ml:
            ax.plot(ml, [rate(by_snr[s]["sphere"]) for s in ml], linestyle="--", color=examplekit.INK_2,
                    label="exact ML (sphere decoder, reference)")
        ax.plot(wins, [rate(by_snr[s]["api"]) for s in wins], marker="o", linestyle="none",
                color=examplekit.SERIES[0], label=f"4colors, {a.budget:g} s per job")
        ax.set_yscale("log")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("bit error rate")
        ax.set_title(f"{mod.upper()}, {nt} users x {nr} antennas", loc="left")
        ax.legend()
        f = out / f"ber_{mod}_{nt}x{nr}.png"
        fig.savefig(f, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(f.name)

    # time to an ML-quality answer
    sd_ms, api_ms, nodes = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        k = (r.get("mod"), r.get("snr_db"))
        if r.get("arm") == "sphere":
            if r.get("nodes") is not None:
                nodes[k + (r["nt"],)].append(r["nodes"])
            if r.get("timed") and r.get("complete"):
                sd_ms[k + (r["nt"],)].append(1000 * r["wall_s"])
        elif r.get("arm") == "api" and not r.get("pack") and r.get("t_to_ml_s") is not None and r.get("block") == 0:
            api_ms[k + (r["nt"],)].append(1000 * r["t_to_ml_s"])
    groups = sorted({k[:2] for k in sd_ms})
    for mod, snr in groups:
        nts = sorted(nt for (m_, s_, nt) in sd_ms if (m_, s_) == (mod, snr))
        win = [nt for nt in nts if api_ms.get((mod, snr, nt))
               and examplekit.median(api_ms[(mod, snr, nt)]) < examplekit.median(sd_ms[(mod, snr, nt)])]
        for nt in nts:
            sel["time_to_ml"].append({"mod": mod, "snr_db": snr, "nt": nt, "drawn": nt in win})
        if not win:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(win, [examplekit.median(sd_ms[(mod, snr, nt)]) for nt in win], marker="s", color=examplekit.SERIES[1],
                label="sphere decoder (exact ML), median local wall")
        ax.plot(win, [examplekit.median(api_ms[(mod, snr, nt)]) for nt in win], marker="o", color=examplekit.SERIES[0],
                label="4colors, median search time to the ML answer")
        ax.set_yscale("log")
        ax.set_xlabel("users (square system)")
        ax.set_ylabel("milliseconds")
        ax.set_title(f"{mod.upper()} at {snr:g} dB", loc="left")
        ax.legend()
        f = out / f"time_to_ml_{mod}_snr{snr:g}.png"
        fig.savefig(f, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(f.name)

    if nodes:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (mod, snr) in enumerate(sorted({k[:2] for k in nodes})):
            nts = sorted(nt for (m_, s_, nt) in nodes if (m_, s_) == (mod, snr))
            ax.plot(nts, [examplekit.median(nodes[(mod, snr, nt)]) for nt in nts], marker=examplekit.MARKERS[i % 8],
                    color=examplekit.SERIES[i % 8], label=f"{mod.upper()}, {snr:g} dB")
        ax.set_yscale("log")
        ax.set_xlabel("users (square system)")
        ax.set_ylabel("sphere-decoder tree nodes (median)")
        ax.legend()
        f = out / "sphere_nodes.png"
        fig.savefig(f, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(f.name)

    inst = mimo.generate(8, 8, "16qam", 15, 1)
    quad, _, _ = mimo.ising_terms(inst)
    import numpy as np
    J = np.zeros((inst.n_spins, inst.n_spins))
    for (i, j), v in quad.items():
        J[i, j] = J[j, i] = v
    fig, ax = plt.subplots(figsize=(4.5, 4))
    lim = float(np.abs(J).max())
    im = ax.imshow(J, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.grid(False)
    ax.set_title("couplings, 8 x 8 16-QAM (32 spins)", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    f = out / "couplings.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(f.name)

    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    for w in written:
        print(out / w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
