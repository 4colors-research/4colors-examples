"""The quadrant chart and the selection file for the pre-flight post, from recorded results only.

    python plot.py RUN_API/out [RUN_QISKIT/out] --figs figs [--easy-s 1] [--esp-line 0.1]

Rows are joined per instance across records (the API record has the solve and the 4colors compile, the timed
record has Qiskit), so the verdict is recomputed here, never read from a half-filled verdict row.

fig_quadrants.png: one dot per instance at (service seconds to the final value, modeled ESP of the 4colors
compile), the two lines drawn. Where the 4colors compile has a strictly higher modeled ESP than Qiskit O3's at the
same budget, Qiskit's ESP is drawn as a hollow dot below it, joined by a thin line. figs/selection.md lists every
instance with its box.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import examplekit
import triage

RULE = ("Every instance is placed by its 4colors signals (a triage, not a contest). Qiskit O3's modeled ESP is "
        "drawn only where the 4colors compile's is strictly higher at the same budget; the full record "
        "is linked. ESP is a model from the published calibration, not a measurement; the "
        "x axis is service search time, not wall time.")


def join(rows):
    """{instance: {t_final_s, esp_4colors, esp_qiskit, family, n, is_optimal}}."""
    out = {}
    for r in rows:
        inst = r.get("instance")
        if inst is None:
            continue
        d = out.setdefault(inst, {"family": r.get("family"), "n": r.get("n")})
        if r.get("kind") == "solve":
            d["t_final_s"] = r.get("t_final_s")
            d["is_optimal"] = r.get("is_optimal")
        elif r.get("kind") == "result" and r.get("arm") == "4colors":
            d["esp_4colors"] = r.get("estimated_esp")
        elif r.get("kind") == "result" and r.get("arm") == "qiskit" and r.get("coupler_set") == "full":
            d["esp_qiskit"] = r.get("estimated_esp")
    return out


def selection(joined, easy_s=triage.EASY_S, esp_line=triage.ESP_LINE):
    """(boxes per instance, instances where Qiskit is drawn, instances with both compiles)."""
    boxes = {k: triage.verdict(d.get("t_final_s"), d.get("esp_4colors"), easy_s=easy_s, esp_line=esp_line)
             for k, d in joined.items()}
    both = sorted(k for k, d in joined.items() if d.get("esp_4colors") is not None and d.get("esp_qiskit") is not None)
    drawn = [k for k in both if joined[k]["esp_4colors"] > joined[k]["esp_qiskit"]]
    return boxes, drawn, both


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--figs", required=True)
    ap.add_argument("--easy-s", type=float, default=triage.EASY_S)
    ap.add_argument("--esp-line", type=float, default=triage.ESP_LINE)
    a = ap.parse_args(argv)
    joined = join(examplekit.load_rows(a.runs))
    boxes, drawn, both = selection(joined, a.easy_s, a.esp_line)
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    plt = examplekit.plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    floor = 1e-6
    for fam in sorted({d["family"] for d in joined.values()}):
        pts = [(max(d["t_final_s"], 1e-3), max(d["esp_4colors"], floor)) for d in joined.values()
               if d["family"] == fam and d.get("t_final_s") is not None and d.get("esp_4colors") is not None]
        if pts:
            ax.scatter(*zip(*pts), label=fam, zorder=3)
    for k in drawn:
        d = joined[k]
        if d.get("t_final_s") is None:
            continue
        x = max(d["t_final_s"], 1e-3)
        ax.plot([x, x], [max(d["esp_qiskit"], floor), max(d["esp_4colors"], floor)], color="#999999", lw=0.8, zorder=2)
        ax.scatter([x], [max(d["esp_qiskit"], floor)], facecolors="none", edgecolors="#666666", zorder=2)
    ax.axvline(a.easy_s, color="#444444", ls="--", lw=1)
    ax.axhline(a.esp_line, color="#444444", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("service search seconds to the final value"); ax.set_ylabel("modeled ESP (published calibration)")
    ax.legend(fontsize=8)
    fig.savefig(figs / "fig_quadrants.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    lines = ["# Selection", "", RULE, "",
             f"Internal (not for publication): Qiskit drawn on {len(drawn)} of {len(both)} instances with both compiles.",
             "", "| instance | box | Qiskit drawn |",
             "|---|---|---|"]
    lines += [f"| {k} | {boxes[k]} | {'yes' if k in drawn else 'no'} |" for k in sorted(boxes)]
    counts = {}
    for b in boxes.values():
        counts[b] = counts.get(b, 0) + 1
    lines += ["", "Boxes: " + ", ".join(f"{b}: {c}" for b, c in sorted(counts.items()))]
    (figs / "selection.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
