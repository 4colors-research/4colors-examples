"""Inspect one cell (one circuit on one device snapshot, every arm's stored output) side by side.

Reads only the files run.py and check.py wrote; compiles nothing. Writes <out>/<device>/<circuit>/report.json and
the SVG figures (device heat map with the busiest logical qubit's route, two-qubit-layer timeline) per arm.

    python inspect_cell.py out/<device>/<circuit> --out inspect

Per arm, every seed's output is counted by sbs.py (one counting function for every arm) and compared with the
counts the run recorded (`agrees_with_record`: a counting difference cannot hide). The figures use the arm's median
output by two-qubit gates (the lower of the two middle ones for an even count), and every arm of the cell shares
one heat-map scale and one timeline scale.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sbs  # noqa: E402

LABEL = {"qiskit_o2": "Qiskit, optimization level 2", "qiskit_o3": "Qiskit, optimization level 3",
         "tket_tk2": "TKET, optimization level 2", "ours_inst": "4colors, no time budget"}
TIMELINE_W, TIMELINE_H = 820.0, 300.0     # target plot size in px for the longest arm / the most qubits


def label(arm):
    if arm in LABEL:
        return LABEL[arm]
    if arm.startswith("qiskit_o3best_T"):
        return f"Qiskit level 3, best seed in {arm[15:]} s"
    if arm.startswith("ours_T"):
        return f"4colors, {arm[6:]} s budget"
    return arm


def shared_scale(n_layers, n_rows):
    """One px-per-layer and px-per-qubit for every arm of a cell, so timelines compare directly."""
    col_w = min(8.0, max(1.2, TIMELINE_W / max(1, n_layers)))
    row_h = min(8.0, max(3.0, TIMELINE_H / max(1, n_rows)))
    return round(col_w, 2), round(row_h, 2)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def jl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def median_row(rows):
    s = sorted(rows, key=lambda r: (r["metrics"]["two_qubit_gates"], r["stem"]))
    return s[(len(s) - 1) // 2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-svg", action="store_true")
    ap.add_argument("--records", nargs="*", default=[],
                    help="more device dirs (other records of the same snapshot) whose rows and outputs are merged")
    ap.add_argument("--checks", nargs="*", default=[], help="more checks.jsonl files")
    a = ap.parse_args(argv)
    cell = os.path.abspath(a.cell)
    dev_dir = os.path.dirname(cell)
    device, cid = os.path.basename(dev_dir), os.path.basename(cell)
    cal = json.load(open(os.path.join(dev_dir, "cal.json")))
    tgt = json.load(open(os.path.join(dev_dir, "target.json")))
    inp = sbs.load(os.path.join(cell, "in.qasm"))
    in_sha = sha(os.path.join(cell, "in.qasm"))
    recorded, cell_of = [], {}
    for d in [dev_dir] + [os.path.abspath(x) for x in a.records]:
        c = os.path.join(d, cid)
        if not os.path.exists(os.path.join(c, "in.qasm")):
            continue
        if sha(os.path.join(c, "in.qasm")) != in_sha:
            raise SystemExit(f"{c}/in.qasm differs from {cell}/in.qasm: not the same input")
        for r in jl(os.path.join(d, "rivals.jsonl")) + jl(os.path.join(d, "ours.jsonl")):
            if r.get("circuit") == cid and r.get("stem"):
                recorded.append(r)
                cell_of[id(r)] = c
    cfiles = [os.path.join(dev_dir, "checks.jsonl")] + a.checks
    checks = {r["stem"]: r for f in cfiles for r in jl(f) if r["circuit"] == cid}
    hub = sbs.busiest_logical(inp)
    out_dir = os.path.join(a.out, device, cid)
    os.makedirs(out_dir, exist_ok=True)
    report = {"device": device, "circuit": cid, "input": sbs.metrics(inp),
              "input_sha256": in_sha, "hub_logical": hub, "arms": {}}
    per_arm = {}
    for rec in recorded:
        q = os.path.join(cell_of[id(rec)], rec["stem"] + ".qasm")
        lj = os.path.join(cell_of[id(rec)], rec["stem"] + ".layouts.json")
        circ = sbs.load(q)
        rep = sbs.inspect(inp, circ, layouts_json=json.load(open(lj)) if os.path.exists(lj) else None, cal=cal)
        rep.update(stem=rec["stem"], seed=rec.get("seed"), qasm_sha256=sha(q), wall_s=rec.get("wall_s"),
                   budget_s=rec.get("budget_s"), billed_s=rec.get("billed_s"), timings=rec.get("timings"),
                   options=rec.get("options"), engine_version=rec.get("engine_version"),
                   check=checks.get(rec["stem"]) if (checks.get(rec["stem"]) or {}).get("qasm_sha256") == sha(q)
                   else None)
        m = rep["metrics"]
        rep["agrees_with_record"] = all(rec.get(k) in (None, m[k]) for k in ("two_qubit_gates", "two_qubit_depth"))
        per_arm.setdefault(rec["arm"], []).append((circ, rep))
    vmax = max([1] + [v for lst in per_arm.values() for c, _ in lst for v in sbs.coupler_counts(c).values()])
    shown = {arm: median_row([r for _, r in lst]) for arm, lst in per_arm.items()}
    circ_of = {r["stem"]: c for lst in per_arm.values() for c, r in lst}
    n_layers = max([r["metrics"]["two_qubit_depth"] for r in shown.values()] or [1])
    n_rows = max([len({q for op in circ_of[r["stem"]].ops if sbs.is_2q(op) for q in op.qubits})
                  for r in shown.values()] or [1])
    col_w, row_h = shared_scale(n_layers, n_rows)
    report["figure_scale"] = {"n_layers": n_layers, "n_rows": n_rows, "px_per_layer": col_w, "px_per_qubit": row_h,
                              "heat_vmax": vmax}
    ring_label = "oracle qubit, start" if cid.startswith("bv") else "busiest logical qubit, start"
    for arm, lst in per_arm.items():
        seeds = []
        for circ, rep in lst:
            bl = rep.pop("_blocks")
            paths = rep.pop("_paths", None)
            if rep["stem"] == shown[arm]["stem"]:
                if paths and hub in paths:
                    rep["hub_path"] = paths[hub]
                if not a.no_svg:
                    init = rep.get("initial_layout") or []
                    hub_path = [p for _, p in paths[hub]] if paths and hub in paths else []
                    ring = [init[hub]] if len(init) > hub else []
                    m = rep["metrics"]
                    with open(os.path.join(out_dir, f"heat-{arm}.svg"), "w") as f:
                        f.write(sbs.device_svg(tgt["n_qubits"], tgt["couplers"], sbs.coupler_counts(circ),
                                               title=label(arm), subtitle=f"{m['two_qubit_gates']} two-qubit gates "
                                               f"on {rep['couplers_used']} couplers", footer=f"{cid}, {device}",
                                               marked=init, ring=ring, vmax=vmax, path=hub_path,
                                               ring_label=ring_label))
                    with open(os.path.join(out_dir, f"timeline-{arm}.svg"), "w") as f:
                        f.write(sbs.timeline_svg(circ, bl, title=label(arm),
                                                 subtitle=f"{cid} on {device}: {m['two_qubit_gates']} two-qubit "
                                                          f"gates, {m['two_qubit_depth']} two-qubit layers",
                                                 col_w=col_w, row_h=row_h, n_layers=n_layers, n_rows=n_rows))
            seeds.append(rep)
        report["arms"][arm] = {"label": label(arm), "shown_stem": shown[arm]["stem"],
                               "seeds": sorted(seeds, key=lambda r: str(r["stem"]))}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True, default=str)
    return report


def row(rep):
    """One table row per arm (its shown output) for the report TSV."""
    out = []
    for arm, a in rep["arms"].items():
        r = next(s for s in a["seeds"] if s["stem"] == a["shown_stem"])
        m, b = r["metrics"], r["blocks"]
        st = r.get("swap_tracking") or {}
        out.append([rep["device"], rep["circuit"], arm, m["two_qubit_gates"], m["two_qubit_depth"],
                    m["depth_with_measure"], m["one_qubit_pulses"], b["swap_class_blocks"],
                    b["two_qubit_gates_in_swap_blocks"], b["fused_move_blocks"], st.get("hidden_moves"),
                    st.get("consistent_after_resolve"), b["slack_2q"], r["couplers_used"],
                    r.get("esp_gates_readout"), r.get("wall_s"), r.get("budget_s"),
                    (r.get("check") or {}).get("status"), r.get("agrees_with_record"), r["qasm_sha256"][:12]])
    return out


HEADER = ["device", "circuit", "arm", "n2", "d2", "depth_m", "p1", "swapblk", "cz_in_swaps", "fusedblk", "hidden",
          "resolved", "slack", "couplers", "esp_model", "wall_s", "budget_s", "check", "agrees", "sha"]

if __name__ == "__main__":
    rep = main()
    for x in row(rep):
        print("\t".join(str(v) for v in x))
