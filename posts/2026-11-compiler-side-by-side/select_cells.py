"""The post's selection rule, applied to inspect_cell.py reports (fixed before any run).

    python select_cells.py inspect/<device> --out inspect/<device>/selection.md

A pairing is shown when ours has strictly fewer two-qubit gates than its Qiskit level-3 pair (medians over seeds)
and every output of both arms passed check.py (PASS, or APPROX within the D2 bound: a rival's own measured distance,
ours at most the bound it was given). Pairings:
    same time     ours_inst  vs qiskit_o3            ("same time" is claimed only where ours' service-side time is
                                                      at most Qiskit's median wall; otherwise both times are stated)
    same budget   ours_T<T>  vs qiskit_o3best_T<T>
The record-only circuit (C11) is never shown; the control (C12) is shown as the control, never as a win. Every other
pairing is listed as left out, with the reason, and stays in the run record.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuits  # noqa: E402


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _verified(arm: dict, bound=None) -> bool:
    for s in arm["seeds"]:
        c = s.get("check") or {}
        if c.get("status") == "PASS":
            continue
        if c.get("status") == "APPROX" and c.get("distance") is not None and (bound is None or c["distance"] <= bound):
            continue
        return False
    return bool(arm["seeds"])


def _service_s(seed: dict):
    t = seed.get("timings") or {}
    return t.get("compile_s", t.get("elapsed_s", seed.get("billed_s")))


def pairings(report: dict):
    arms = report["arms"]
    out = []
    if "ours_inst" in arms and "qiskit_o3" in arms:
        out.append(("same time", "ours_inst", "qiskit_o3"))
    for arm in arms:
        if arm.startswith("ours_T") and f"qiskit_o3best_T{arm[6:]}" in arms:
            out.append((f"same budget {arm[6:]} s", arm, f"qiskit_o3best_T{arm[6:]}"))
    return out


def select(reports):
    rows = []
    for rep in reports:
        info = circuits.CIRCUITS.get(rep["circuit"], {})
        for kind, ours, rival in pairings(rep):
            o, r = rep["arms"][ours], rep["arms"][rival]
            on2 = _median([s["metrics"]["two_qubit_gates"] for s in o["seeds"]])
            rn2 = _median([s["metrics"]["two_qubit_gates"] for s in r["seeds"]])
            bound = _median([(s.get("options") or {}).get("max_approximation_error") for s in o["seeds"]])
            ok = _verified(r) and _verified(o, bound if bound is not None else 1e-9)
            row = {"device": rep["device"], "circuit": rep["circuit"], "cid": info.get("cid"), "pairing": kind,
                   "ours": ours, "rival": rival, "ours_2q": on2, "rival_2q": rn2, "verified": ok}
            if kind == "same time":
                ot = _median([_service_s(s) for s in o["seeds"]])
                rt = _median([s.get("wall_s") for s in r["seeds"]])
                row.update(ours_s=ot, rival_s=rt, same_time=(ot is not None and rt is not None and ot <= rt))
            if info.get("record_only"):
                row["shown"], row["reason"] = False, "record only (OD-14)"
            elif not ok:
                row["shown"], row["reason"] = False, "an output is not verified"
            elif info.get("control"):
                row["shown"], row["reason"] = True, "control"
            elif on2 is not None and rn2 is not None and on2 < rn2:
                row["shown"], row["reason"] = True, "fewer two-qubit gates"
            else:
                row["shown"], row["reason"] = False, "tie or loss"
            rows.append(row)
    return rows


def markdown(rows) -> str:
    shown = [r for r in rows if r["shown"]]
    lines = [f"Internal (not for publication): pairings shown {len(shown)} of {len(rows)} (selection rule in "
             "the post).", "",
             "| device | circuit | pairing | ours 2q | Qiskit O3 2q | shown | reason |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['device']} | {r['circuit']} | {r['pairing']} | {r['ours_2q']} | {r['rival_2q']} | "
                     f"{'yes' if r['shown'] else 'no'} | {r['reason']} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inspect_dir", help="a device dir of inspect_cell.py output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    reports = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.inspect_dir, "*", "report.json")))]
    rows = select(reports)
    with open(a.out, "w") as f:
        f.write(markdown(rows))
    with open(os.path.splitext(a.out)[0] + ".jsonl", "w") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    print(markdown(rows))


if __name__ == "__main__":
    main()
