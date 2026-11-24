"""Cells, the selection rule and the report card table, from the run records only.

A cell is (family, qubits). Its value per arm is the median two-qubit gate count over the cell's instances and
seeds. The rival per (circuit, seed) is the LOWER of Qiskit's two arms (best of trials within T, and one default
call), so the bar 4colors has to clear is the stricter of the two.

Selection rule (fixed before the run, printed under every chart):
  a cell is shown when the 4colors median is lower than the rival median by more than the tie band (2 %), every
  4colors compile of the cell returned a circuit, none was labeled wrong, and no Qiskit stream of the cell
  overran T on its first trial (such cells are reported apart, never as a Qiskit loss).
A family is shown when at least one of its cells is shown. Ties, losses and apart cells stay in the record and in
`selection.md`; they are never charted.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import examplekit

TIE_BAND = 0.02
WRONG_SUFFIX = "Do not use this result."


def _key(r) -> Tuple[str, int]:
    return r["family"], r["qubits"]


def cells(rows: List[dict], budget: float = 10.0) -> Dict[Tuple[str, int], dict]:
    res = [r for r in rows if r.get("kind") == "result" and r.get("budget_s") == budget]
    circ = {r["circuit_id"]: r for r in rows if r.get("kind") == "circuit"}
    by: Dict[Tuple[str, int], dict] = {}
    pair: Dict[Tuple[str, int], dict] = {}
    for r in res:
        c = by.setdefault(_key(r), {"ours": [], "rival": [], "best": [], "once": [], "depth_ours": [],
                                    "depth_rival": [], "refused": 0, "wrong": 0, "over_budget": 0, "labels": {},
                                    "input_2q": r.get("input_2q"), "circuits": set()})
        c["circuits"].add(r["circuit_id"])
        if r["arm"] == "4colors":
            if r.get("refusal_code") or r.get("two_qubit_gates") is None:
                c["refused"] += 1
                continue
            if str(r.get("verification_reason") or "").endswith(WRONG_SUFFIX):
                c["wrong"] += 1
            c["ours"].append(r["two_qubit_gates"]); c["depth_ours"].append(r.get("two_qubit_depth"))
            lab = r.get("verification_label") or "none"
            c["labels"][lab] = c["labels"].get(lab, 0) + 1
        else:
            pair.setdefault((r["circuit_id"], r["seed"]), {})[r["arm"]] = r
    for (cid, _seed), arms in pair.items():
        if "qiskit-o3-best" not in arms or "qiskit-o3-once" not in arms:
            continue
        b, o = arms["qiskit-o3-best"], arms["qiskit-o3-once"]
        c = by[_key(b)]
        c["over_budget"] += int(bool(b.get("over_budget")))
        low = min((b, o), key=lambda r: (r["two_qubit_gates"], r["two_qubit_depth"]))
        c["rival"].append(low["two_qubit_gates"]); c["depth_rival"].append(low["two_qubit_depth"])
        c["best"].append(b["two_qubit_gates"]); c["once"].append(o["two_qubit_gates"])
    for k, c in by.items():
        c["domain"] = next((circ[i].get("domain") for i in c["circuits"] if i in circ), None)
        c["sha256"] = sorted({circ[i]["sha256"] for i in c["circuits"] if i in circ})
        c["circuits"] = sorted(c["circuits"])
        for f in ("ours", "rival", "best", "once", "depth_ours", "depth_rival"):
            c["median_" + f] = examplekit.median(c[f])
        mo, mr = c["median_ours"], c["median_rival"]
        c["pct_fewer"] = None if mo is None or not mr else 100.0 * (mr - mo) / mr
        c["verdict"] = verdict(c)
    return by


def verdict(c: dict) -> str:
    if c["refused"] or c["median_ours"] is None:
        return "no answer"
    if c["wrong"]:
        return "wrong label"
    if c["median_rival"] is None:
        return "no rival"
    if c["over_budget"]:
        return "apart: Qiskit overran T"
    if c["median_ours"] < (1 - TIE_BAND) * c["median_rival"]:
        return "win"
    if c["median_ours"] > (1 + TIE_BAND) * c["median_rival"]:
        return "loss"
    return "tie"


def shown(by: Dict[Tuple[str, int], dict]) -> List[Tuple[str, int]]:
    return sorted((k for k, c in by.items() if c["verdict"] == "win"), key=lambda k: (-by[k]["pct_fewer"], k))


def caption(by, n_families_run: int) -> str:
    return ("Cells (family, size) where the 4colors median used more than 2 % fewer two-qubit gates than the lower "
            "of Qiskit O3's best-of-trials and its single default call, at the same 10 s, on FakeFez with its "
            "published calibration; every 4colors compile in a shown cell returned a circuit. "
            "The full record, every cell included, is linked from the run id.")


def internal_counts(by, n_families_run: int) -> str:
    """Owner 23 Nov: no tallies in copy. The counts stay in selection.md, marked internal."""
    fams = sorted({k[0] for k in shown(by)})
    return (f"Internal (not for publication): we ran {n_families_run} families; {len(fams)} are shown. "
            f"Cells shown: {len(shown(by))} of {len(by)}.")


def selection_md(by, n_families_run: int) -> str:
    lines = ["# Selection", "", caption(by, n_families_run), "", internal_counts(by, n_families_run), "", "| family | qubits | verdict | median 4colors | "
             "median rival | % fewer | labels |", "|---|---|---|---|---|---|---|"]
    for k in sorted(by):
        c = by[k]
        pct = "" if c["pct_fewer"] is None else f"{c['pct_fewer']:.1f}"
        lines.append(f"| {k[0]} | {k[1]} | {c['verdict']} | {c['median_ours']} | {c['median_rival']} | {pct} | "
                     f"{c['labels']} |")
    lines += ["", "Ties, losses and apart cells are in this file and the record only; no chart or table shows them."]
    return "\n".join(lines)


def card_table(by) -> str:
    """The post's table: shown cells only."""
    lines = ["| Family (domain) | Qubits | Input 2q | Qiskit O3 (lower of best-of and once) | 4colors, 10 s | Fewer |",
             "|---|---|---|---|---|---|"]
    for k in shown(by):
        c = by[k]
        lines.append(f"| {k[0]} ({c['domain']}) | {k[1]} | {c['input_2q']} | {c['median_rival']} | "
                     f"{c['median_ours']} | {c['pct_fewer']:.1f} % |")
    return "\n".join(lines)
