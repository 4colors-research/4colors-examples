"""Scoring for the "edge of the possible" post: when is a hardware answer still usable?

Plain Python, no Qiskit, no API, so every rule here is unit-tested offline.

A known-answer circuit (Bernstein-Vazirani here) has one correct output bitstring. A run of `shots` shots on a
real chip returns counts. We call the run **usable** when a reader who only sees the counts would pick the
right answer with confidence:

1. the correct bitstring is the single most frequent outcome, and
2. it leads the runner-up (the most frequent wrong bitstring) by z >= USABLE_Z, where
   z = (k_answer - k_runner_up) / sqrt(k_answer + k_runner_up)   (two Poisson counts).

Rule 1 is the convention of Mundada et al., Phys. Rev. Applied 20, 024034 (2023) (the right answer is the most
probable outcome). Rule 2 stops a tie of 2 counts against 1 from counting as an answer: at 50+ qubits almost
every wrong outcome is seen once, so "most frequent" alone is too weak.

The **edge** of an arm on a device is the largest size n in the pre-registered grid whose cell is usable.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

#: Margin a usable answer needs over the runner-up (in standard errors). Fixed before any run.
USABLE_Z = 3.0
#: A pair (ours vs rival) is a win or a loss only at |z| >= this, and only if both arms pass A/A.
PAIR_Z = 3.0
#: An arm whose two repetitions differ by z >= this fails A/A; its cells are never charted.
AA_Z = 3.0
#: PUB order inside one job is shuffled with this seed (fixed before any run).
ORDER_SEED = 7


def merge_counts(reps: Iterable[Mapping[str, int]]) -> Dict[str, int]:
    """Sum the counts of the repetitions of one arm (A/A reps of the same compiled circuit)."""
    out: Dict[str, int] = {}
    for c in reps:
        for key, v in c.items():
            out[key] = out.get(key, 0) + int(v)
    return out


def score_counts(counts: Mapping[str, int], answer: str, *, z_min: float = USABLE_Z) -> dict:
    """Score one arm's counts against the known answer (a key in the same bit order as `counts`)."""
    shots = sum(int(v) for v in counts.values())
    if shots <= 0:
        raise ValueError("no shots")
    k = int(counts.get(answer, 0))
    others = sorted((int(v) for key, v in counts.items() if key != answer), reverse=True)
    runner = others[0] if others else 0
    rank = 1 + sum(1 for v in others if v >= k) if k > 0 else None   # a tie for first is not first
    p = k / shots
    margin_z = (k - runner) / math.sqrt(k + runner) if k + runner > 0 else 0.0
    return {"shots": shots, "k": k, "p": p, "se": math.sqrt(p * (1 - p) / shots), "runner_up": runner,
            "rank": rank, "margin_z": margin_z, "usable": rank == 1 and margin_z >= z_min}


def bitwise_majority(counts: Mapping[str, int]) -> str:
    """The per-bit majority vote over all shots (ties to '0'). A weaker, multi-shot decoder: recorded as a
    labeled extra column, never the post's usability rule."""
    keys = list(counts)
    if not keys:
        raise ValueError("no counts")
    width = len(keys[0])
    ones = [0] * width
    total = 0
    for key, v in counts.items():
        total += v
        for i, ch in enumerate(key):
            if ch == "1":
                ones[i] += v
    return "".join("1" if 2 * o > total else "0" for o in ones)


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """z of p1 - p2 with each arm's own standard error (0 when both SEs are 0)."""
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) / se if se > 0 else 0.0


def aa_z(rep0: Mapping[str, int], rep1: Mapping[str, int], answer: str) -> float:
    """A/A check of one arm: |z| between its two repetitions' exact-answer rates."""
    n0, n1 = sum(rep0.values()), sum(rep1.values())
    return abs(two_prop_z(rep0.get(answer, 0), n0, rep1.get(answer, 0), n1))


def pair_verdict(ours: dict, rival: dict, aa_ours: float, aa_rival: float) -> dict:
    """Pre-registered verdict for ours vs rival on one cell (both dicts from `score_counts`)."""
    z = two_prop_z(ours["k"], ours["shots"], rival["k"], rival["shots"])
    if rival["k"] == 0:
        ratio = ours["p"] / (0.5 / rival["shots"]) if ours["k"] else None
        ratio_note = "rival k = 0: ratio >= shown value"
    else:
        ratio, ratio_note = ours["p"] / rival["p"], ""
    if max(aa_ours, aa_rival) >= AA_Z:
        verdict = "AA-FAIL"
    elif z >= PAIR_Z:
        verdict = "WIN"
    elif z <= -PAIR_Z:
        verdict = "LOSS"
    else:
        verdict = "TIE"
    return {"z": z, "ratio": ratio, "ratio_note": ratio_note, "verdict": verdict,
            "edge_flip": verdict == "WIN" and ours["usable"] and not rival["usable"]}


def edge(cells: Sequence[dict], arm: str) -> dict:
    """The largest n whose cell for `arm` is usable, and whether every smaller grid size was usable too.
    `cells` rows carry n, arm, usable (and aa_ok, an arm that failed A/A never sets an edge)."""
    mine = sorted((c for c in cells if c["arm"] == arm), key=lambda c: c["n"])
    ok = [c["n"] for c in mine if c["usable"] and c.get("aa_ok", True)]
    if not ok:
        return {"arm": arm, "edge_n": None, "monotone": True, "sizes": [c["n"] for c in mine]}
    top = max(ok)
    below = [c for c in mine if c["n"] <= top]
    return {"arm": arm, "edge_n": top, "monotone": all(c["usable"] and c.get("aa_ok", True) for c in below),
            "sizes": [c["n"] for c in mine]}


def shown(pairs: Sequence[dict]) -> List[dict]:
    """The cells a chart or table may show (owner, 22 Nov 2026: wins only; ties and losses stay in the record
    and the caption says which cells are shown)."""
    return [p for p in pairs if p["verdict"] == "WIN"]


def pub_order(keys: Sequence[str], seed: int = ORDER_SEED) -> List[str]:
    """The fixed shuffled order of the PUBs in one job (so drift inside the job hits both arms alike)."""
    out = list(keys)
    random.Random(seed).shuffle(out)
    return out


def qpu_estimate_s(n_pubs: int, shots: int, *, s_per_shot: float, overhead_s: float) -> float:
    """QPU seconds for one job: a fixed overhead plus a per-shot cost taken from an earlier recorded job on the
    same device family (see the post). An estimate for the budget request, never a measurement."""
    return overhead_s + n_pubs * shots * s_per_shot


def first_unusable(cells: Sequence[dict], arm: str) -> Optional[int]:
    """The smallest grid size where `arm` is not usable (None if it is usable everywhere)."""
    for c in sorted((c for c in cells if c["arm"] == arm), key=lambda c: c["n"]):
        if not (c["usable"] and c.get("aa_ok", True)):
            return c["n"]
    return None
