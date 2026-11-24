"""RNA secondary structure as the stem-selection QUBO of Fox et al. (PLoS Comput Biol 18(4): e1010032, 2022).

A stem m = (i, j, k) is k stacked base pairs (i, j), (i+1, j-1), ..., (i+k-1, j-k+1), each A-U, G-C or G-U.
One 0/1 variable per candidate stem. The energy is the paper's Eq 13:

    H(q) = sum_i [c_L (k_i^2 - 2 k_i mu + mu^2) - c_B k_i^2] q_i  -  sum_{j<l} c_B k_j k_l delta_jl q_j q_l

with mu the length of the longest candidate stem, delta_jl = c_P when stems j and l cross (a pseudoknot) and 1
otherwise. The paper forbids overlapping stems (two stems that share a base) with an infinite penalty; here that
rule is a set of exact rows instead: one row per base, "at most one chosen stem uses this base" (or, equivalently,
one "not both" row per overlapping pair). The default parameters are the paper's: c_B = 1, c_L = 10, c_P = 0.5,
stems of at least 3 pairs (all sub-stems included), at least 2 unpaired bases in a hairpin loop.

Coefficients are sent as 2 H (SCALE) so that c_P = 0.5 stays an integer; energies are exact.

This module has no solver of its own. `exact_reference` is a labeled exact branch and bound over sets of
compatible stems, used only to score answers on small instances; it refuses (returns status "gave_up") rather
than guess when its node budget runs out.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
C_B, C_L, C_P = 1, 10, Fraction(1, 2)
SCALE = 2                       # the model holds SCALE * H so every coefficient is an integer
BRACKETS = ["()", "[]", "{}", "<>"]

# The worked example of Fox et al. 2022 (18 nt).
FOX_EXAMPLE = "GGAAGCAAACAUCCCUGU"


@dataclass(frozen=True)
class Stem:
    i: int          # first base of the outer pair
    j: int          # last base of the outer pair
    k: int          # number of stacked pairs

    def pairs(self) -> List[Tuple[int, int]]:
        return [(self.i + t, self.j - t) for t in range(self.k)]

    def bases(self) -> set:
        return {b for p in self.pairs() for b in p}


# ---- sequences ------------------------------------------------------------------------------------

def clean(seq: str) -> str:
    """Upper case, T read as U; anything but A, C, G, U is refused."""
    s = "".join(seq.split()).upper().replace("T", "U")
    bad = set(s) - set("ACGU")
    if bad:
        raise ValueError(f"not an RNA sequence: {sorted(bad)}")
    return s


def random_sequence(length: int, seed: int) -> str:
    rnd = random.Random(seed)
    return "".join(rnd.choice("ACGU") for _ in range(length))


def read_fasta(path) -> List[Tuple[str, str]]:
    """(name, sequence) records of a FASTA file; lines starting with '>' name the next record."""
    out, name, buf = [], None, []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if name is not None:
                out.append((name, clean("".join(buf))))
            name, buf = line[1:].strip() or f"seq{len(out) + 1}", []
        else:
            buf.append(line)
    if name is not None:
        out.append((name, clean("".join(buf))))
    return out


# ---- stems and their relations --------------------------------------------------------------------

def can_pair(a: str, b: str) -> bool:
    return (a, b) in PAIRS


def stems(seq: str, min_stem: int = 3, min_loop: int = 2) -> List[Stem]:
    """Every candidate stem (all sub-stems included): k >= min_stem stacked canonical pairs whose innermost pair
    encloses at least min_loop unpaired bases. Sorted by (i, j, k)."""
    seq = clean(seq)
    n = len(seq)
    out = []
    for i in range(n):
        for j in range(n - 1, i, -1):
            k = 0
            while i + k < j - k and can_pair(seq[i + k], seq[j - k]):
                k += 1
                inner_loop = (j - k + 1) - (i + k - 1) - 1
                if inner_loop < min_loop:
                    break
                if k >= min_stem:
                    out.append(Stem(i, j, k))
    return sorted(set(out), key=lambda s: (s.i, s.j, s.k))


def overlap(a: Stem, b: Stem) -> bool:
    """Two stems overlap when they use a common base."""
    return bool(a.bases() & b.bases())


def crossing(a: Stem, b: Stem) -> bool:
    """A pseudoknot: non-overlapping stems whose outer pairs cross, i_a < i_b < j_a < j_b or the reverse."""
    if overlap(a, b):
        return False
    return a.i < b.i < a.j < b.j or b.i < a.i < b.j < a.j


def longest(st: Sequence[Stem]) -> int:
    return max((s.k for s in st), default=0)


# ---- the energy (Eq 13) ---------------------------------------------------------------------------

def coefficients(st: Sequence[Stem], c_b=C_B, c_l=C_L, c_p=C_P):
    """(linear, quadratic) of H as exact Fractions: linear[i], quadratic[(a, b)] for a < b, non-overlapping pairs
    only (an overlapping pair is excluded by the rows, so its term never counts in a feasible answer)."""
    mu = longest(st)
    lin = [Fraction(c_l) * (s.k ** 2 - 2 * s.k * mu + mu ** 2) - Fraction(c_b) * s.k ** 2 for s in st]
    quad = {}
    for a, b in itertools.combinations(range(len(st)), 2):
        if overlap(st[a], st[b]):
            continue
        delta = Fraction(c_p) if crossing(st[a], st[b]) else Fraction(1)
        quad[(a, b)] = -Fraction(c_b) * st[a].k * st[b].k * delta
    return lin, quad


def energy(st: Sequence[Stem], x: Sequence[int], c_b=C_B, c_l=C_L, c_p=C_P) -> Fraction:
    """H of a 0/1 choice of stems, straight from Eq 13 (independent of the model code)."""
    mu = longest(st)
    on = [i for i, v in enumerate(x) if v]
    h = Fraction(0)
    for i in on:
        k = st[i].k
        h += Fraction(c_l) * (k * k - 2 * k * mu + mu * mu) - Fraction(c_b) * k * k
    for a, b in itertools.combinations(on, 2):
        delta = Fraction(c_p) if crossing(st[a], st[b]) else Fraction(1)
        h -= Fraction(c_b) * st[a].k * st[b].k * delta
    return h


def compatible(st: Sequence[Stem], x: Sequence[int]) -> bool:
    on = [st[i] for i, v in enumerate(x) if v]
    return not any(overlap(a, b) for a, b in itertools.combinations(on, 2))


# ---- the QUBO with exact rows ---------------------------------------------------------------------

def overlap_rows(st: Sequence[Stem], n_bases: int, kind: str = "base") -> List[Dict[int, int]]:
    """The no-overlap rule as rows (each: sum <= 1).

    kind="base": one row per base used by two or more stems ("at most one chosen stem uses base b").
    kind="pairs": one "not both" row per overlapping pair (x_a + x_b <= 1), the literal form of the paper's rule.
    Both admit exactly the sets of pairwise non-overlapping stems."""
    if kind == "base":
        rows = []
        for b in range(n_bases):
            users = [i for i, s in enumerate(st) if b in s.bases()]
            if len(users) > 1:
                rows.append({i: 1 for i in users})
        # identical rows (the same stems use several bases) are sent once
        seen, out = set(), []
        for r in rows:
            key = tuple(sorted(r))
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out
    if kind == "pairs":
        return [{a: 1, b: 1} for a, b in itertools.combinations(range(len(st)), 2) if overlap(st[a], st[b])]
    raise ValueError("kind is 'base' or 'pairs'")


def build_model(seq: str, st: Sequence[Stem], rows: str = "base", c_b=C_B, c_l=C_L, c_p=C_P):
    """A fourcolors.qubo Model holding SCALE * H, with the no-overlap rows."""
    import fourcolors.qubo

    lin, quad = coefficients(st, c_b, c_l, c_p)
    Q = {}
    for i, v in enumerate(lin):
        w = v * SCALE
        if w:
            assert w.denominator == 1, "choose parameters with SCALE * H integral"
            Q[(i, i)] = int(w)
    for (a, b), v in quad.items():
        w = v * SCALE
        if w:
            assert w.denominator == 1, "choose parameters with SCALE * H integral"
            Q[(a, b)] = int(w)
    m = fourcolors.qubo.to_model(Q, n=len(st))
    for r in overlap_rows(st, len(seq), rows):
        m.add_constraint(r, hi=1)
    return m


# ---- exact reference (labeled, small instances only) ----------------------------------------------

def brute_force(st: Sequence[Stem], **params) -> Tuple[Fraction, List[int]]:
    """Every subset (tests only; n <= 20)."""
    if len(st) > 20:
        raise ValueError("brute force is for at most 20 stems")
    best, arg = Fraction(0), [0] * len(st)
    for bits in itertools.product((0, 1), repeat=len(st)):
        if compatible(st, bits):
            e = energy(st, bits, **params)
            if e < best:
                best, arg = e, list(bits)
    return best, arg


def exact_reference(st: Sequence[Stem], max_nodes: int = 2_000_000, c_b=C_B, c_l=C_L, c_p=C_P) -> dict:
    """Minimum of H over sets of non-overlapping stems, by depth-first branch and bound (in exact integers,
    SCALE * H).

    Every quadratic term is <= 0, so a valid lower bound below a node is the current energy plus, for each
    remaining stem compatible with the chosen ones, min(0, its linear term + its terms with the chosen stems +
    all its terms with the other remaining compatible stems). Returns status "exact" (proved optimal), or
    "gave_up" when max_nodes is reached (then the energy is only the best found and must not be called exact)."""
    import numpy as np

    n = len(st)
    lin, quad = coefficients(st, c_b, c_l, c_p)
    L = np.array([int(v * SCALE) for v in lin], dtype=np.int64)
    Q = np.zeros((n, n), dtype=np.int64)
    for (a, b), v in quad.items():
        Q[a, b] = Q[b, a] = int(v * SCALE)
    bases = [s.bases() for s in st]
    OV = np.array([[bool(bases[a] & bases[b]) for b in range(n)] for a in range(n)], dtype=bool).reshape(n, n)
    best = {"e": 0, "x": []}
    nodes = [0]

    def dfs(cur, chosen, qc, cand):
        nodes[0] += 1
        if nodes[0] > max_nodes:
            raise _GaveUp
        if cur < best["e"]:
            best["e"], best["x"] = cur, list(chosen)
        if cand.size == 0:
            return
        g = L[cand] + qc[cand] + Q[np.ix_(cand, cand)].sum(axis=1)
        if cur + int(np.minimum(g, 0).sum()) >= best["e"]:
            return
        r, rest = int(cand[0]), cand[1:]
        dfs(cur + int(L[r] + qc[r]), chosen + [r], qc + Q[r], rest[~OV[r, rest]])   # take r
        dfs(cur, chosen, qc, rest)                                                  # leave r

    order = np.array(sorted(range(n), key=lambda i: L[i]), dtype=np.int64)      # most attractive stems first
    status = "exact"
    try:
        dfs(0, [], np.zeros(n, dtype=np.int64), order)
    except _GaveUp:
        status = "gave_up"
    x = [0] * n
    for i in best["x"]:
        x[i] = 1
    return {"status": status, "energy": Fraction(best["e"], SCALE), "x": x, "nodes": nodes[0]}


class _GaveUp(Exception):
    pass


# ---- output ---------------------------------------------------------------------------------------

def chosen(st: Sequence[Stem], x: Sequence[int]) -> List[Stem]:
    return [st[i] for i, v in enumerate(x) if v]


def base_pairs(st: Sequence[Stem], x: Sequence[int]) -> List[Tuple[int, int]]:
    return sorted(p for s in chosen(st, x) for p in s.pairs())


def dot_bracket(n: int, sel: Iterable[Stem]) -> str:
    """Dot-bracket string; crossing stems go to the next bracket kind: () then [] then {} then <>."""
    levels: List[List[Stem]] = []
    out = ["."] * n
    for s in sorted(sel, key=lambda s: (s.i, s.j)):
        for lv, members in enumerate(levels):
            if not any(crossing(s, t) for t in members):
                members.append(s)
                break
        else:
            lv = len(levels)
            levels.append([s])
        if lv >= len(BRACKETS):
            raise ValueError("more than four mutually crossing levels")
        o, c = BRACKETS[lv]
        for a, b in s.pairs():
            out[a], out[b] = o, c
    return "".join(out)


def has_pseudoknot(sel: Sequence[Stem]) -> bool:
    return any(crossing(a, b) for a, b in itertools.combinations(sel, 2))
