"""Power grids as graphs, and balanced partitioning as a QUBO with exact rows.

Input: a MATPOWER case file (`.m`), the format of the PGLib-OPF benchmark library (CC BY 4.0). Only the `mpc.bus`
and `mpc.branch` matrices are read: buses become nodes, in-service branches become edges (parallel branches are
merged, their weights added). You download the case files yourself; this module downloads nothing.

Why partition: a parallel power-flow or dynamic simulation splits the grid across workers. Each worker gets a part of
about equal work (the balance rows), and every branch between two parts costs communication at every step (the cut).

Models (`x` 0/1; every row is exact, nothing is a penalty):
- bisection: x_i = 1 puts bus i in part 1.
      minimize  sum_{(i,j)} w_ij (x_i + x_j - 2 x_i x_j)            (weight of the cut branches)
      subject to  lo <= sum_i c_i x_i <= hi                          (one balance row)
- k-way: x[i, p] = 1 puts bus i in part p (variable i * k + p).
      minimize  sum_{(i,j)} w_ij (1 - sum_p x_ip x_jp)
      subject to  sum_p x_ip = 1 for every bus (one-hot rows), lo <= sum_i c_i x_ip <= hi for every part
c_i is the work of bus i: 1 per bus (`unit`), or 1 + its number of branches (`degree`).
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import fourcolors.qubo


# ---- MATPOWER --------------------------------------------------------------------------------------

def _matrix(text: str, name: str) -> List[List[float]]:
    m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.S)
    if not m:
        raise ValueError(f"no mpc.{name} matrix")
    rows = []
    for line in m.group(1).splitlines():
        line = line.split("%", 1)[0].strip().rstrip(";").strip()
        if line:
            rows.append([float(v) for v in line.replace(",", " ").split()])
    return rows


def parse_matpower(text: str) -> dict:
    """{'buses': [bus ids], 'branches': [(from id, to id, x, status)]} from a MATPOWER case."""
    bus = _matrix(text, "bus")
    branch = _matrix(text, "branch")
    return {"buses": [int(r[0]) for r in bus],
            "branches": [(int(r[0]), int(r[1]), r[3], int(r[10]) if len(r) > 10 else 1) for r in branch]}


def to_graph(case: dict, weight: str = "unit") -> dict:
    """{'n', 'edges': [(i, j)], 'w': [int], 'ids': [bus id]} on node indices 0..n-1.

    weight `unit`: every branch 1 (merged parallel branches add up); `susceptance`: round(|1/x| / median), at least 1,
    so stiff branches cost more to cut."""
    ids = case["buses"]
    index = {b: i for i, b in enumerate(ids)}
    raw: Dict[Tuple[int, int], float] = {}
    xs = [abs(x) for _, _, x, s in case["branches"] if s and x]
    med = sorted(1 / x for x in xs)[len(xs) // 2] if xs else 1.0
    for f, t, x, status in case["branches"]:
        if not status or f == t:
            continue
        i, j = sorted((index[f], index[t]))
        val = 1.0 if weight == "unit" else max(1.0, (1 / abs(x)) / med if x else 1.0)
        raw[(i, j)] = raw.get((i, j), 0.0) + val
    edges = sorted(raw)
    return {"n": len(ids), "edges": edges, "w": [max(1, int(round(raw[e]))) for e in edges], "ids": ids}


def work(g: dict, kind: str = "unit") -> List[int]:
    if kind == "unit":
        return [1] * g["n"]
    deg = [0] * g["n"]
    for i, j in g["edges"]:
        deg[i] += 1
        deg[j] += 1
    return [1 + d for d in deg]


def bounds(total: int, parts: int, imbalance: float) -> Tuple[int, int]:
    """Integer [lo, hi] for one part's work: within (1 +- imbalance) of total / parts."""
    avg = total / parts
    return int(math.ceil(avg * (1 - imbalance))), int(math.floor(avg * (1 + imbalance)))


# ---- models ---------------------------------------------------------------------------------------

def bisection_model(g: dict, c: Sequence[int], imbalance: float):
    lin = [0.0] * g["n"]
    quad: Dict[Tuple[int, int], float] = {}
    for (i, j), w in zip(g["edges"], g["w"]):
        lin[i] += w
        lin[j] += w
        quad[(i, j)] = quad.get((i, j), 0.0) - 2 * w
    m = fourcolors.qubo.to_model(quad, lin, n=g["n"])
    lo, hi = bounds(sum(c), 2, imbalance)
    m.add_constraint({i: int(c[i]) for i in range(g["n"])}, lo=lo, hi=hi)
    return m


def kway_model(g: dict, c: Sequence[int], parts: int, imbalance: float):
    n, k = g["n"], parts
    quad: Dict[Tuple[int, int], float] = {}
    for (i, j), w in zip(g["edges"], g["w"]):
        for p in range(k):
            a, b = i * k + p, j * k + p
            quad[(a, b)] = quad.get((a, b), 0.0) - w
    m = fourcolors.qubo.to_model(quad, [0.0] * (n * k), n=n * k, offset=float(sum(g["w"])))
    for i in range(n):
        m.add_constraint({i * k + p: 1 for p in range(k)}, lo=1, hi=1)
    lo, hi = bounds(sum(c), k, imbalance)
    for p in range(k):
        m.add_constraint({i * k + p: int(c[i]) for i in range(n)}, lo=lo, hi=hi)
    return m


# ---- decode and check ------------------------------------------------------------------------------

def decode(x: Sequence[int], n: int, parts: int) -> Optional[List[int]]:
    """Part of each bus, or None if some bus has no part or several (k-way); bisection is x itself."""
    if parts == 2 and len(x) == n:
        return [int(v) for v in x]
    out = []
    for i in range(n):
        ps = [p for p in range(parts) if x[i * parts + p]]
        if len(ps) != 1:
            return None
        out.append(ps[0])
    return out


def cut_weight(g: dict, part: Sequence[int]) -> int:
    return sum(w for (i, j), w in zip(g["edges"], g["w"]) if part[i] != part[j])


def part_work(part: Sequence[int], c: Sequence[int], parts: int) -> List[int]:
    s = [0] * parts
    for i, p in enumerate(part):
        s[p] += c[i]
    return s


def balanced(part: Sequence[int], c: Sequence[int], parts: int, imbalance: float) -> bool:
    lo, hi = bounds(sum(c), parts, imbalance)
    return all(lo <= s <= hi for s in part_work(part, c, parts))


# ---- exact reference (labeled; small grids) --------------------------------------------------------

def milp_reference(g: dict, c: Sequence[int], parts: int, imbalance: float, time_limit_s: float = 60.0) -> dict:
    """Min-cut balanced partition by a MILP (scipy.optimize.milp, the HiGHS solver inside SciPy).

    Variables x[i, p] (one-hot) and z_e >= x_ip - x_jp, z_e >= x_jp - x_ip for every part p; minimize sum w_e z_e.
    Returns {'status': 'optimal' | 'time_limit' | ..., 'cut', 'bound', 'part'}; only 'optimal' is a proven optimum."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n, k, E = g["n"], parts, len(g["edges"])
    nx = n * k
    nv = nx + E
    cost = np.zeros(nv)
    cost[nx:] = g["w"]
    rows = n + k + 2 * k * E + 1
    A = lil_matrix((rows, nv))
    lo = np.empty(rows)
    hi = np.empty(rows)
    r = 0
    for i in range(n):
        for p in range(k):
            A[r, i * k + p] = 1
        lo[r] = hi[r] = 1
        r += 1
    blo, bhi = bounds(sum(c), k, imbalance)
    for p in range(k):
        for i in range(n):
            A[r, i * k + p] = c[i]
        lo[r], hi[r] = blo, bhi
        r += 1
    for e, (i, j) in enumerate(g["edges"]):
        for p in range(k):
            for a, b in ((i, j), (j, i)):
                A[r, nx + e] = 1
                A[r, a * k + p] = -1
                A[r, b * k + p] = 1
                lo[r], hi[r] = 0, np.inf
                r += 1
    A[r, 0] = 1                                    # symmetry: bus 0 in part 0
    lo[r] = hi[r] = 1
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options={"time_limit": time_limit_s, "disp": False})
    status = {0: "optimal", 1: "time_limit"}.get(res.status, f"status_{res.status}")
    part = None
    if res.x is not None:
        xv = np.round(res.x[:nx]).astype(int)
        part = decode(list(xv), n, k) if k > 2 else [int(1 - xv[i * k]) for i in range(n)]
    return {"status": status, "cut": None if part is None else cut_weight(g, part),
            "bound": getattr(res, "mip_dual_bound", None), "part": part}
