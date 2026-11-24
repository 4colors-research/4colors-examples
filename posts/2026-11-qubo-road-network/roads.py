"""Max-Cut and balanced partition of an OpenStreetMap road network, as QUBO models.

Loading (nothing is downloaded unless you ask for it):
- `--graphml file.graphml`: a graph saved by osmnx (`osmnx.save_graphml`).
- `--osm file.osm`: an OSM XML extract, read by `osmnx.graph_from_xml`. A Geofabrik `.osm.pbf` extract
  converts with osmium: `osmium tags-filter in.osm.pbf w/highway -o roads.osm.pbf` then
  `osmium cat roads.osm.pbf -o roads.osm`.
- `--place "Utrecht, Netherlands" --allow-download`: osmnx fetches the drivable roads from the Overpass API.
OpenStreetMap data is (c) OpenStreetMap contributors and licensed under the Open Database License (ODbL):
credit it on any figure or table you publish, and share derived databases under the same license.

The road graph becomes a simple undirected graph: one node per intersection or dead end (osmnx's
simplified graph), one edge per pair of nodes joined by any road, both directions and parallel roads merged.
Edge weights: 1 (`unit`) or the road length in whole meters (`length`, the shortest parallel road), so
every coefficient is an integer and energies are exact.

Problems (x_i in {0,1} is the side of node i):
- Max-Cut: maximize cut(x) = Σ w_ij (x_i + x_j - 2 x_i x_j).
- Balanced partition: minimize cut(x) subject to lo <= Σ x_i <= hi (one exact linear row), which splits
  the region into two near-equal halves with as few road crossings as possible.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

OSM_CREDIT = "Map data (c) OpenStreetMap contributors, ODbL"


def load_graph(*, graphml=None, osm=None, place=None, allow_download=False, network_type="drive"):
    """An osmnx road graph from a local file, or (only with allow_download) from Overpass."""
    import osmnx as ox

    if graphml:
        return ox.load_graphml(graphml)
    if osm:
        return ox.graph_from_xml(osm, simplify=True, retain_all=False)
    if place:
        if not allow_download:
            raise SystemExit("downloading needs --allow-download (the data is ODbL; see the README)")
        return ox.graph_from_place(place, network_type=network_type, simplify=True)
    raise SystemExit("give --graphml, --osm or --place")


def to_simple(G, weight: str = "unit") -> dict:
    """Simple undirected graph arrays from a (multi, directed) road graph: node ids 0..n-1 (with the OSM
    ids and coordinates kept), edges u < v, integer weights."""
    nodes = list(G.nodes)
    index = {v: i for i, v in enumerate(nodes)}
    best = {}
    for a, b, data in G.edges(data=True):
        i, j = index[a], index[b]
        if i == j:
            continue                                       # a loop road cannot be cut
        key = (i, j) if i < j else (j, i)
        if weight == "unit":
            w = 1
        else:
            w = max(1, int(round(float(data.get("length", 1.0)))))
        best[key] = min(best.get(key, w), w)               # parallel roads and both directions: one edge
    keys = sorted(best)
    u = np.array([k[0] for k in keys], dtype=np.int64)
    v = np.array([k[1] for k in keys], dtype=np.int64)
    w = np.array([best[k] for k in keys], dtype=np.int64)
    xs = np.array([float(G.nodes[a].get("x", np.nan)) for a in nodes])
    ys = np.array([float(G.nodes[a].get("y", np.nan)) for a in nodes])
    return {"n": len(nodes), "u": u, "v": v, "w": w, "osmid": np.array([str(a) for a in nodes]), "x": xs, "y": ys,
            "weight": weight}


def cut_value(g: dict, x) -> int:
    x = np.asarray(x)
    return int(g["w"][x[g["u"]] != x[g["v"]]].sum())


def qubo_matrix(g: dict) -> sp.coo_matrix:
    """Q with E(x) = x^T Q x = cut(x): diagonal Σ_j w_ij (linear, as x*x = x), upper triangle -2 w_ij."""
    n = g["n"]
    deg = np.bincount(g["u"], weights=g["w"], minlength=n) + np.bincount(g["v"], weights=g["w"], minlength=n)
    rows = np.concatenate([np.arange(n), g["u"]])
    cols = np.concatenate([np.arange(n), g["v"]])
    vals = np.concatenate([deg.astype(np.int64), -2 * g["w"]])
    keep = vals != 0
    return sp.coo_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n))


def balance_bounds(n: int, imbalance: float):
    """lo <= Σ x <= hi for sides within imbalance * n of a perfect half."""
    slack = int(np.floor(imbalance * n / 2))
    return n // 2 - slack, (n + 1) // 2 + slack


def maxcut_model(g: dict):
    import fourcolors.qubo as fq

    return fq.to_model(qubo_matrix(g), sense="max")


def bisection_model(g: dict, imbalance: float):
    import fourcolors.qubo as fq

    m = fq.to_model(qubo_matrix(g), sense="min")
    lo, hi = balance_bounds(g["n"], imbalance)
    m.add_constraint(((i, 1) for i in range(g["n"])), lo=lo, hi=hi)     # one exact row: lo <= Σ x <= hi
    return m, (lo, hi)


# ---- a simple local-search baseline ---------------------------------------------------------------

def _adjacency(g: dict):
    n = g["n"]
    A = sp.coo_matrix((np.concatenate([g["w"], g["w"]]), (np.concatenate([g["u"], g["v"]]),
                                                         np.concatenate([g["v"], g["u"]]))), shape=(n, n)).tocsr()
    P = A.copy(); P.data = np.ones_like(P.data, dtype=np.float64)
    return A, P


def _independent(P, cand, gain, rng):
    """Candidates that beat every candidate neighbor on a random priority: no two are adjacent, so their
    gains add up exactly when they are flipped together."""
    pr = np.where(cand, rng.random(len(cand)) + 1.0, 0.0)
    M = P.copy()
    M.data = pr[M.indices]                          # each stored entry (i, j) holds j's priority
    nb = np.asarray(M.max(axis=1).todense()).ravel()
    return cand & (pr > nb)


def local_search(g: dict, x0, *, maximize: bool, time_limit_s: float, seed: int = 0, balanced: bool = False):
    """Greedy 1-flip descent, many independent flips per round, until no flip helps or time runs out.
    balanced=True flips as many nodes from side 0 as from side 1 in each round, so Σx never changes.
    Returns (x, rounds, seconds)."""
    rng = np.random.default_rng(seed)
    A, P = _adjacency(g)
    x = np.asarray(x0, dtype=np.int64).copy()
    t0 = time.perf_counter()
    rounds = 0
    while time.perf_counter() - t0 < time_limit_s:
        s = 2 * x - 1
        same = s * (A @ s)                          # Σ_j w_ij s_i s_j: + for same-side weight, - for cut
        gain = same if maximize else -same          # change of the objective if node i flips
        sel = _independent(P, gain > 0, gain, rng)
        if balanced:
            a, b = np.flatnonzero(sel & (x == 0)), np.flatnonzero(sel & (x == 1))
            k = min(len(a), len(b))
            a = a[np.argsort(-gain[a])[:k]]; b = b[np.argsort(-gain[b])[:k]]
            sel = np.zeros_like(sel); sel[a] = True; sel[b] = True
        if not sel.any():
            break
        x[sel] = 1 - x[sel]
        rounds += 1
    return x, rounds, time.perf_counter() - t0


def geometric_split(g: dict, lo: int, hi: int):
    """Split at the median of the longer coordinate axis: the obvious first answer for a road map."""
    n = g["n"]
    span_x = np.nanmax(g["x"]) - np.nanmin(g["x"]) if np.isfinite(g["x"]).any() else 0
    span_y = np.nanmax(g["y"]) - np.nanmin(g["y"]) if np.isfinite(g["y"]).any() else 0
    key = g["x"] if span_x >= span_y else g["y"]
    order = np.argsort(np.nan_to_num(key, nan=0.0), kind="stable")
    x = np.zeros(n, dtype=np.int64)
    x[order[n - (lo + hi) // 2:]] = 1
    return x
