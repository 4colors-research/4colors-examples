"""Route assignment as a QUBO, in the formulation of Neukart et al. (Front. ICT 4, 29, 2017), scaled up.

Each car i has a few candidate routes j (its shortest path and alternatives); a 0/1 variable q[i, j] says car i takes
route j. One exact row per car says it takes exactly one route. The objective is the paper's congestion measure,
the sum over street segments of the squared number of cars on it:

    H = sum_s ( sum_{(i, j): s in route(i, j)} q[i, j] )^2

Under the one-hot rows, q^2 = q gives a linear term (the route's length in segments) and each pair of routes of
DIFFERENT cars sharing segments gets 2 x (number of shared segments). Pairs of routes of the same car are left out:
at most one of them is 1 under the rows, so they never contribute. With the rows in force `energy` equals
`congestion` exactly (tested).

Road graphs: `grid_city` (a synthetic grid with a few fast arterials, for tests and the smoke run) and `osm_graph`
(drivable ways from an OpenStreetMap .osm XML file the reader downloads at run time; data (c) OpenStreetMap
contributors, ODbL). Everything is plain Python plus heapq; no routing engine.
"""
from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

Graph = Dict[int, Dict[int, float]]          # node -> {neighbor: length in meters}
Route = Tuple[Tuple[int, int], ...]          # the segments (u, v) with u < v, in travel order

DRIVABLE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link", "living_street"}


# ---- road graphs -----------------------------------------------------------------------------------

def grid_city(w: int, h: int, *, block_m: float = 100.0, arterial_every: int = 4, arterial_speedup: float = 2.0
              ) -> Tuple[Graph, Dict[int, Tuple[float, float]]]:
    """A w x h street grid; every `arterial_every`-th row and column is faster (its edges count shorter), so
    shortest paths pile onto the arterials the way they do in a real city."""
    G: Graph = defaultdict(dict)
    xy = {}
    for r in range(h):
        for c in range(w):
            u = r * w + c
            xy[u] = (c * block_m, r * block_m)
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < h and cc < w:
                    v = rr * w + cc
                    fast = (dr == 0 and r % arterial_every == 0) or (dc == 0 and c % arterial_every == 0)
                    L = block_m / arterial_speedup if fast else block_m
                    G[u][v] = G[v][u] = L
    return dict(G), xy


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    s = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371008.8 * math.asin(math.sqrt(s))


def osm_graph(path: str) -> Tuple[Graph, Dict[int, Tuple[float, float]]]:
    """The drivable street graph of an .osm XML file (largest connected component), nodes renumbered 0..n-1,
    edge lengths in meters. Coordinates are (lon, lat) for plotting. Streams the file (xml.etree.iterparse)."""
    import xml.etree.ElementTree as ET

    coords: Dict[str, Tuple[float, float]] = {}
    ways: List[List[str]] = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "node":
            coords[el.get("id")] = (float(el.get("lat")), float(el.get("lon")))
            el.clear()
        elif el.tag == "way":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if tags.get("highway") in DRIVABLE and tags.get("access") not in ("no", "private"):
                ways.append([nd.get("ref") for nd in el.findall("nd")])
            el.clear()
    G: Graph = defaultdict(dict)
    for refs in ways:
        for a, b in zip(refs, refs[1:]):
            if a in coords and b in coords and a != b:
                d = haversine_m(coords[a], coords[b])
                G[a][b] = G[b][a] = min(d, G[a].get(b, math.inf))
    comp = largest_component(G)
    ids = {old: k for k, old in enumerate(sorted(comp))}
    H: Graph = {ids[u]: {ids[v]: d for v, d in G[u].items() if v in ids} for u in comp}
    xy = {ids[u]: (coords[u][1], coords[u][0]) for u in comp}
    return H, xy


def largest_component(G) -> set:
    seen, best = set(), set()
    for s in G:
        if s in seen:
            continue
        comp, stack = {s}, [s]
        while stack:
            u = stack.pop()
            for v in G[u]:
                if v not in comp:
                    comp.add(v)
                    stack.append(v)
        seen |= comp
        if len(comp) > len(best):
            best = comp
    return best


# ---- cars and candidate routes ---------------------------------------------------------------------

def shortest_path(G: Graph, s: int, t: int, extra: Dict[Tuple[int, int], float] | None = None) -> List[int]:
    """Dijkstra; `extra` multiplies the length of listed segments (used to find alternatives)."""
    extra = extra or {}
    dist, prev, pq = {s: 0.0}, {}, [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, L in G[u].items():
            nd = d + L * extra.get(seg(u, v), 1.0)
            if nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if t not in dist:
        return []
    path = [t]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return path[::-1]


def seg(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def route_of(path: Sequence[int]) -> Route:
    return tuple(seg(a, b) for a, b in zip(path, path[1:]))


def route_length(G: Graph, route: Route) -> float:
    return sum(G[u][v] for u, v in route)


def candidate_routes(G: Graph, s: int, t: int, k: int = 3, penalty: float = 1.6) -> List[Route]:
    """The shortest route, then up to k-1 alternatives: each next route is the shortest path after multiplying the
    length of every segment already used by `penalty` (the penalty method of alternative routing). Duplicates are
    dropped, so a car can have fewer than k routes."""
    routes: List[Route] = []
    extra: Dict[Tuple[int, int], float] = {}
    for _ in range(k * 2):
        if len(routes) == k:
            break
        p = shortest_path(G, s, t, extra)
        if not p:
            break
        r = route_of(p)
        if r not in routes:
            routes.append(r)
        for e in r:
            extra[e] = extra.get(e, 1.0) * penalty
    return routes


def random_cars(G: Graph, n_cars: int, seed: int, *, min_hops: int = 5) -> List[Tuple[int, int]]:
    """Origin-destination pairs drawn uniformly over nodes, at least `min_hops` apart by shortest path."""
    rng = random.Random(seed)
    nodes = sorted(G)
    cars = []
    while len(cars) < n_cars:
        s, t = rng.choice(nodes), rng.choice(nodes)
        if s != t and len(shortest_path(G, s, t)) > min_hops:
            cars.append((s, t))
    return cars


# ---- the model --------------------------------------------------------------------------------------

def variables(routes: Sequence[Sequence[Route]]) -> List[Tuple[int, int]]:
    """(car, route index) for every variable, car-major."""
    return [(i, j) for i, rs in enumerate(routes) for j in range(len(rs))]


def qubo_terms(routes: Sequence[Sequence[Route]]):
    """(linear, quadratic) of the congestion objective: linear[v] = segments on route v; quadratic[(a, b)] =
    2 x shared segments, for a < b on different cars."""
    var = variables(routes)
    index = {vc: k for k, vc in enumerate(var)}
    on_segment: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    linear = [0] * len(var)
    for (i, j), k in index.items():
        for e in routes[i][j]:
            on_segment[e].append(k)
            linear[k] += 1
    quad: Dict[Tuple[int, int], int] = defaultdict(int)
    for ks in on_segment.values():
        ks.sort()
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                a, b = ks[x], ks[y]
                if var[a][0] != var[b][0]:
                    quad[(a, b)] += 2
    return linear, dict(quad)


def count_terms(routes: Sequence[Sequence[Route]]) -> Dict[str, int]:
    """Model size before building it: variables, distinct quadratic terms, rows, the busiest segment's load."""
    linear, quad = qubo_terms(routes)
    loads = defaultdict(int)
    for rs in routes:
        for r in rs:
            for e in r:
                loads[e] += 1
    return {"n_vars": len(linear), "n_quad": len(quad), "n_rows": len(routes),
            "max_candidates_on_segment": max(loads.values(), default=0)}


def build_model(routes: Sequence[Sequence[Route]]):
    """The QUBO with one exact row per car: sum_j q[i, j] == 1."""
    import fourcolors.qubo

    linear, quad = qubo_terms(routes)
    m = fourcolors.qubo.to_model(quad, linear, n=len(linear), sense="min")
    k = 0
    for rs in routes:
        m.add_constraint({k + j: 1 for j in range(len(rs))}, lo=1, hi=1)
        k += len(rs)
    return m


def decode(routes: Sequence[Sequence[Route]], x: Sequence[int]) -> List[int] | None:
    """The chosen route index per car, or None when some car has not exactly one route."""
    choice, k = [], 0
    for rs in routes:
        picked = [j for j in range(len(rs)) if x[k + j]]
        if len(picked) != 1:
            return None
        choice.append(picked[0])
        k += len(rs)
    return choice


def encode(routes: Sequence[Sequence[Route]], choice: Sequence[int]) -> List[int]:
    x = []
    for rs, c in zip(routes, choice):
        x += [int(j == c) for j in range(len(rs))]
    return x


def loads(routes: Sequence[Sequence[Route]], choice: Sequence[int]) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = defaultdict(int)
    for rs, c in zip(routes, choice):
        for e in rs[c]:
            out[e] += 1
    return dict(out)


def congestion(routes: Sequence[Sequence[Route]], choice: Sequence[int]) -> int:
    """The paper's objective: the sum over segments of the squared load. Plain Python, independent of the solver."""
    return sum(L * L for L in loads(routes, choice).values())


def summary(G: Graph, routes, choice) -> Dict[str, float]:
    ld = loads(routes, choice)
    return {"congestion": sum(L * L for L in ld.values()), "max_load": max(ld.values(), default=0),
            "total_km": round(sum(route_length(G, rs[c]) for rs, c in zip(routes, choice)) / 1000, 3)}


# ---- classical baselines (plain Python, run locally; none of them imitates the service) -----------------

def all_shortest(routes) -> List[int]:
    """Everyone takes route 0, their shortest path: what navigation without coordination does."""
    return [0] * len(routes)


def random_choice(routes, seed: int) -> List[int]:
    """Each car a uniformly random candidate: the baseline the 2017 paper compared against."""
    rng = random.Random(seed)
    return [rng.randrange(len(rs)) for rs in routes]


def best_response(routes, start: Sequence[int], max_rounds: int = 50) -> Tuple[List[int], int]:
    """Best-response dynamics of the congestion game: each car in turn switches to the candidate that lowers the
    objective most, until no car can improve (a local optimum, a Nash equilibrium of the potential game) or
    `max_rounds` passes. Returns the choice and the number of passes. The classical first thing to try."""
    choice = list(start)
    ld: Dict[Tuple[int, int], int] = defaultdict(int)
    for rs, c in zip(routes, choice):
        for e in rs[c]:
            ld[e] += 1
    for rnd in range(1, max_rounds + 1):
        moved = False
        for i, rs in enumerate(routes):
            cur = rs[choice[i]]
            for e in cur:
                ld[e] -= 1
            cost = [sum(2 * ld[e] + 1 for e in r) for r in rs]      # change in sum L^2 when adding route r
            best = min(range(len(rs)), key=lambda j: (cost[j], j != choice[i], j))
            if best != choice[i]:
                moved = True
                choice[i] = best
            for e in rs[choice[i]]:
                ld[e] += 1
        if not moved:
            return choice, rnd
    return choice, max_rounds
