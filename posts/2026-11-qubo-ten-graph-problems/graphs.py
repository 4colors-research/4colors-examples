"""Ten graph problems as QUBO models with exact rows, after Lucas (2014).

A. Lucas, "Ising formulations of many NP problems", Frontiers in Physics 2, 5 (2014), wrote penalty forms for dozens
of NP problems. The 4colors Optimizer for QUBO also takes hard rules as exact linear rows,
`m.add_constraint({var: coef}, lo=..., hi=...)`, so each builder below is the objective plus a few rows, with no
penalty weight to tune. `penalty_form` shows the classic Lucas-style alternative for one problem.

Every builder takes a networkx graph and returns a `Problem`: the model, a `decode(x)` that turns the returned 0/1
vector into a graph object (a node set, a coloring, an edge set, a tour) and a `check(answer)` that verifies it in
plain Python and scores it, independently of the solver.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import fourcolors.qubo


@dataclass
class Problem:
    name: str
    model: Any                                   # fourcolors.qubo.Model
    decode: Callable[[List[int]], Any]
    check: Callable[[Any], Dict[str, Any]]       # {"valid": bool, "score": number, ...}
    goal: str                                    # what the score means, e.g. "cut edges (max)"
    info: Dict[str, Any] = field(default_factory=dict)


def _nodes(G) -> Tuple[list, Dict[Any, int]]:
    nodes = list(G.nodes())
    return nodes, {v: i for i, v in enumerate(nodes)}


def _model(Q: Dict[Tuple[int, int], int], lin: List[int], n: int, sense: str = "min"):
    return fourcolors.qubo.to_model(Q, lin, n=n, sense=sense)


def _w(G, u, v, key="weight") -> int:
    return int(G[u][v].get(key, 1))


# 1. Max-Cut ----------------------------------------------------------------------------------------
def max_cut(G) -> Problem:
    """Maximize the edges between the two sides: x_u + x_v - 2 x_u x_v per edge. No rows."""
    nodes, ix = _nodes(G)
    Q, lin = {}, [0] * len(nodes)
    for u, v in G.edges():
        i, j = sorted((ix[u], ix[v]))
        lin[i] += 1; lin[j] += 1
        Q[(i, j)] = Q.get((i, j), 0) - 2
    m = _model(Q, lin, len(nodes), "max")
    decode = lambda x: {nodes[i] for i in range(len(nodes)) if x[i]}
    check = lambda S: {"valid": True, "score": sum(1 for u, v in G.edges() if (u in S) != (v in S))}
    return Problem("max_cut", m, decode, check, "edges cut (max)")


# 2. Maximum independent set ------------------------------------------------------------------------
def max_independent_set(G) -> Problem:
    """Maximize the chosen nodes; one row x_u + x_v <= 1 per edge."""
    nodes, ix = _nodes(G)
    m = _model({}, [1] * len(nodes), len(nodes), "max")
    for u, v in G.edges():
        m.add_constraint({ix[u]: 1, ix[v]: 1}, hi=1)
    decode = lambda x: {nodes[i] for i in range(len(nodes)) if x[i]}
    check = lambda S: {"valid": all(not G.has_edge(u, v) for u, v in itertools.combinations(S, 2)),
                       "score": len(S)}
    return Problem("max_independent_set", m, decode, check, "nodes in the set (max)")


def penalty_form_independent_set(G, P: int = 2):
    """The Lucas-style penalty form of the same problem, for comparison: min -sum x + P * sum_edges x_u x_v.
    Any P > 1 gives the same optimum on an unweighted graph; on weighted or constrained problems P must be tuned,
    which is the work exact rows remove."""
    nodes, ix = _nodes(G)
    Q = {tuple(sorted((ix[u], ix[v]))): P for u, v in G.edges()}
    return _model(Q, [-1] * len(nodes), len(nodes), "min")


# 3. Minimum vertex cover ---------------------------------------------------------------------------
def min_vertex_cover(G) -> Problem:
    """Minimize the chosen nodes; one row x_u + x_v >= 1 per edge (every edge touched)."""
    nodes, ix = _nodes(G)
    m = _model({}, [1] * len(nodes), len(nodes), "min")
    for u, v in G.edges():
        m.add_constraint({ix[u]: 1, ix[v]: 1}, lo=1)
    decode = lambda x: {nodes[i] for i in range(len(nodes)) if x[i]}
    check = lambda S: {"valid": all(u in S or v in S for u, v in G.edges()), "score": len(S)}
    return Problem("min_vertex_cover", m, decode, check, "nodes in the cover (min)")


# 4. Maximum clique ---------------------------------------------------------------------------------
def max_clique(G) -> Problem:
    """Maximize the chosen nodes; one row x_u + x_v <= 1 per NON-edge (an independent set of the complement)."""
    nodes, ix = _nodes(G)
    m = _model({}, [1] * len(nodes), len(nodes), "max")
    for u, v in itertools.combinations(nodes, 2):
        if not G.has_edge(u, v):
            m.add_constraint({ix[u]: 1, ix[v]: 1}, hi=1)
    decode = lambda x: {nodes[i] for i in range(len(nodes)) if x[i]}
    check = lambda S: {"valid": all(G.has_edge(u, v) for u, v in itertools.combinations(S, 2)), "score": len(S)}
    return Problem("max_clique", m, decode, check, "nodes in the clique (max)")


# 5. Graph coloring ---------------------------------------------------------------------------------
def coloring(G, k: int) -> Problem:
    """x[v,c] one-hot: exactly one color per node (rows); minimize the edges whose ends share a color."""
    nodes, ix = _nodes(G)
    n = len(nodes) * k
    var = lambda v, c: ix[v] * k + c
    Q = {}
    for u, v in G.edges():
        for c in range(k):
            key = tuple(sorted((var(u, c), var(v, c))))
            Q[key] = Q.get(key, 0) + 1
    m = _model(Q, [0] * n, n, "min")
    for v in nodes:
        m.add_constraint({var(v, c): 1 for c in range(k)}, lo=1, hi=1)

    def decode(x):
        return {v: next((c for c in range(k) if x[var(v, c)]), -1) for v in nodes}

    def check(col):
        ok = all(0 <= col[v] < k for v in nodes)
        return {"valid": ok, "score": sum(1 for u, v in G.edges() if col[u] == col[v]), "colors": k}
    return Problem("coloring", m, decode, check, f"conflicting edges with {k} colors (min; 0 = a proper coloring)",
                   {"k": k})


# 6. Balanced graph partitioning (bisection) --------------------------------------------------------
def bisection(G) -> Problem:
    """Max-Cut turned around: minimize the cut edges, with one row: exactly half the nodes on side 1."""
    nodes, ix = _nodes(G)
    n = len(nodes)
    Q, lin = {}, [0] * n
    for u, v in G.edges():
        i, j = sorted((ix[u], ix[v]))
        lin[i] += 1; lin[j] += 1
        Q[(i, j)] = Q.get((i, j), 0) - 2
    m = _model(Q, lin, n, "min")
    m.add_constraint({i: 1 for i in range(n)}, lo=n // 2, hi=n // 2)
    decode = lambda x: {nodes[i] for i in range(n) if x[i]}
    check = lambda S: {"valid": len(S) == n // 2, "score": sum(1 for u, v in G.edges() if (u in S) != (v in S))}
    return Problem("bisection", m, decode, check, "edges cut between two equal halves (min)")


# 7. Minimum dominating set -------------------------------------------------------------------------
def min_dominating_set(G) -> Problem:
    """Minimize the chosen nodes; per node, it or a neighbor is chosen: x_v + sum_{u in N(v)} x_u >= 1."""
    nodes, ix = _nodes(G)
    m = _model({}, [1] * len(nodes), len(nodes), "min")
    for v in nodes:
        m.add_constraint({ix[u]: 1 for u in [v, *G.neighbors(v)]}, lo=1)
    decode = lambda x: {nodes[i] for i in range(len(nodes)) if x[i]}
    check = lambda S: {"valid": all(v in S or any(u in S for u in G.neighbors(v)) for v in nodes),
                       "score": len(S)}
    return Problem("min_dominating_set", m, decode, check, "nodes in the dominating set (min)")


# 8. Maximum-weight matching ------------------------------------------------------------------------
def max_weight_matching(G) -> Problem:
    """One variable per edge; maximize the chosen weight; per node, at most one chosen edge."""
    edges = list(G.edges())
    m = _model({}, [_w(G, u, v) for u, v in edges], len(edges), "max")
    for v in G.nodes():
        inc = [k for k, e in enumerate(edges) if v in e]
        if len(inc) > 1:
            m.add_constraint({k: 1 for k in inc}, hi=1)
    decode = lambda x: [edges[k] for k in range(len(edges)) if x[k]]
    check = lambda M: {"valid": len({v for e in M for v in e}) == 2 * len(M),
                       "score": sum(_w(G, u, v) for u, v in M)}
    return Problem("max_weight_matching", m, decode, check, "total weight of the matching (max)")


# 9. Minimal maximal matching (Lucas section 4.5) ---------------------------------------------------
def min_maximal_matching(G) -> Problem:
    """The smallest matching that cannot be extended. Rows: at most one chosen edge per node (a matching), and per
    edge, it or an edge touching one of its ends is chosen (maximal). Minimize the chosen edges."""
    edges = list(G.edges())
    m = _model({}, [1] * len(edges), len(edges), "min")
    for v in G.nodes():
        inc = [k for k, e in enumerate(edges) if v in e]
        if len(inc) > 1:
            m.add_constraint({k: 1 for k in inc}, hi=1)
    for k, (u, v) in enumerate(edges):
        m.add_constraint({j: 1 for j, e in enumerate(edges) if u in e or v in e}, lo=1)

    def decode(x):
        return [edges[k] for k in range(len(edges)) if x[k]]

    def check(M):
        covered = {v for e in M for v in e}
        matching = len(covered) == 2 * len(M)
        maximal = all(u in covered or v in covered for u, v in edges)
        return {"valid": matching and maximal, "score": len(M)}
    return Problem("min_maximal_matching", m, decode, check, "edges in a maximal matching (min)")


# 10. Traveling salesperson (small, and honestly the wrong tool at scale) ----------------------------
def tsp(G) -> Problem:
    """x[c,t] = city c at step t. Rows: each city once, each step one city. Objective: the distance between the
    cities at steps t and t+1 (wrapping). n cities need n^2 variables: fine for a toy, the wrong tool for real tours."""
    nodes, ix = _nodes(G)
    n = len(nodes)
    var = lambda c, t: ix[c] * n + t
    Q = {}
    for u, v in itertools.permutations(nodes, 2):
        d = _w(G, u, v)
        for t in range(n):
            key = tuple(sorted((var(u, t), var(v, (t + 1) % n))))
            Q[key] = Q.get(key, 0) + d
    m = _model(Q, [0] * (n * n), n * n, "min")
    for c in nodes:
        m.add_constraint({var(c, t): 1 for t in range(n)}, lo=1, hi=1)
    for t in range(n):
        m.add_constraint({var(c, t): 1 for c in nodes}, lo=1, hi=1)

    def decode(x):
        tour = []
        for t in range(n):
            here = [c for c in nodes if x[var(c, t)]]
            tour.append(here[0] if len(here) == 1 else None)
        return tour

    def check(tour):
        ok = None not in tour and sorted(map(str, tour)) == sorted(map(str, nodes))
        length = sum(_w(G, tour[t], tour[(t + 1) % n]) for t in range(n)) if ok else None
        return {"valid": ok, "score": length}
    return Problem("tsp", m, decode, check, "tour length (min)")


# ---- instances: networkx generators only (no downloaded datasets) ----------------------------------
def weighted(G, seed: int, lo: int = 1, hi: int = 9):
    """Integer edge weights from a seeded generator (integers keep the model exact)."""
    import random

    rnd = random.Random(seed)
    for u, v in G.edges():
        G[u][v]["weight"] = rnd.randint(lo, hi)
    return G


def tsp_instance(n: int, seed: int, scale: int = 100):
    """A complete graph on n random points in a square, integer Euclidean distances."""
    import math
    import random

    import networkx as nx

    rnd = random.Random(seed)
    pts = [(rnd.random() * scale, rnd.random() * scale) for _ in range(n)]
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = max(1, round(math.dist(pts[u], pts[v])))
    return G


def instances(size: str = "default", seed: int = 7) -> List[Tuple[str, Any, dict]]:
    """[(problem name, graph, builder kwargs)] for run.py. `size="small"` gives tiny graphs for offline tests."""
    import networkx as nx

    s = size == "small"
    rr = lambda d, n: nx.random_regular_graph(d, n, seed=seed)
    gnp = lambda n, p: nx.gnp_random_graph(n, p, seed=seed)
    return [
        ("max_cut", rr(3, 8 if s else 100), {}),
        ("max_independent_set", gnp(8 if s else 60, 0.3 if s else 0.08), {}),
        ("min_vertex_cover", gnp(8 if s else 60, 0.3 if s else 0.08), {}),
        ("max_clique", gnp(8 if s else 40, 0.5 if s else 0.3), {}),
        ("coloring", gnp(5 if s else 40, 0.5 if s else 0.15), {"k": 3 if s else 4}),
        ("bisection", rr(3, 8 if s else 60), {}),
        ("min_dominating_set", nx.convert_node_labels_to_integers(nx.grid_2d_graph(3 if s else 7, 3 if s else 7)), {}),
        ("max_weight_matching", weighted(gnp(7 if s else 30, 0.4 if s else 0.15), seed), {}),
        ("min_maximal_matching", gnp(7 if s else 24, 0.35 if s else 0.2), {}),
        ("tsp", tsp_instance(4 if s else 7, seed), {}),
    ]


BUILDERS = {"max_cut": max_cut, "max_independent_set": max_independent_set, "min_vertex_cover": min_vertex_cover,
            "max_clique": max_clique, "coloring": coloring, "bisection": bisection,
            "min_dominating_set": min_dominating_set, "max_weight_matching": max_weight_matching,
            "min_maximal_matching": min_maximal_matching, "tsp": tsp}


def build(name: str, G, **kw) -> Problem:
    return BUILDERS[name](G, **kw)
