"""Maximum independent set on King's-lattice unit-disk graphs, the graph family of the Rydberg-atom MIS experiments.

An L x L square grid of sites, each filled with an atom with probability `filling` (seeded). Two atoms are joined
when they are within distance sqrt(2) in lattice units: horizontal, vertical and diagonal neighbors, the King's
graph. An independent set is a set of atoms with no two joined (no two Rydberg excitations inside the blockade
radius).

Two models of the same problem:
- `rows_model`: maximize the number of chosen atoms, one exact row x_u + x_v <= 1 per edge.
- `penalty_model`: the textbook penalty QUBO, minimize -sum x + P * sum_edges x_u x_v; any P > 1 has the same
  optimum on an unweighted graph. Built as a scipy sparse matrix so a million atoms fit in memory.

Checks (plain Python, independent of the solver): `violations` counts edges with both ends chosen; `repair` drops one
end of each violated edge (the usual post-processing); `block_bound` is a certificate: every 2 x 2 block of the
lattice is a clique of the King's graph, so an independent set holds at most one atom per non-empty block.
`exact_mis` is a labeled exact reference for small instances (an integer program solved by HiGHS through scipy).
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

Edge = Tuple[int, int]

FILLING = 0.8                      # the site filling of the published experiments (about 80%)
NEIGHBORS = [(0, 1), (1, -1), (1, 0), (1, 1)]   # each undirected King's-graph edge once


def side_for_atoms(atoms: int, filling: float = FILLING) -> int:
    """The lattice side L whose expected atom count filling * L^2 is closest to `atoms`."""
    return max(2, round(math.sqrt(atoms / filling)))


def lattice(L: int, filling: float = FILLING, seed: int = 1) -> Tuple[List[Tuple[int, int]], List[Edge]]:
    """Atom positions (row, col), indexed 0..n-1 in row-major order, and the King's-graph edges (i < j)."""
    rng = random.Random(seed)
    occupied = [[rng.random() < filling for _ in range(L)] for _ in range(L)]
    index: Dict[Tuple[int, int], int] = {}
    pos: List[Tuple[int, int]] = []
    for r in range(L):
        for c in range(L):
            if occupied[r][c]:
                index[(r, c)] = len(pos)
                pos.append((r, c))
    edges: List[Edge] = []
    for (r, c), i in index.items():
        for dr, dc in NEIGHBORS:
            j = index.get((r + dr, c + dc))
            if j is not None:
                edges.append((min(i, j), max(i, j)))
    edges.sort()
    return pos, edges


def lattice_arrays(L: int, filling: float = FILLING, seed: int = 1):
    """The same instance as `lattice`, as numpy arrays (positions n x 2, edges m x 2), for a million atoms.

    Uses the same random stream as `lattice`, so both give identical instances (tested)."""
    import numpy as np

    rng = random.Random(seed)
    occ = np.fromiter((rng.random() < filling for _ in range(L * L)), dtype=bool, count=L * L).reshape(L, L)
    idx = -np.ones((L, L), dtype=np.int64)
    rr, cc = np.nonzero(occ)
    idx[rr, cc] = np.arange(rr.size)
    pairs = []
    for dr, dc in NEIGHBORS:
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < L) & (c2 >= 0) & (c2 < L)
        a, b = idx[rr[ok], cc[ok]], idx[r2[ok], c2[ok]]
        keep = b >= 0
        pairs.append(np.stack([np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])], axis=1))
    e = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)
    e = e[np.lexsort((e[:, 1], e[:, 0]))]
    return np.stack([rr, cc], axis=1), e


def rows_model(n: int, edges):
    """max sum x  s.t.  x_u + x_v <= 1 for every edge (exact rows; no penalty weight)."""
    import fourcolors.qubo

    m = fourcolors.qubo.to_model({}, [1] * n, n=n, sense="max")
    for u, v in edges:
        m.add_constraint({int(u): 1, int(v): 1}, hi=1)
    return m


def penalty_model(n: int, edges, P: int = 2):
    """min -sum x + P * sum_edges x_u x_v, as a sparse upper-triangular matrix (diagonal = linear for BINARY)."""
    import numpy as np
    import scipy.sparse as sp
    import fourcolors.qubo

    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    rows = np.concatenate([np.arange(n), e[:, 0]])
    cols = np.concatenate([np.arange(n), e[:, 1]])
    vals = np.concatenate([-np.ones(n, dtype=np.int64), np.full(len(e), P, dtype=np.int64)])
    return fourcolors.qubo.to_model(sp.coo_matrix((vals, (rows, cols)), shape=(n, n)), sense="min")


def violations(x: Sequence[int], edges) -> int:
    """Edges with both ends chosen (0 means the set is independent)."""
    return sum(1 for u, v in edges if x[u] and x[v])


def repair(x: Sequence[int], edges) -> List[int]:
    """Drop the higher-indexed end of every violated edge; the result is always independent."""
    y = list(int(v) for v in x)
    for u, v in edges:
        if y[u] and y[v]:
            y[v] = 0
    return y


def block_bound(pos: Sequence[Tuple[int, int]]) -> int:
    """Upper bound on the MIS: the number of non-empty 2 x 2 blocks (each block is a clique of the King's graph)."""
    return len({(int(r) // 2, int(c) // 2) for r, c in pos})


def exact_mis(n: int, edges) -> int:
    """The exact MIS size by integer programming (scipy.optimize.milp, HiGHS). A labeled reference for small n."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    import scipy.sparse as sp

    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(e) == 0:
        return n
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.repeat(np.arange(len(e)), 2), e.ravel())), shape=(len(e), n))
    res = milp(c=-np.ones(n), constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(n),
               bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"milp: {res.message}")
    return int(round(-res.fun))


def model_terms(n: int, n_edges: int) -> Dict[str, int]:
    """Sizes the service sees: variables, quadratic terms (penalty form) or rows (rows form)."""
    return {"n_vars": n, "n_quad_penalty": n_edges, "n_rows_exact": n_edges}
