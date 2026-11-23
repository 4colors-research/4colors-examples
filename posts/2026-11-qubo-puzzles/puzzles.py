"""Sudoku, N-queens and map coloring as one-hot models with exact rows.

Every puzzle uses the same idea: a 0/1 variable per (thing, choice), and "pick exactly one" written as an exact row
`m.add_constraint({...: 1}, lo=1, hi=1)` instead of a penalty. The decoders turn a returned 0/1 vector back into the
puzzle's answer, and the verifiers check that answer in plain Python with no reference to the model.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import fourcolors.qubo


def _zero_model(n: int):
    """A model with no objective at all: every feasible assignment is a solution."""
    return fourcolors.qubo.to_model({}, [0] * n, n=n)


# ---- Sudoku ---------------------------------------------------------------------------------------

# The puzzle from the Wikipedia article "Sudoku" (0 = blank), widely reproduced.
WIKIPEDIA_PUZZLE = ("530070000600195000098000060800060003400803001700020006060000280000419005000080079")


def parse_grid(text: str) -> List[List[int]]:
    digits = [int(ch) if ch.isdigit() else 0 for ch in text if ch.isdigit() or ch == "."]
    size = int(round(len(digits) ** 0.5))
    assert size * size == len(digits), "a grid needs a square number of cells"
    return [digits[r * size:(r + 1) * size] for r in range(size)]


def box_of(size: int, r: int, c: int) -> int:
    b = int(round(size ** 0.5))
    return (r // b) * b + c // b


def sudoku_model(grid: List[List[int]], eliminate: bool = True):
    """(model, variables) with variables[k] = (row, col, digit).

    eliminate=True keeps only the digits a blank cell can still take given the clues (the usual first step of any
    Sudoku solver), so the clues need no rows. eliminate=False is the textbook model: size^3 variables, the clues
    fixed by rows x = 1."""
    size = len(grid)
    digits = range(1, size + 1)
    given = {(r, c): grid[r][c] for r in range(size) for c in range(size) if grid[r][c]}
    if eliminate:
        used = set()
        for (r, c), d in given.items():
            used |= {("r", r, d), ("c", c, d), ("b", box_of(size, r, c), d)}
        variables = [(r, c, d) for r in range(size) for c in range(size) if (r, c) not in given for d in digits
                     if not {("r", r, d), ("c", c, d), ("b", box_of(size, r, c), d)} & used]
    else:
        variables = [(r, c, d) for r in range(size) for c in range(size) for d in digits]
    m = _zero_model(len(variables))
    groups: Dict[tuple, List[int]] = {}
    for k, (r, c, d) in enumerate(variables):
        for key in (("cell", r, c), ("r", r, d), ("c", c, d), ("b", box_of(size, r, c), d)):
            groups.setdefault(key, []).append(k)
    for key, ks in groups.items():
        m.add_constraint({k: 1 for k in ks}, lo=1, hi=1)              # exactly one
    if not eliminate:
        index = {v: k for k, v in enumerate(variables)}
        for (r, c), d in given.items():
            m.add_constraint({index[(r, c, d)]: 1}, lo=1, hi=1)        # the clue
    return m, variables


def sudoku_decode(grid, variables, x) -> List[List[int]]:
    """The filled grid: clues, plus each blank cell's chosen digit (0 if none, or the last if several)."""
    out = [row[:] for row in grid]
    for (r, c, d), xi in zip(variables, x):
        if xi:
            out[r][c] = d
    return out


def sudoku_valid(grid, filled) -> bool:
    size = len(grid)
    want = set(range(1, size + 1))
    if any(grid[r][c] and grid[r][c] != filled[r][c] for r in range(size) for c in range(size)):
        return False
    rows = all(set(filled[r]) == want for r in range(size))
    cols = all({filled[r][c] for r in range(size)} == want for c in range(size))
    boxes = all({filled[r][c] for r in range(size) for c in range(size) if box_of(size, r, c) == b} == want
                for b in range(size))
    return rows and cols and boxes


def sudoku_backtrack(grid) -> Optional[List[List[int]]]:
    """A plain backtracking solver, the reference answer."""
    size = len(grid)
    g = [row[:] for row in grid]

    def ok(r, c, d):
        if d in g[r] or any(g[i][c] == d for i in range(size)):
            return False
        return not any(g[i][j] == d for i in range(size) for j in range(size)
                       if box_of(size, i, j) == box_of(size, r, c))

    def go(k):
        if k == size * size:
            return True
        r, c = divmod(k, size)
        if g[r][c]:
            return go(k + 1)
        for d in range(1, size + 1):
            if ok(r, c, d):
                g[r][c] = d
                if go(k + 1):
                    return True
        g[r][c] = 0
        return False

    return g if go(0) else None


# ---- N-queens -------------------------------------------------------------------------------------

def nqueens_model(n: int):
    """n^2 variables, x[r*n + c] = 1 for a queen at (r, c). Rows and columns: exactly one. Diagonals: at most one."""
    m = _zero_model(n * n)
    for r in range(n):
        m.add_constraint({r * n + c: 1 for c in range(n)}, lo=1, hi=1)
    for c in range(n):
        m.add_constraint({r * n + c: 1 for r in range(n)}, lo=1, hi=1)
    for s in range(-(n - 2), n - 1):          # r - c = s, every diagonal with at least two squares
        m.add_constraint({r * n + (r - s): 1 for r in range(n) if 0 <= r - s < n}, hi=1)
    for s in range(1, 2 * n - 2):             # r + c = s
        m.add_constraint({r * n + (s - r): 1 for r in range(n) if 0 <= s - r < n}, hi=1)
    return m


def nqueens_decode(n: int, x) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(n) for c in range(n) if x[r * n + c]]


def nqueens_valid(n: int, queens) -> bool:
    if len(queens) != n:
        return False
    rows, cols = {r for r, _ in queens}, {c for _, c in queens}
    d1, d2 = {r - c for r, c in queens}, {r + c for r, c in queens}
    return len(rows) == len(cols) == len(d1) == len(d2) == n


# ---- map coloring ---------------------------------------------------------------------------------

# 48 contiguous states + DC, land borders only (corner contacts such as Four Corners excluded): 49 nodes, 107 edges.
US_ADJ = {
    "AL": "FL GA MS TN", "AZ": "CA NM NV UT", "AR": "LA MO MS OK TN TX", "CA": "AZ NV OR",
    "CO": "KS NE NM OK UT WY", "CT": "MA NY RI", "DC": "MD VA", "DE": "MD NJ PA", "FL": "AL GA",
    "GA": "AL FL NC SC TN", "ID": "MT NV OR UT WA WY", "IL": "IA IN KY MO WI",
    "IN": "IL KY MI OH", "IA": "IL MN MO NE SD WI", "KS": "CO MO NE OK",
    "KY": "IL IN MO OH TN VA WV", "LA": "AR MS TX", "ME": "NH", "MD": "DC DE PA VA WV",
    "MA": "CT NH NY RI VT", "MI": "IN OH WI", "MN": "IA ND SD WI", "MS": "AL AR LA TN",
    "MO": "AR IA IL KS KY NE OK TN", "MT": "ID ND SD WY", "NE": "CO IA KS MO SD WY",
    "NV": "AZ CA ID OR UT", "NH": "MA ME VT", "NJ": "DE NY PA", "NM": "AZ CO OK TX",
    "NY": "CT MA NJ PA VT", "NC": "GA SC TN VA", "ND": "MN MT SD", "OH": "IN KY MI PA WV",
    "OK": "AR CO KS MO NM TX", "OR": "CA ID NV WA", "PA": "DE MD NJ NY OH WV", "RI": "CT MA",
    "SC": "GA NC", "SD": "IA MN MT ND NE WY", "TN": "AL AR GA KY MO MS NC VA",
    "TX": "AR LA NM OK", "UT": "AZ CO ID NV WY", "VT": "MA NH NY", "VA": "DC KY MD NC TN WV",
    "WA": "ID OR", "WV": "KY MD OH PA VA", "WI": "IA IL MI MN", "WY": "CO ID MT NE SD UT",
}

# Approximate geographic centers (lat, lon), for drawing only.
US_LATLON = {
    "AL": (32.8, -86.8), "AZ": (34.3, -111.7), "AR": (34.9, -92.4), "CA": (37.2, -119.5),
    "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DC": (38.6, -76.6), "DE": (39.0, -75.5),
    "FL": (28.6, -82.4), "GA": (32.7, -83.4), "ID": (44.4, -114.6), "IL": (40.0, -89.2),
    "IN": (39.9, -86.3), "IA": (42.1, -93.5), "KS": (38.5, -98.4), "KY": (37.5, -85.3),
    "LA": (31.1, -92.0), "ME": (45.4, -69.2), "MD": (39.3, -77.2), "MA": (42.3, -71.8),
    "MI": (44.3, -85.4), "MN": (46.3, -94.3), "MS": (32.7, -89.7), "MO": (38.4, -92.5),
    "MT": (47.0, -109.6), "NE": (41.5, -99.8), "NV": (39.3, -116.6), "NH": (43.7, -71.6),
    "NJ": (40.2, -74.7), "NM": (34.4, -106.1), "NY": (42.9, -75.5), "NC": (35.6, -79.4),
    "ND": (47.5, -100.5), "OH": (40.3, -82.8), "OK": (35.6, -97.5), "OR": (43.9, -120.6),
    "PA": (40.9, -77.8), "RI": (41.7, -71.5), "SC": (33.9, -80.9), "SD": (44.4, -100.2),
    "TN": (35.9, -86.4), "TX": (31.5, -99.3), "UT": (39.3, -111.7), "VT": (44.1, -72.7),
    "VA": (37.5, -78.9), "WA": (47.4, -120.5), "WV": (38.6, -80.6), "WI": (44.6, -89.9),
    "WY": (43.0, -107.6),
}


def us_edges() -> Tuple[List[str], List[Tuple[str, str]]]:
    nodes = sorted(US_ADJ)
    edges = sorted({tuple(sorted((u, v))) for u, vs in US_ADJ.items() for v in vs.split()})
    return nodes, edges


def coloring_model(nodes, edges, k: int, all_exact: bool = False):
    """x[i*k + c] = 1 when node i gets color c. Each node: exactly one color (exact row).

    all_exact=False: the objective counts same-colored neighbors, sum over edges and colors of x_uc x_vc; its
    minimum is the fewest conflicts possible with k colors (0 when the map is k-colorable).
    all_exact=True: no objective; every edge and color is a row x_uc + x_vc <= 1, so a k-coloring is the only
    feasible answer, and a map that has none ends with reason "no_feasible_found"."""
    pos = {v: i for i, v in enumerate(nodes)}
    Q = {} if all_exact else {(pos[u] * k + c, pos[v] * k + c) if pos[u] < pos[v] else (pos[v] * k + c, pos[u] * k + c): 1
                              for u, v in edges for c in range(k)}
    m = fourcolors.qubo.to_model(Q, [0] * (len(nodes) * k), n=len(nodes) * k)
    for i in range(len(nodes)):
        m.add_constraint({i * k + c: 1 for c in range(k)}, lo=1, hi=1)
    if all_exact:
        for u, v in edges:
            for c in range(k):
                m.add_constraint({pos[u] * k + c: 1, pos[v] * k + c: 1}, hi=1)
    return m


def coloring_decode(nodes, k: int, x) -> Dict[str, int]:
    """node -> color, or -1 when its one-hot row is not exactly one."""
    out = {}
    for i, v in enumerate(nodes):
        chosen = [c for c in range(k) if x[i * k + c]]
        out[v] = chosen[0] if len(chosen) == 1 else -1
    return out


def coloring_conflicts(edges, colors) -> List[Tuple[str, str]]:
    return [(u, v) for u, v in edges if colors[u] >= 0 and colors[u] == colors[v]]


def min_conflicts(nodes, edges, k: int, upper: Optional[int] = None) -> int:
    """The fewest same-colored edges any k-coloring has, by branch and bound (plain Python, exact)."""
    adj = {v: set() for v in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # most-constrained first: a BFS from the highest-degree node keeps neighbors close in the order
    start = max(nodes, key=lambda v: len(adj[v]))
    order, seen = [], {start}
    queue = [start]
    while queue:
        v = queue.pop(0)
        order.append(v)
        for w in sorted(adj[v], key=lambda w: -len(adj[w])):
            if w not in seen:
                seen.add(w)
                queue.append(w)
    order += [v for v in nodes if v not in seen]
    best = [upper if upper is not None else len(edges) + 1]
    color: Dict[str, int] = {}

    def go(i, cost, used):
        if cost >= best[0]:
            return
        if i == len(order):
            best[0] = cost
            return
        v = order[i]
        for c in range(min(k, used + 1)):          # colors are interchangeable: open at most one new one
            extra = sum(1 for w in adj[v] if color.get(w) == c)
            color[v] = c
            go(i + 1, cost + extra, max(used, c + 1))
            del color[v]

    go(0, 0, 0)
    return best[0]
