"""The 2D transverse-field Ising model on an OPEN rows x cols square grid, as Trotter steps, and its structure.

    H = -J sum_<ij> Z_i Z_j - h sum_i X_i        (nearest neighbors of the grid, no wrap-around)

One first-order Trotter step is exp(-i dt J Z_i Z_j) on every bond (written CX, RZ(2 dt), CX, the textbook form)
followed by RX(2 h dt) on every site. This is the same construction as the `tfim2d` family of the compiler report
card (compiler-report-card/families.py, square grids, 2 steps), so its rows can be read against this post's.

Structure the compile can use:
* every ZZ term of a step commutes with every other, so their order is free;
* the bonds of a grid split into 4 matchings (horizontal even/odd columns, vertical even/odd rows): within one
  matching no qubit is used twice, so a matching is one layer of two-qubit gates;
* a bulk site has 4 neighbors, a heavy-hex qubit at most 3, so the grid cannot sit on the chip as it is: some
  qubits must move between matchings.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

Bond = Tuple[int, int]


def parse_grid(text: str) -> Tuple[int, int]:
    """'7x8' -> (7, 8); '6' -> (6, 6)."""
    if "x" in text:
        r, c = text.lower().split("x")
        return int(r), int(c)
    return int(text), int(text)


def site(r: int, c: int, cols: int) -> int:
    return r * cols + c


def bonds(rows: int, cols: int) -> List[Bond]:
    """Nearest-neighbor bonds of the open grid, horizontal first (as the report card's generator orders them)."""
    h = [(site(r, c, cols), site(r, c + 1, cols)) for r in range(rows) for c in range(cols - 1)]
    v = [(site(r, c, cols), site(r + 1, c, cols)) for r in range(rows - 1) for c in range(cols)]
    return h + v


def matchings(rows: int, cols: int) -> Dict[str, List[Bond]]:
    """The 4 matchings that partition the bonds (a proper edge coloring of the grid with 4 colors)."""
    out = {"h_even": [], "h_odd": [], "v_even": [], "v_odd": []}
    for r in range(rows):
        for c in range(cols - 1):
            out["h_even" if c % 2 == 0 else "h_odd"].append((site(r, c, cols), site(r, c + 1, cols)))
    for r in range(rows - 1):
        for c in range(cols):
            out["v_even" if r % 2 == 0 else "v_odd"].append((site(r, c, cols), site(r + 1, c, cols)))
    return out


def degrees(rows: int, cols: int) -> Dict[int, int]:
    d = {q: 0 for q in range(rows * cols)}
    for a, b in bonds(rows, cols):
        d[a] += 1; d[b] += 1
    return d


def circuit(rows: int, cols: int, steps: int = 1, dt: float = 0.2, h: float = 1.0, measure: bool = True):
    """`steps` Trotter steps from |0...0> (all spins up, the ordered state a quench starts from), measured."""
    from qiskit import QuantumCircuit

    n = rows * cols
    qc = QuantumCircuit(n, name=f"tfim2d_{rows}x{cols}_s{steps}")
    for _ in range(steps):
        for u, v in bonds(rows, cols):
            qc.cx(u, v); qc.rz(2 * dt, v); qc.cx(u, v)
        qc.rx(2 * h * dt, range(n))
    if measure:
        qc.measure_all()
    return qc


def logical_floor(rows: int, cols: int, steps: int = 1) -> int:
    """Two-qubit gates per circuit with no routing at all: 2 per bond per step (one ZZ rotation is 2 CZ)."""
    return 2 * len(bonds(rows, cols)) * steps


def magnetization_z(counts: Dict[str, int]) -> float:
    """<Z> averaged over sites from measured counts (bit 1 = spin down). A helper for readers who run the
    circuit; this post reports gate counts only."""
    shots = sum(counts.values())
    if not shots:
        return 0.0
    n = len(next(iter(counts)).replace(" ", ""))
    tot = 0.0
    for bits, k in counts.items():
        b = bits.replace(" ", "")
        tot += k * sum(1 - 2 * int(x) for x in b) / n
    return tot / shots
