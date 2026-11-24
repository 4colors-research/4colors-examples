"""Model code for the Pauli-grouping post: Hamiltonians as Pauli terms, the two commutation rules, the conflict
graph, the coloring model with exact one-hot rows, decoding and checking, two classical baselines, the modeled
shot cost, and the diagonalization circuit of a commuting group.

Labels follow Qiskit's convention: a Pauli label is a string over I, X, Y, Z with qubit 0 as the RIGHTMOST
character ("IZ" is Z on qubit 0). A Hamiltonian is a list of (label, real coefficient) pairs, identity removed.

Nothing here calls the service. The coloring model is solved by the 4colors Optimizer for QUBO in run.py; the
baselines (Sorted Insertion, Qiskit's `group_commuting`) are the comparison arms, labeled as such.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Term = Tuple[str, float]

# ---- Hamiltonians ---------------------------------------------------------------------------------


def _sparse(terms: Iterable[Term]):
    from qiskit.quantum_info import SparsePauliOp

    terms = list(terms)
    return SparsePauliOp([t[0] for t in terms], [t[1] for t in terms])


def _clean(op, tol: float = 1e-12) -> List[Term]:
    """Real Pauli terms of a SparsePauliOp, simplified, without identity, sorted by label."""
    op = op.simplify(atol=tol)
    out: Dict[str, float] = {}
    for label, c in zip(op.paulis.to_labels(), op.coeffs):
        if set(label) <= {"I"}:
            continue
        if abs(c.imag) > 1e-9:
            raise ValueError(f"term {label} has an imaginary coefficient {c}: the operator is not Hermitian")
        if abs(c.real) > tol:
            out[label] = out.get(label, 0.0) + float(c.real)
    return sorted(out.items())


def _jw_annihilation(j: int, n: int):
    """a_j under Jordan-Wigner: Z on qubits < j, (X + iY)/2 on qubit j."""
    from qiskit.quantum_info import SparsePauliOp

    def label(p):
        s = ["I"] * n
        for q in range(j):
            s[n - 1 - q] = "Z"
        s[n - 1 - j] = p
        return "".join(s)

    return SparsePauliOp([label("X"), label("Y")], [0.5, 0.5j])


def fermi_hubbard(lx: int, ly: int = 1, t: float = 1.0, u: float = 4.0) -> List[Term]:
    """The Fermi-Hubbard model on an open lx x ly grid, Jordan-Wigner, 2 * lx * ly qubits (spin up on even
    qubits, spin down on odd): -t sum_<ij>,s (a+_is a_js + h.c.) + u sum_i n_i,up n_i,down."""
    sites = lx * ly
    n = 2 * sites
    a = [_jw_annihilation(j, n) for j in range(n)]
    ad = [x.adjoint() for x in a]
    op = None

    def add(x):
        nonlocal op
        op = x if op is None else op + x

    for (i, j) in grid_edges(lx, ly):
        for s in (0, 1):
            p, q = 2 * i + s, 2 * j + s
            add(-t * (ad[p].dot(a[q]) + ad[q].dot(a[p])))
    for i in range(sites):
        add(u * ad[2 * i].dot(a[2 * i]).dot(ad[2 * i + 1].dot(a[2 * i + 1])))  # dot = operator product
    return _clean(op)


def heisenberg(lx: int, ly: int = 1, j: float = 1.0, h: float = 0.0) -> List[Term]:
    """The Heisenberg model on an open lx x ly grid: j sum_<ab> (XX + YY + ZZ) + h sum_a Z, one qubit per site."""
    n = lx * ly
    terms: Dict[str, float] = {}
    for (a, b) in grid_edges(lx, ly):
        for p in "XYZ":
            s = ["I"] * n
            s[n - 1 - a] = s[n - 1 - b] = p
            terms["".join(s)] = terms.get("".join(s), 0.0) + j
    if h:
        for a in range(n):
            s = ["I"] * n
            s[n - 1 - a] = "Z"
            terms["".join(s)] = h
    return sorted(terms.items())


def grid_edges(lx: int, ly: int) -> List[Tuple[int, int]]:
    """Nearest-neighbor pairs of an open lx x ly grid, site index x + lx * y."""
    out = []
    for y in range(ly):
        for x in range(lx):
            i = x + lx * y
            if x + 1 < lx:
                out.append((i, i + 1))
            if y + 1 < ly:
                out.append((i, i + lx))
    return out


_OF_TERM = re.compile(r"\(?\s*([-+0-9.eEj ]+?)\s*\)?\s*\[([^\]]*)\]")


def parse_openfermion(text: str, n_qubits: Optional[int] = None) -> List[Term]:
    """Terms from OpenFermion QubitOperator text, the form HamLib stores ("(0.5+0j) [X0 Y1] +\\n..."). Qubit k
    of a factor like Z3 becomes position k from the right. Imaginary parts must be zero."""
    raw: List[Tuple[complex, List[Tuple[str, int]]]] = []
    for coef, body in _OF_TERM.findall(text):
        c = complex(coef.replace(" ", ""))
        factors = [(f[0], int(f[1:])) for f in body.split()]
        raw.append((c, factors))
    if not raw:
        raise ValueError("no terms found")
    n = n_qubits if n_qubits is not None else 1 + max((q for _, fs in raw for _, q in fs), default=0)
    out: Dict[str, float] = {}
    for c, fs in raw:
        if abs(c.imag) > 1e-9:
            raise ValueError(f"imaginary coefficient {c}")
        s = ["I"] * n
        for p, q in fs:
            s[n - 1 - q] = p
        label = "".join(s)
        if set(label) <= {"I"}:
            continue
        out[label] = out.get(label, 0.0) + c.real
    return sorted((k, v) for k, v in out.items() if v != 0)


# ---- commutation and the conflict graph -----------------------------------------------------------


def symplectic(labels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """(x, z) bit matrices, one row per label, column q = qubit q."""
    m, n = len(labels), len(labels[0]) if labels else 0
    x = np.zeros((m, n), dtype=bool)
    z = np.zeros((m, n), dtype=bool)
    for i, lab in enumerate(labels):
        for pos, p in enumerate(lab):
            q = n - 1 - pos
            x[i, q] = p in "XY"
            z[i, q] = p in "ZY"
    return x, z


def commute_gc(a: str, b: str) -> bool:
    """General commutation: the two Paulis differ (both non-identity, unequal) on an even number of qubits."""
    return sum(1 for p, q in zip(a, b) if p != "I" and q != "I" and p != q) % 2 == 0


def commute_qwc(a: str, b: str) -> bool:
    """Qubit-wise commutation: on every qubit the two letters are equal or one is the identity."""
    return all(p == "I" or q == "I" or p == q for p, q in zip(a, b))


def conflict_edges(labels: Sequence[str], kind: str = "gc", chunk: int = 512) -> List[Tuple[int, int]]:
    """Pairs (i < j) of terms that may NOT share a group under `kind` ("gc" or "qwc"). Vectorized as integer
    matrix products in chunks of rows, so 30,000 terms take seconds to minutes, not hours."""
    if kind not in ("gc", "qwc"):
        raise ValueError("kind is 'gc' or 'qwc'")
    x, z = symplectic(labels)
    m = len(labels)
    if kind == "gc":
        xi, zi = x.astype(np.int32), z.astype(np.int32)
    else:
        nz = (x | z).astype(np.int32)
        letters = [(x & ~z).astype(np.int32), (x & z).astype(np.int32), (~x & z).astype(np.int32)]   # X, Y, Z
    out: List[Tuple[int, int]] = []
    for s in range(0, m, chunk):
        a = slice(s, min(m, s + chunk))
        if kind == "gc":
            bad = ((xi[a] @ zi.T + zi[a] @ xi.T) % 2).astype(bool)      # odd symplectic product: anticommute
        else:
            both = nz[a] @ nz.T                                         # qubits where both act
            same = sum(e[a] @ e.T for e in letters)                     # ... with the same letter
            bad = both != same
        ii, jj = np.nonzero(bad)
        keep = ii + s < jj
        out.extend((int(i), int(j)) for i, j in zip(ii[keep] + s, jj[keep]))
    return out


# ---- the coloring model ---------------------------------------------------------------------------


def var(i: int, c: int, k: int) -> int:
    return i * k + c


def coloring_model(m: int, edges: Sequence[Tuple[int, int]], k: int, fix_first: bool = True):
    """The K-coloring of the conflict graph as a 4colors model: one 0/1 variable per (term, color), an exact row
    "exactly one color" per term, and the objective = the number of conflicting pairs that share a color. An
    answer with objective 0 is a valid grouping into at most k groups. `fix_first` puts term 0 in color 0 (a
    symmetry cut: it changes nothing but the color names)."""
    import fourcolors.qubo

    q: Dict[Tuple[int, int], int] = {}
    for (i, j) in edges:
        for c in range(k):
            a, b = sorted((var(i, c, k), var(j, c, k)))
            q[(a, b)] = q.get((a, b), 0) + 1
    model = fourcolors.qubo.to_model(q, [0] * (m * k), n=m * k)
    for i in range(m):
        model.add_constraint({var(i, c, k): 1 for c in range(k)}, lo=1, hi=1)
    if fix_first and m:
        model.add_constraint({var(0, 0, k): 1}, lo=1, hi=1)
    return model


def decode(x: Sequence[int], m: int, k: int) -> Optional[List[int]]:
    """The color of each term, or None when some term has not exactly one color."""
    colors = []
    for i in range(m):
        on = [c for c in range(k) if x[var(i, c, k)]]
        if len(on) != 1:
            return None
        colors.append(on[0])
    return colors


def groups_from_colors(colors: Sequence[int]) -> List[List[int]]:
    """Term indices per used color, colors renumbered in order of first use; empty colors dropped."""
    by: Dict[int, List[int]] = {}
    for i, c in enumerate(colors):
        by.setdefault(c, []).append(i)
    return [by[c] for c in sorted(by, key=lambda c: by[c][0])]


def n_conflicts(colors: Sequence[int], edges: Sequence[Tuple[int, int]]) -> int:
    return sum(1 for i, j in edges if colors[i] == colors[j])


def valid_grouping(labels: Sequence[str], groups: Sequence[Sequence[int]], kind: str = "gc") -> bool:
    """Every term in exactly one group, and every pair inside a group commutes under `kind` (plain Python)."""
    rule = commute_gc if kind == "gc" else commute_qwc
    seen = sorted(i for g in groups for i in g)
    if seen != list(range(len(labels))):
        return False
    return all(rule(labels[a], labels[b]) for g in groups for x, a in enumerate(g) for b in g[x + 1:])


# ---- baselines ------------------------------------------------------------------------------------


def sorted_insertion(terms: Sequence[Term], kind: str = "gc") -> List[List[int]]:
    """Sorted Insertion (Crawford et al., Quantum 5, 385, 2021): terms by |coefficient|, largest first, each into
    the first group it commutes with entirely, else a new group."""
    rule = commute_gc if kind == "gc" else commute_qwc
    order = sorted(range(len(terms)), key=lambda i: (-abs(terms[i][1]), terms[i][0]))
    groups: List[List[int]] = []
    for i in order:
        for g in groups:
            if all(rule(terms[i][0], terms[j][0]) for j in g):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def qiskit_grouping(terms: Sequence[Term], kind: str = "gc") -> List[List[int]]:
    """Qiskit's `SparsePauliOp.group_commuting(qubit_wise=...)`, mapped back to term indices."""
    index = {lab: i for i, (lab, _) in enumerate(terms)}
    out = []
    for g in _sparse(terms).group_commuting(qubit_wise=(kind == "qwc")):
        out.append(sorted(index[lab] for lab in g.paulis.to_labels()))
    return out


# ---- the modeled measurement cost -----------------------------------------------------------------


def modeled_shot_factor(terms: Sequence[Term], groups: Sequence[Sequence[int]]) -> float:
    """MODELED shots for a fixed precision, relative to measuring every term on its own. Each group's variance is
    modeled as the sum of its squared coefficients (covariances set to zero), shots are split in proportion to
    each group's standard deviation, so the total is (sum_g sqrt(sum_{i in g} c_i^2))^2 / eps^2; the factor
    divides by the same quantity with one group per term, (sum_i |c_i|)^2. Lower is better; it is a model, not a
    measurement, and it ignores the state (the post says so)."""
    grouped = sum(math.sqrt(sum(terms[i][1] ** 2 for i in g)) for g in groups) ** 2
    alone = sum(abs(c) for _, c in terms) ** 2
    return grouped / alone if alone else float("nan")


def shot_shares(terms: Sequence[Term], groups: Sequence[Sequence[int]]) -> List[float]:
    """Each group's modeled share of the shots (proportional to its modeled standard deviation; sums to 1)."""
    sd = [math.sqrt(sum(terms[i][1] ** 2 for i in g)) for g in groups]
    tot = sum(sd)
    return [s / tot for s in sd] if tot else [1.0 / len(groups)] * len(groups)


def priced_cost(terms: Sequence[Term], groups: Sequence[Sequence[int]], two_qubit: Sequence[Optional[int]]) -> \
        Optional[float]:
    """MODELED two-qubit gates per shot, averaged over the shot shares: sum_g share_g * 2q_g, where 2q_g is the
    compiled two-qubit count of group g's diagonalization circuit (None if any group is unpriced)."""
    if any(v is None for v in two_qubit):
        return None
    return sum(s * v for s, v in zip(shot_shares(terms, groups), two_qubit))


# ---- diagonalization circuits ---------------------------------------------------------------------


def independent(labels: Sequence[str]) -> List[str]:
    """A maximal independent subset (over GF(2), signs ignored) of the Paulis, in input order."""
    x, z = symplectic(labels)
    rows = np.concatenate([x, z], axis=1).astype(np.uint8)
    basis: List[np.ndarray] = []
    pivots: List[int] = []
    keep: List[str] = []
    for lab, r in zip(labels, rows):
        v = r.copy()
        for b, p in zip(basis, pivots):
            if v[p]:
                v ^= b
        nz = np.flatnonzero(v)
        if nz.size:
            basis.append(v)
            pivots.append(int(nz[0]))
            keep.append(lab)
    return keep


def diagonalization_circuit(labels: Sequence[str]):
    """A Clifford circuit U (no measurements) with U P U^dagger = +-(a Z-string) for every P in `labels`, which must
    commute pairwise. Built from the stabilizer state of an independent subset: its Clifford C prepares a state
    stabilized by them, so C^dagger maps each to a Z-string. Measure every qubit after U and each term's value is
    the parity of its Z-string's bits."""
    from qiskit.quantum_info import StabilizerState

    if not all(commute_gc(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]):
        raise ValueError("the group does not commute")
    gens = independent(labels)
    state = StabilizerState.from_stabilizer_list(gens, allow_redundant=False, allow_underconstrained=True)
    return state.clifford.adjoint().to_circuit()


def diagonalizes(circuit, labels: Sequence[str]) -> bool:
    """Plain check: every Pauli of `labels`, conjugated by `circuit` (a Clifford), has no X or Y left."""
    from qiskit.quantum_info import Clifford, Pauli

    cliff = Clifford(circuit)
    return all(not Pauli(lab).evolve(cliff, frame="s").x.any() for lab in labels)


def needs_entangling(labels: Sequence[str]) -> bool:
    """False when the group is qubit-wise commuting (one-qubit basis changes suffice, no compile needed)."""
    return not all(commute_qwc(a, b) for i, a in enumerate(labels) for b in labels[i + 1:])
