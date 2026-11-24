"""Model code for the qLDPC post: bivariate bicycle (BB) codes, GF(2) tools, logical operators, the distance model
with exact parity rows, the plain-Python certificate, the Tanner-graph target, one syndrome-extraction round, and a
stabilizer-simulation check of a compiled round.

A BB code (Bravyi et al., Nature 627, 778, 2024) is a CSS code with H_X = [A | B] and H_Z = [B^T | A^T], where A and
B are sums of three monomials in x = S_l (x) I_m and y = I_l (x) S_m, S the cyclic shift. The parameters of the
codes in `BB` are checked by the unit tests (n and k computed here; the distances are the paper's, cited, never
computed by us).

Nothing here calls the service. The distance model is solved by the 4colors Optimizer for QUBO in run.py and the
syndrome round is compiled by the 4colors Quantum Compiler there.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# name: (l, m, A monomials, B monomials, the distance the paper reports). Monomial (a, b) = x^a y^b.
# [OD-12 legal tone check] on naming the codes and their source in the post.
BB: Dict[str, Tuple[int, int, list, list, Optional[int]]] = {
    "bb_72_12_6": (6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], 6),
    "bb_90_8_10": (15, 3, [(9, 0), (0, 1), (0, 2)], [(0, 0), (2, 0), (7, 0)], 10),
    "bb_108_8_10": (9, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], 10),
    "bb_144_12_12": (12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], 12),
    "bb_288_12_18": (12, 12, [(3, 0), (0, 2), (0, 7)], [(0, 3), (1, 0), (2, 0)], 18),
}
# A = 1 + x, B = 1 + y on a 3 x 3 torus: the toric code [[18, 2, 3]], small enough to check by enumeration.
TORIC_3 = (3, 3, [(0, 0), (1, 0)], [(0, 0), (0, 1)], 3)


# ---- GF(2) ----------------------------------------------------------------------------------------


def rref(M: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Reduced row echelon form over GF(2) and the pivot columns."""
    M = (np.asarray(M) % 2).astype(np.uint8).copy()
    rows, cols = M.shape
    r = 0
    piv: List[int] = []
    for c in range(cols):
        hit = np.flatnonzero(M[r:, c])
        if hit.size == 0:
            continue
        p = r + int(hit[0])
        if p != r:
            M[[r, p]] = M[[p, r]]
        others = np.flatnonzero(M[:, c])
        others = others[others != r]
        M[others] ^= M[r]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return M[:r], piv


def rank(M: np.ndarray) -> int:
    return len(rref(M)[1])


def nullspace(M: np.ndarray) -> np.ndarray:
    """A basis (rows) of {v : M v = 0 mod 2}."""
    R, piv = rref(M)
    n = np.asarray(M).shape[1]
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for i, p in enumerate(piv):
            v[p] = R[i, f]
        basis.append(v)
    return np.array(basis, dtype=np.uint8).reshape(len(basis), n)


def in_rowspace(M: np.ndarray, v: np.ndarray) -> bool:
    return rank(np.vstack([M, v])) == rank(M)


# ---- codes ----------------------------------------------------------------------------------------


def _shift(n: int) -> np.ndarray:
    return np.roll(np.eye(n, dtype=np.uint8), 1, axis=1)


def bb_matrices(l: int, m: int, a_terms, b_terms) -> Tuple[np.ndarray, np.ndarray]:
    """(H_X, H_Z) of the BB code with A = sum x^a y^b over a_terms, B likewise."""
    x = np.kron(_shift(l), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), _shift(m))

    def poly(terms):
        out = np.zeros((l * m, l * m), dtype=np.int64)
        for a, b in terms:
            out += np.linalg.matrix_power(x.astype(np.int64), a) @ np.linalg.matrix_power(y.astype(np.int64), b)
        return (out % 2).astype(np.uint8)

    A, B = poly(a_terms), poly(b_terms)
    return np.hstack([A, B]), np.hstack([B.T, A.T])


def code(name: str):
    """(H_X, H_Z, the paper's distance) for a name in BB, or "toric_3"."""
    l, m, a, b, d = TORIC_3 if name == "toric_3" else BB[name]
    hx, hz = bb_matrices(l, m, a, b)
    return hx, hz, d


def is_css(hx: np.ndarray, hz: np.ndarray) -> bool:
    return not ((hx.astype(np.int64) @ hz.T.astype(np.int64)) % 2).any()


def n_logical(hx: np.ndarray, hz: np.ndarray) -> int:
    return hx.shape[1] - rank(hx) - rank(hz)


def x_logicals(hx: np.ndarray, hz: np.ndarray) -> np.ndarray:
    """k X-type logical operators: vectors in ker(H_Z) independent modulo the row space of H_X."""
    k = n_logical(hx, hz)
    span = hx.copy()
    out = []
    for v in nullspace(hz):
        if not in_rowspace(span, v):
            out.append(v)
            span = np.vstack([span, v])
        if len(out) == k:
            break
    return np.array(out, dtype=np.uint8)


# ---- the distance model ---------------------------------------------------------------------------


def slack_bits(max_even: int) -> int:
    """Bits needed for an integer s with 0 <= 2 s <= max_even."""
    return max(1, math.ceil(math.log2(max_even // 2 + 1))) if max_even >= 2 else 0


def distance_layout(hx: np.ndarray, logical: np.ndarray):
    """Variable layout: e_0..e_{n-1}, then the slack bits of each check row, then those of the logical row.
    Returns (n_vars, [(row support, slack indices)], (logical support, slack indices))."""
    n = hx.shape[1]
    nxt = n
    rows = []
    for r in hx:
        sup = [int(j) for j in np.flatnonzero(r)]
        b = slack_bits(len(sup))
        rows.append((sup, list(range(nxt, nxt + b))))
        nxt += b
    lsup = [int(j) for j in np.flatnonzero(logical)]
    b = slack_bits(len(lsup) - 1)
    lrow = (lsup, list(range(nxt, nxt + b)))
    nxt += b
    return nxt, rows, lrow


def distance_model(hx: np.ndarray, logical: np.ndarray):
    """The minimum-weight Z-type error that commutes with every X check and anticommutes with `logical`, as a 4colors
    model: minimize sum_j e_j subject to, exactly,
        sum_{j in row} e_j - 2 sum_t 2^t s_t = 0    for every row of H_X   (even overlap: commutes)
        sum_{j in l}   e_j - 2 sum_t 2^t s_t = 1    for the logical l      (odd overlap: anticommutes)
    Any answer meeting every row is a nontrivial Z logical, so its weight is an upper bound on d_Z (the usual
    integer-programming formulation of the distance, written as exact rows)."""
    import fourcolors.qubo

    n = hx.shape[1]
    n_vars, rows, (lsup, lslack) = distance_layout(hx, logical)
    lin = [1] * n + [0] * (n_vars - n)
    model = fourcolors.qubo.to_model({}, lin, n=n_vars)
    for sup, slack in rows:
        terms = {j: 1 for j in sup}
        terms.update({s: -2 * 2 ** t for t, s in enumerate(slack)})
        model.add_constraint(terms, lo=0, hi=0)
    terms = {j: 1 for j in lsup}
    terms.update({s: -2 * 2 ** t for t, s in enumerate(lslack)})
    model.add_constraint(terms, lo=1, hi=1)
    return model


def error_of(x: Sequence[int], n: int) -> np.ndarray:
    return np.asarray(list(x[:n]), dtype=np.uint8)


def certify(hx: np.ndarray, logical: np.ndarray, e: np.ndarray) -> bool:
    """Plain check: e commutes with every X check (H_X e = 0 mod 2) and anticommutes with `logical`; then e is a
    nontrivial Z logical and |e| is an upper bound on the code's Z distance."""
    e = np.asarray(e, dtype=np.int64)
    return not ((hx.astype(np.int64) @ e) % 2).any() and int(logical.astype(np.int64) @ e) % 2 == 1


def brute_force_distance(hx: np.ndarray, hz: np.ndarray) -> int:
    """The Z distance by enumeration (small codes only, n <= 20): the least weight of e in ker(H_X) outside the row
    space of H_Z."""
    n = hx.shape[1]
    if n > 20:
        raise ValueError("enumeration is for small codes only")
    ker = nullspace(hx)
    best = n + 1
    for bits in range(1, 2 ** len(ker)):
        coef = np.array([(bits >> i) & 1 for i in range(len(ker))], dtype=np.uint8)
        e = (coef @ ker) % 2
        w = int(e.sum())
        if 0 < w < best and not in_rowspace(hz, e):
            best = w
    return best


# ---- the chip and the circuit ---------------------------------------------------------------------


def qubit_roles(hx: np.ndarray, hz: np.ndarray) -> Dict[str, List[int]]:
    """Data qubits 0..n-1, then one ancilla per X check, then one per Z check."""
    n, rx, rz = hx.shape[1], hx.shape[0], hz.shape[0]
    return {"data": list(range(n)), "x_checks": list(range(n, n + rx)), "z_checks": list(range(n + rx, n + rx + rz))}


def tanner_couplers(hx: np.ndarray, hz: np.ndarray) -> List[Tuple[int, int]]:
    """The couplers of the code's own chip: one per (check, data) incidence of the Tanner graph."""
    roles = qubit_roles(hx, hz)
    out = []
    for H, anc in ((hx, roles["x_checks"]), (hz, roles["z_checks"])):
        for i, row in enumerate(H):
            out += [(int(j), anc[i]) for j in np.flatnonzero(row)]
    return sorted(out)


def tanner_description(hx: np.ndarray, hz: np.ndarray):
    """The Tanner-graph chip as a target description (format version 2, cz, the default one-qubit set),
    built with the client's own `targets.build_description`. No calibration: a design, not a device."""
    from fourcolors.quantum import targets

    n = hx.shape[1] + hx.shape[0] + hz.shape[0]
    return targets.build_description(n, "cz", targets.DEFAULT_1Q, tanner_couplers(hx, hz))


def syndrome_round(hx: np.ndarray, hz: np.ndarray, measure: bool = True):
    """One syndrome-extraction round: every Z check (CNOT data -> ancilla), then every X check (H, CNOT ancilla ->
    data, H), the CNOTs of each check in support order; measurements at the end, X checks into clbits 0..rx-1,
    Z checks into rx..rx+rz-1. A plain sequential schedule, correct for any CSS code; it is NOT the interleaved
    depth-8 schedule of Bravyi et al. (the post says so)."""
    from qiskit import QuantumCircuit

    roles = qubit_roles(hx, hz)
    n_q = len(roles["data"]) + len(roles["x_checks"]) + len(roles["z_checks"])
    rx, rz = hx.shape[0], hz.shape[0]
    qc = QuantumCircuit(n_q, rx + rz if measure else 0)
    # layer t of a check is its t-th support element: each layer touches every data qubit at most once per kind
    for i, row in enumerate(hz):
        for j in np.flatnonzero(row):
            qc.cx(int(j), roles["z_checks"][i])
    for a in roles["x_checks"]:
        qc.h(a)
    for i, row in enumerate(hx):
        for j in np.flatnonzero(row):
            qc.cx(roles["x_checks"][i], int(j))
    for a in roles["x_checks"]:
        qc.h(a)
    if measure:
        for i, a in enumerate(roles["x_checks"]):
            qc.measure(a, i)
        for i, a in enumerate(roles["z_checks"]):
            qc.measure(a, rx + i)
    return qc


# ---- checking a compiled round by Pauli propagation -----------------------------------------------
#
# A compiler may re-synthesize the round into rotations that are Clifford only as a whole (Qiskit O3 moves rz
# rotations through CZs), so a gate-by-gate stabilizer simulation cannot read its output. Instead each measured
# qubit's Z is propagated backwards through the compiled gates (Heisenberg picture, O -> U^dagger O U) as a sum of
# Pauli strings: exact for any one- and two-qubit gates, and small in practice because the non-Clifford pieces
# cancel. Its expectation in a product input state gives the probability of each syndrome bit.

_LETTERS = None


def _letters():
    global _LETTERS
    if _LETTERS is None:
        _LETTERS = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    return _LETTERS


def _transfer(mat: np.ndarray, k: int) -> Dict[int, List[Tuple[int, float]]]:
    """U^dagger P U in the Pauli basis, for every Pauli P on k qubits (index l0 + 4 l1, l0 on the gate's first
    qubit): {input: [(output, coefficient)]}."""
    L = _letters()
    basis = L if k == 1 else [np.kron(L[i // 4], L[i % 4]) for i in range(16)]
    dim = 2 ** k
    out = {}
    for i, P in enumerate(basis):
        Q = mat.conj().T @ P @ mat
        terms = [(j, float(np.real(np.trace(B.conj().T @ Q)) / dim)) for j, B in enumerate(basis)]
        out[i] = [(j, c) for j, c in terms if abs(c) > 1e-10]
    return out


def _letter(x: int, z: int, q: int) -> int:
    bx, bz = (x >> q) & 1, (z >> q) & 1
    return {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}[(bx, bz)]


def _bits(letter: int) -> Tuple[int, int]:
    return [(0, 0), (1, 0), (1, 1), (0, 1)][letter]


def measured_observables(compiled, max_terms: int = 4096) -> Dict[int, Dict[Tuple[int, int], float]]:
    """{clbit: the measured qubit's Z, propagated back to the start of the circuit, as {(x mask, z mask): coef}}.
    Raises ValueError for a 3+ qubit gate, a reset, a qubit used after its measurement, or more than `max_terms`
    Pauli strings (then the check is not run)."""
    from qiskit.quantum_info import Operator

    gates = []
    meas: Dict[int, Tuple[int, int]] = {}          # clbit -> (instruction index, qubit)
    done = set()
    for idx, inst in enumerate(compiled.data):
        op = inst.operation
        qs = [compiled.find_bit(q).index for q in inst.qubits]
        if op.name in ("barrier", "delay"):
            continue
        if any(q in done for q in qs):
            raise ValueError("a qubit is used after its measurement")
        if op.name == "measure":
            meas[compiled.find_bit(inst.clbits[0]).index] = (len(gates), qs[0])
            done.add(qs[0])
            continue
        if len(qs) > 2 or op.name == "reset":
            raise ValueError(f"unsupported instruction {op.name} on {len(qs)} qubits")
        gates.append((op, qs))
    cache: Dict[tuple, dict] = {}
    obs = {}
    for clbit, (stop, q) in meas.items():
        terms: Dict[Tuple[int, int], float] = {(0, 1 << q): 1.0}
        for op, qs in reversed(gates[:stop]):
            if not any(((x | z) >> p) & 1 for (x, z) in terms for p in qs):
                continue
            key = (op.name, tuple(float(p) for p in op.params), len(qs))
            if key not in cache:
                cache[key] = _transfer(Operator(op).data, len(qs))
            tr = cache[key]
            new: Dict[Tuple[int, int], float] = {}
            for (x, z), c in terms.items():
                idx = sum(_letter(x, z, p) * 4 ** i for i, p in enumerate(qs))
                if idx == 0:
                    new[(x, z)] = new.get((x, z), 0.0) + c
                    continue
                mask = sum(1 << p for p in qs)
                bx, bz = x & ~mask, z & ~mask
                for j, cj in tr[idx]:
                    nx, nz = bx, bz
                    for i, p in enumerate(qs):
                        lx, lz = _bits((j // 4 ** i) % 4)
                        nx |= lx << p
                        nz |= lz << p
                    new[(nx, nz)] = new.get((nx, nz), 0.0) + c * cj
            terms = {k: v for k, v in new.items() if abs(v) > 1e-10}
            if len(terms) > max_terms:
                raise ValueError(f"more than {max_terms} Pauli strings")
        obs[clbit] = terms
    return obs


def expectation(terms: Dict[Tuple[int, int], float], state: Dict[int, str]) -> float:
    """<psi| O |psi> for a product state: `state` maps qubit -> "0", "1", "+" or "-" (missing qubits are "0")."""
    value = {"0": {3: 1, 1: 0, 2: 0}, "1": {3: -1, 1: 0, 2: 0}, "+": {1: 1, 2: 0, 3: 0}, "-": {1: -1, 2: 0, 3: 0}}
    tot = 0.0
    for (x, z), c in terms.items():
        v = c
        m = x | z
        q = 0
        while m and v:
            if m & 1:
                v *= value[state.get(q, "0")][_letter(x, z, q)]
            m >>= 1
            q += 1
        tot += v
    return tot


def initial_positions(compiled, n_virtual: int) -> List[int]:
    """Physical qubit of each input qubit at the start: from the compiled circuit's layout (Qiskit sets it, and so
    does `CompileResult.to_qiskit`), else the identity."""
    lay = getattr(compiled, "layout", None)
    if lay is None:
        return list(range(n_virtual))
    return list(lay.initial_index_layout(filter_ancillas=True))[:n_virtual]


def syndrome_check(compiled, hx: np.ndarray, hz: np.ndarray, n_errors: int = 4, seed: int = 1) -> dict:
    """Inject data errors and compare the deterministic syndrome bits of the compiled round with H e: X errors on
    |0...0> data must give exactly the Z-check bits H_Z e (with certainty); Z errors on |+...+> data exactly the
    X-check bits H_X e. Returns {"syndrome_check": "pass" | "fail" | "not_checked", "detail": ...}."""
    n = hx.shape[1]
    rx, rz = hx.shape[0], hz.shape[0]
    try:
        obs = measured_observables(compiled)
    except ValueError as exc:
        return {"syndrome_check": "not_checked", "detail": str(exc)}
    if sorted(obs) != list(range(rx + rz)):
        return {"syndrome_check": "not_checked", "detail": "measurements do not cover every check"}
    pos = initial_positions(compiled, n + rx + rz)
    rng = np.random.default_rng(seed)
    for trial in range(n_errors):
        for kind in ("X", "Z"):
            e = np.zeros(n, dtype=np.uint8)
            e[rng.choice(n, size=1 + trial % 3, replace=False)] = 1
            if kind == "X":
                state = {pos[j]: ("1" if e[j] else "0") for j in range(n)}
                bits, expect = range(rx, rx + rz), (hz.astype(np.int64) @ e) % 2
            else:
                state = {pos[j]: ("-" if e[j] else "+") for j in range(n)}
                bits, expect = range(rx), (hx.astype(np.int64) @ e) % 2
            for b, want in zip(bits, expect):
                ev = expectation(obs[b], state)
                if abs(ev - (1 - 2 * int(want))) > 1e-6:
                    return {"syndrome_check": "fail", "detail": f"{kind} error trial {trial}: clbit {b} <Z>={ev:.3f}"}
    return {"syndrome_check": "pass", "detail": f"{2 * n_errors} injected errors"}
