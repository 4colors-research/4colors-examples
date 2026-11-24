"""The twelve circuits of the longer-compiles post, fixed before any run (see the post).

Every builder is plain Qiskit plus the blog kits, deterministic in its arguments, and returns a gate-level circuit
(no gate on more than two qubits, no high-level objects), so both compilers read exactly the same gate list.
`export(out_dir)` writes each circuit once as OpenQASM 3 and QPY and returns their sha256: both arms read the files.

    python circuits.py --out circuits/          # writes <id>.qasm, <id>.qpy and circuits.json (ids, sizes, sha256)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit  # noqa: E402
import circuitkit as ck  # noqa: E402


def _angles(seed: int, k: int, lo: float = 0.05, hi: float = 0.6) -> List[float]:
    rnd = random.Random(seed)
    return [lo + (hi - lo) * rnd.random() for _ in range(k)]


def gate_level(qc, max_rounds: int = 16):
    """Decompose, one level at a time, every instruction that is not a standard one- or two-qubit gate
    (evolution gates, library boxes, multi-controlled gates), until only BASIC gates are left."""
    for _ in range(max_rounds):
        other = sorted({i.operation.name for i in qc.data
                        if i.operation.name not in BASIC or len(i.qubits) > 2})
        if not other:
            return qc
        qc = qc.decompose(gates_to_decompose=other)
    raise ValueError(f"not gate-level after {max_rounds} rounds: {other}")


#: Gates left as they are: one- and two-qubit standard gates that every compiler reads directly.
BASIC = frozenset({"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz", "p", "u", "u1",
                   "u2", "u3", "cx", "cz", "cy", "cp", "rzz", "rxx", "ryy", "swap", "cry", "crz", "crx", "ch",
                   "cu", "cu1", "cu3", "measure", "barrier", "global_phase"})


# ---- the families ---------------------------------------------------------------------------------------

def qaoa_ba(n: int, m: int = 3, p: int = 3):
    """QAOA Max-Cut, p layers, on a Barabasi-Albert graph (m edges per new node, seed = n). [CHECK m against
    the Benchpress source before the post calls it "Benchpress style".]"""
    return ck.qaoa_maxcut(n, ck.barabasi_albert(n, m, seed=n), p=p)


def qaoa_3reg(n: int, p: int = 3):
    """QAOA Max-Cut, p layers, on a random 3-regular graph (seed = n)."""
    return ck.qaoa_maxcut(n, examplekit.random_regular(n, 3, seed=n), p=p)


def tfim2d(rows: int, cols: int, steps: int = 3, dt: float = 0.1, h: float = 1.0, J: float = 1.0):
    """First-order Trotter steps of the 2D transverse-field Ising model on a rows x cols open grid:
    rzz(2 J dt) on every grid edge, then rx(2 h dt) on every qubit."""
    from qiskit import QuantumCircuit

    n = rows * cols
    edges = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
    edges += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
    qc = QuantumCircuit(n, name=f"tfim2d_{rows}x{cols}")
    for _ in range(steps):
        for a, b in edges:
            qc.rzz(2 * J * dt, a, b)
        qc.rx(2 * h * dt, range(n))
    return qc


def _jw(n: int, p: int, dagger: bool):
    """The Jordan-Wigner image of a_p (or a_p^dagger) on n modes: Z_0 ... Z_{p-1} (X_p +- i Y_p) / 2."""
    from qiskit.quantum_info import SparsePauliOp

    sign = -1j if dagger else 1j
    return SparsePauliOp.from_sparse_list([("Z" * p + "X", list(range(p + 1)), 0.5),
                                           ("Z" * p + "Y", list(range(p + 1)), 0.5 * sign)], num_qubits=n)


def excitation_generator(n: int, creators: Sequence[int], annihilators: Sequence[int]):
    """G = i (T - T^dagger) for T = a^dagger_{c1} ... a_{a1} ...: Hermitian, so exp(-i theta G) is the unitary
    cluster term. Returned as a simplified SparsePauliOp with real coefficients."""
    from qiskit.quantum_info import SparsePauliOp

    T = SparsePauliOp.from_sparse_list([("", [], 1.0)], num_qubits=n)
    for c in creators:
        T = T.dot(_jw(n, c, True))
    for a in annihilators:
        T = T.dot(_jw(n, a, False))
    G = (1j * (T - T.adjoint())).simplify(atol=1e-12)
    return SparsePauliOp(G.paulis, G.coeffs.real)


def uccsd(n_spatial: int = 8, n_alpha: int = 1, n_beta: int = 1, seed: int = 16):
    """A spin-conserving UCCSD-type ansatz (singles and doubles, one first-order Trotter step) on 2 n_spatial
    spin orbitals, Jordan-Wigner mapped: alpha orbitals 0..n_spatial-1, beta orbitals after them, started from
    the Hartree-Fock state. Angles are small and fixed by `seed` (a compile input, not a converged state)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate

    n = 2 * n_spatial
    occ_a, vir_a = list(range(n_alpha)), list(range(n_alpha, n_spatial))
    occ_b = [n_spatial + i for i in range(n_beta)]
    vir_b = [n_spatial + i for i in range(n_beta, n_spatial)]
    terms: List[Tuple[List[int], List[int]]] = []
    for occ, vir in ((occ_a, vir_a), (occ_b, vir_b)):
        terms += [([a], [i]) for i in occ for a in vir]
    for occ, vir in ((occ_a, vir_a), (occ_b, vir_b)):          # same-spin doubles
        terms += [([a, b], [j, i]) for x, i in enumerate(occ) for j in occ[x + 1:]
                  for y, a in enumerate(vir) for b in vir[y + 1:]]
    terms += [([a, b], [j, i]) for i in occ_a for j in occ_b for a in vir_a for b in vir_b]   # mixed-spin doubles
    qc = QuantumCircuit(n, name=f"uccsd_{n}")
    for q in occ_a + occ_b:
        qc.x(q)
    for theta, (cre, ann) in zip(_angles(seed, len(terms), 0.01, 0.1), terms):
        qc.append(PauliEvolutionGate(excitation_generator(n, cre, ann), time=theta), range(n))
    return gate_level(qc)


def hea_full(n: int, reps: int = 2, seed: int = 16):
    """Hardware-efficient ansatz with full entanglement (Qiskit's efficient_su2), parameters bound from `seed`."""
    from qiskit.circuit.library import efficient_su2

    qc = efficient_su2(n, reps=reps, entanglement="full")
    qc = qc.assign_parameters(_angles(seed, qc.num_parameters, -math.pi, math.pi))
    return gate_level(qc)


def zz_kernel(n: int, reps: int = 2, seed: int = 20):
    """A ZZ feature map (Havlicek et al. 2019) with full entanglement, written with rzz so the commuting blocks
    are visible to every compiler: per repetition, h on every qubit, rz(2 x_i), then rzz(2 (pi - x_i)(pi - x_j))
    on every pair. The data point x is fixed by `seed`."""
    from qiskit import QuantumCircuit

    x = _angles(seed, n, 0.0, math.pi)
    qc = QuantumCircuit(n, name=f"zzkernel_{n}")
    for _ in range(reps):
        qc.h(range(n))
        for i in range(n):
            qc.rz(2 * x[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.rzz(2 * (math.pi - x[i]) * (math.pi - x[j]), i, j)
    return qc


def portfolio_xy(n: int, budget: int = None, p: int = 2, seed: int = 20):
    """QAOA for a portfolio model with an XY (ring) mixer that keeps the number of chosen assets fixed: start
    with `budget` ones, cost layer rz on every asset and rzz on every pair (a dense covariance), mixer
    XX+YY on the ring in two alternating layers."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    budget = n // 2 if budget is None else budget
    rnd = random.Random(seed)
    mu = [rnd.uniform(-1, 1) for _ in range(n)]
    cov = {(i, j): rnd.uniform(-0.5, 0.5) for i in range(n) for j in range(i + 1, n)}
    qc = QuantumCircuit(n, name=f"portfolio_xy_{n}")
    for q in range(budget):
        qc.x(q)
    for layer in range(p):
        gamma, beta = 0.4 + 0.1 * layer, 0.6 - 0.1 * layer
        for i in range(n):
            qc.rz(2 * gamma * mu[i], i)
        for (i, j), c in cov.items():
            qc.rzz(2 * gamma * c, i, j)
        for start in (0, 1):
            for i in range(start, n, 2):
                qc.append(XXPlusYYGate(2 * beta), [i, (i + 1) % n])
    return gate_level(qc)


def grover(n: int = 8, marked: int = 181, iterations: int = 2):
    """Grover search on n qubits for one marked bit string, `iterations` rounds (fewer than optimal: a compile
    input, not a search). Multi-controlled gates are unrolled to one- and two-qubit gates without ancillas."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import grover_operator

    oracle = QuantumCircuit(n)
    zeros = [q for q in range(n) if not (marked >> q) & 1]
    if zeros:
        oracle.x(zeros)
    oracle.h(n - 1)
    oracle.mcx(list(range(n - 1)), n - 1)
    oracle.h(n - 1)
    if zeros:
        oracle.x(zeros)
    g = grover_operator(oracle)
    qc = QuantumCircuit(n, name=f"grover_{n}")
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(g, inplace=True)
    return gate_level(qc)


def qae(n_eval: int = 6, n_state: int = 3, seed: int = 9):
    """Canonical (phase-estimation) amplitude estimation: an n_state-qubit preparation A (ry layer and a cry
    chain, angles fixed by `seed`), the Grover operator Q of A with the good state "last qubit is 1", controlled
    Q^(2^k) on evaluation qubit k (as 2^k controlled copies), then the inverse QFT on the evaluation qubits."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import grover_operator
    from qiskit.synthesis import synth_qft_full

    a = QuantumCircuit(n_state, name="A")
    th = _angles(seed, 2 * n_state - 1, 0.2, 1.2)
    for q in range(n_state):
        a.ry(th[q], q)
    for q in range(n_state - 1):
        a.cry(th[n_state + q], q, q + 1)
    oracle = QuantumCircuit(n_state)
    oracle.z(n_state - 1)
    cq = gate_level(grover_operator(oracle, state_preparation=a)).to_gate(label="Q").control(1)
    qc = QuantumCircuit(n_eval + n_state, name=f"qae_{n_eval + n_state}")
    state = list(range(n_eval, n_eval + n_state))
    qc.compose(a, state, inplace=True)
    for k in range(n_eval):
        qc.h(k)
        for _ in range(2 ** k):
            qc.append(cq, [k] + state)
    qc.compose(synth_qft_full(n_eval, inverse=True, do_swaps=True), range(n_eval), inplace=True)
    return gate_level(qc)


#: id -> (family, builder). The post's twelve circuits (see the post). Dev split only.
CIRCUITS: Dict[str, Tuple[str, Callable]] = {
    "qaoa_ba3_27_p3": ("qaoa_barabasi_albert", lambda: qaoa_ba(27)),
    "qaoa_ba3_50_p3": ("qaoa_barabasi_albert", lambda: qaoa_ba(50)),
    "qaoa_ba3_100_p3": ("qaoa_barabasi_albert", lambda: qaoa_ba(100)),
    "qaoa_3reg_50_p3": ("qaoa_3_regular", lambda: qaoa_3reg(50)),
    "tfim2d-n36": ("tfim2d", lambda: tfim2d(6, 6)),
    "uccsd-n16": ("uccsd", lambda: uccsd(8, 1, 1)),
    "hea_full-n16": ("hea_full", lambda: hea_full(16)),
    "zzkernel-n20": ("zz_kernel", lambda: zz_kernel(20)),
    "portfolio_xy-n20": ("portfolio_xy", lambda: portfolio_xy(20)),
    "qft-n32": ("qft", lambda: ck.qft(32)),
    "grover-n8": ("grover", lambda: grover(8)),
    "qae-n9": ("qae", lambda: qae(6, 3)),
}

#: The pilot's circuits (see the post): the same families, other sizes, never the post's circuits.
PILOT: Dict[str, Tuple[str, Callable]] = {
    "qaoa_ba3_40_p3": ("qaoa_barabasi_albert", lambda: qaoa_ba(40)),
    "qaoa_3reg_40_p3": ("qaoa_3_regular", lambda: qaoa_3reg(40)),
    "hea_full-n12": ("hea_full", lambda: hea_full(12)),
    "zzkernel-n16": ("zz_kernel", lambda: zz_kernel(16)),
}


def build(cid: str):
    """(circuit, circuit fields) for one id of CIRCUITS or PILOT."""
    family, fn = {**CIRCUITS, **PILOT}[cid]
    qc = fn()
    qc.name = cid
    return qc, ck.circuit_fields(qc, cid, family)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(out_dir, ids: Sequence[str] = tuple(CIRCUITS)) -> List[dict]:
    """Write <id>.qasm (OpenQASM 3) and <id>.qpy for each id, and circuits.json with sizes and sha256."""
    from qiskit import qasm3, qpy

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in ids:
        qc, fields = build(cid)
        (out / f"{cid}.qasm").write_text(qasm3.dumps(qc))
        with open(out / f"{cid}.qpy", "wb") as fh:
            qpy.dump(qc, fh)
        rows.append({**fields, "qasm": f"{cid}.qasm", "qpy": f"{cid}.qpy",
                     "sha256_qasm": sha256(out / f"{cid}.qasm"), "sha256_qpy": sha256(out / f"{cid}.qpy")})
    (out / "circuits.json").write_text(json.dumps(rows, indent=1))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--ids", default=",".join(CIRCUITS), help="comma list (default: the post's twelve)")
    a = p.parse_args(argv)
    for r in export(a.out, [x for x in a.ids.split(",") if x]):
        print(f"{r['circuit_id']:18s} {r['qubits']:4d} qubits {r['input_2q']:6d} two-qubit gates  {r['sha256_qpy'][:12]}")


if __name__ == "__main__":
    main()
