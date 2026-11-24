"""Parametric ansatze for the parametric-circuit posts, named by a short spec string.

    hea-12x4         hardware-efficient: 12 qubits, 4 layers of ry+rz on every qubit, then cz on a line
    hea-ring-120x2   the same with cz on a ring (the closing pair needs routing on any chip)
    qaoa-16-p2       standard QAOA, p = 2, on a random 3-regular graph with 16 nodes (graph seed 1)
    ma-16            multi-angle QAOA p = 1 (one angle per edge and per node) on the same graph
    uccsd-8-4        UCCSD-style: 8 spin orbitals, 4 occupied; every single and double excitation, Jordan-Wigner

Every builder returns the circuit in INPUT_BASIS (core.py) with its parameters intact and no measurements: the
same logical circuit goes to both compilers.
"""
import itertools

import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from . import core


def reg3_instance(n, seed=1):
    g = nx.random_regular_graph(3, n, seed=seed)
    return dict(id=f"reg3-n{n}-s{seed}", kind="maxcut", nodes=n,
                edges=[(int(a), int(b), 1.0) for a, b in sorted(g.edges)])


def to_input(qc):
    """INPUT_BASIS, optimization level 0: parameters kept, nothing simplified."""
    return transpile(qc, basis_gates=core.INPUT_BASIS, optimization_level=0)


def hea(n, layers, ring=False):
    th = ParameterVector("θ", 2 * n * layers)
    qc = QuantumCircuit(n)
    k = 0
    for _ in range(layers):
        for q in range(n):
            qc.ry(th[k], q)
            qc.rz(th[k + 1], q)
            k += 2
        for q in range(n - 1):
            qc.cz(q, q + 1)
        if ring and n > 2:
            qc.cz(n - 1, 0)
    return qc


def _jw(n, ops):
    """Pauli string (Qiskit order, qubit 0 rightmost) with Z on every qubit strictly between consecutive pairs."""
    label = ["I"] * n
    for q, p in ops:
        label[n - 1 - q] = p
    return "".join(label)


def _excitation(n, occ, vir):
    """Anti-Hermitian (single or double) excitation generator, as a sum of commuting Pauli strings with
    Jordan-Wigner Z strings: returns a SparsePauliOp G with exp(-i theta G) the excitation's unitary."""
    if len(occ) == 1:
        (i,), (a,) = occ, vir
        z = [(q, "Z") for q in range(i + 1, a)]
        return SparsePauliOp([_jw(n, [(i, "X"), (a, "Y")] + z), _jw(n, [(i, "Y"), (a, "X")] + z)], [0.5, -0.5])
    (i, j), (a, b) = occ, vir
    z = [(q, "Z") for q in range(i + 1, j)] + [(q, "Z") for q in range(a + 1, b)]
    terms, coeffs = [], []
    signs = {"XXXY": 1, "XXYX": 1, "XYYY": 1, "YXYY": 1, "XYXX": -1, "YXXX": -1, "YYXY": -1, "YYYX": -1}
    for word, s in signs.items():
        terms.append(_jw(n, [(i, word[0]), (j, word[1]), (a, word[2]), (b, word[3])] + z))
        coeffs.append(s / 8)
    return SparsePauliOp(terms, coeffs)


def uccsd_style(n_orb, n_occ):
    """Hartree-Fock reference (X on the occupied orbitals), then exp(-i theta_k G_k) for every single and double
    excitation from occupied to virtual orbitals. A chemistry-shaped parametric circuit, not a molecule."""
    occ, vir = range(n_occ), range(n_occ, n_orb)
    gens = [((i,), (a,)) for i in occ for a in vir]
    gens += [(o, v) for o in itertools.combinations(occ, 2) for v in itertools.combinations(vir, 2)]
    th = ParameterVector("θ", len(gens))
    qc = QuantumCircuit(n_orb)
    for i in occ:
        qc.x(i)
    for k, (o, v) in enumerate(gens):
        qc.append(PauliEvolutionGate(_excitation(n_orb, o, v), time=th[k]), range(n_orb))
    return qc


def build(spec):
    """(input circuit, instance or None) for a spec string (see the module docstring)."""
    kind, rest = spec.split("-", 1)
    if kind == "hea":
        ring = rest.startswith("ring-")
        n, layers = map(int, rest.replace("ring-", "").split("x"))
        return to_input(hea(n, layers, ring)), None
    if kind == "qaoa":
        n, p = rest.split("-p")
        inst = reg3_instance(int(n))
        return to_input(core.build(inst, f"std-p{int(p)}")[0]), inst
    if kind == "ma":
        inst = reg3_instance(int(rest))
        return to_input(core.build(inst, "ma-p1")[0]), inst
    if kind == "uccsd":
        n, occ = map(int, rest.split("-"))
        return to_input(uccsd_style(n, occ)), None
    raise ValueError(spec)
