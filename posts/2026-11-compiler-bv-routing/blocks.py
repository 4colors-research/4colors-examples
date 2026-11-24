"""Read a compiled circuit block by block: which two-qubit blocks are routing, and which do logical work too.

Take the two-qubit gates that act on one pair of qubits back to back (with the single-qubit gates between them):
that is a block, and its 4x4 matrix has Weyl coordinates (a, b, c) that do not change under single-qubit gates
(Zhang et al. 2003; Shende, Markov, Bullock 2004). They name the block's class and the fewest CZ it needs:

    (0, 0, 0)            local       0 CZ   single-qubit gates only
    (pi/4, 0, 0)         cx          1 CZ   a CNOT (or CZ) up to single-qubit gates
    (pi/4, pi/4, 0)      cx_move     2 CZ   a CNOT followed by a SWAP: the gate and the move in one
    (pi/4, pi/4, pi/4)   swap        3 CZ   a SWAP the router inserted
    anything else        other

In Bernstein-Vazirani every logical gate is a CNOT into the oracle qubit, so a router that moves the oracle with
SWAPs pays 3 CZ per hop plus 1 CZ for the CNOT, while a "cx_move" block does the CNOT and the hop in 2 CZ.
`block_counts(qc)` returns how many blocks of each class a circuit has and how many CZ they use.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

CLASSES = ("local", "cx", "cx_move", "swap", "other")
_Q = np.pi / 4


def weyl_class(u: np.ndarray, tol: float = 1e-6) -> str:
    """The class of a 4x4 unitary by its Weyl coordinates (see the module docstring)."""
    from qiskit.synthesis import TwoQubitWeylDecomposition

    d = TwoQubitWeylDecomposition(u)
    a, b, c = (abs(x) for x in (d.a, d.b, d.c))
    near = lambda x, y: abs(x - y) < tol  # noqa: E731
    if near(a, 0) and near(b, 0) and near(c, 0):
        return "local"
    if near(a, _Q) and near(b, 0) and near(c, 0):
        return "cx"
    if near(a, _Q) and near(b, _Q) and near(c, 0):
        return "cx_move"
    if near(a, _Q) and near(b, _Q) and near(c, _Q):
        return "swap"
    return "other"


def block_counts(qc) -> Dict[str, Dict[str, int]]:
    """{"blocks": {class: count}, "two_qubit_gates": {class: two-qubit gates inside}} for a compiled circuit
    (measurements and barriers are ignored; blocks come from Qiskit's Collect2qBlocks)."""
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from qiskit.quantum_info import Operator
    from qiskit.transpiler.passes import Collect2qBlocks, RemoveBarriers

    bare = RemoveBarriers()(qc.remove_final_measurements(inplace=False))
    dag = circuit_to_dag(bare)
    pass_ = Collect2qBlocks()
    pass_.run(dag)
    blocks = {k: 0 for k in CLASSES}
    gates = {k: 0 for k in CLASSES}
    for block in pass_.property_set["block_list"]:
        qubits = sorted({q for node in block for q in node.qargs}, key=lambda q: bare.find_bit(q).index)
        n2 = sum(1 for node in block if len(node.qargs) == 2)
        if len(qubits) != 2 or n2 == 0:
            continue
        sub = QuantumCircuit(2)
        where = {q: i for i, q in enumerate(qubits)}
        for node in block:
            sub.append(node.op, [where[q] for q in node.qargs])
        k = weyl_class(Operator(sub).data)
        blocks[k] += 1
        gates[k] += n2
    return {"blocks": blocks, "two_qubit_gates": gates}
