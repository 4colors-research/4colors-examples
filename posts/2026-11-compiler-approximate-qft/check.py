"""One scorer for both compilers: how far is a compiled circuit from the exact QFT?

`infidelity(compiled, reference, ...)` prepares random product inputs drawn from the six single-qubit Pauli
eigenstates (|0>, |1>, |+>, |->, |+i>, |-i>) on the input qubits, runs the compiled circuit (cut down to the qubits it
touches), then the inverse of the reference on the qubits where the compiled circuit left each input qubit, then the
inverse of the preparation, and reads the probability that every touched qubit is back in |0>. That probability is
the state fidelity for that input; the result is 1 - mean fidelity (and the worst input).

Why these inputs: computational-basis inputs alone miss diagonal errors (a Z on an input qubit only changes a global
phase on |0> and |1>); the |+>-type inputs see them. Why this works at 32 qubits: the QFT has low entanglement, so a
matrix-product-state simulation handles it (`method="matrix_product_state"`); small cases can use "statevector".
This is a sampled estimate, not a proof of equivalence; the service's own verification record is reported beside it.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

#: The six Pauli eigenstates, as the gates that prepare them from |0>.
PREP = {"0": [], "1": [("x",)], "+": [("h",)], "-": [("x",), ("h",)], "+i": [("h",), ("s",)],
        "-i": [("h",), ("sdg",)]}
NON_GATES = ("measure", "barrier", "delay", "reset")


def positions(compiled, n_logical: int):
    """(initial, final) physical index of each input qubit in `compiled` (trivial when it has no layout)."""
    lay = getattr(compiled, "layout", None)
    if lay is None:
        return list(range(n_logical)), list(range(n_logical))
    init = lay.initial_index_layout(filter_ancillas=True)[:n_logical]
    final = lay.final_index_layout(filter_ancillas=True)[:n_logical]
    return list(init), list(final)


def reduce(compiled, n_logical: int):
    """The compiled circuit on the qubits it touches (plus the input qubits' start and end positions), with the
    positions of each input qubit at the start and the end in that smaller circuit."""
    from qiskit import QuantumCircuit

    init, final = positions(compiled, n_logical)
    touched = {compiled.find_bit(q).index for ins in compiled.data if ins.operation.name not in NON_GATES
               for q in ins.qubits}
    active = sorted(touched | set(init) | set(final))
    pos = {p: k for k, p in enumerate(active)}
    red = QuantumCircuit(len(active), global_phase=compiled.global_phase)
    for ins in compiled.data:
        if ins.operation.name in NON_GATES:
            continue
        red.append(ins.operation, [pos[compiled.find_bit(q).index] for q in ins.qubits])
    return red, [pos[p] for p in init], [pos[p] for p in final]


def random_inputs(n: int, count: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    keys = list(PREP)
    return [[keys[i] for i in rng.integers(0, len(keys), n)] for _ in range(count)]


def _prep(qc, labels: Sequence[str], where: Sequence[int], inverse: bool = False):
    for lab, q in zip(labels, where):
        ops = PREP[lab]
        for (name,) in (reversed(ops) if inverse else ops):
            if inverse:
                name = {"s": "sdg", "sdg": "s"}.get(name, name)
            getattr(qc, name)(q)


def scoring_circuit(compiled, reference, labels: Sequence[str]):
    """The circuit whose all-zero amplitude is <input| ref^-1 C |input>, on the touched qubits."""
    n = reference.num_qubits
    red, init, final = reduce(compiled, n)
    qc = red.copy_empty_like()
    _prep(qc, labels, init)
    qc.compose(red, inplace=True)
    qc.compose(reference.inverse(), final, inplace=True)
    _prep(qc, labels, final, inverse=True)
    return qc


def fidelity_one(compiled, reference, labels, method: str = "matrix_product_state") -> float:
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    qc = scoring_circuit(compiled, reference, labels)
    qc.save_amplitudes([0])
    sim = AerSimulator(method=method, max_parallel_threads=1)
    amp = sim.run(transpile(qc, sim, optimization_level=0)).result().data()["amplitudes"][0]
    return float(abs(amp) ** 2)


def infidelity(compiled, reference, *, n_inputs: int = 8, seed: int = 1, method: str = "matrix_product_state",
               max_active: Optional[int] = None) -> dict:
    """1 - mean fidelity over `n_inputs` random Pauli-eigenstate product inputs, the worst input's infidelity, and
    the number of touched qubits. Returns infidelity None when more than `max_active` qubits are touched."""
    red, _, _ = reduce(compiled, reference.num_qubits)
    if max_active is not None and red.num_qubits > max_active:
        return {"infidelity": None, "worst_infidelity": None, "active_qubits": red.num_qubits, "check_inputs": 0}
    fids = [fidelity_one(compiled, reference, labels, method)
            for labels in random_inputs(reference.num_qubits, n_inputs, seed)]
    clip = lambda v: max(0.0, float(v))
    return {"infidelity": clip(1 - float(np.mean(fids))), "worst_infidelity": clip(1 - min(fids)),
            "active_qubits": red.num_qubits, "check_inputs": n_inputs, "check_method": method}


def aqft(n: int, m: int):
    """The textbook approximate QFT (Coppersmith): the QFT on n qubits (no final swaps) keeping only the controlled
    phases of angle at least pi / 2**m. Qiskit's `approximation_degree` counts the dropped smallest angles:
    degree = n - 1 - m. Returns None when nothing is dropped (n <= m + 1)."""
    from qiskit.synthesis import synth_qft_full

    degree = n - 1 - m
    if degree <= 0:
        return None
    qc = synth_qft_full(n, do_swaps=False, approximation_degree=degree)
    qc.name = f"aqft{m}_{n}"
    return qc


def dropped_phase_bound(n: int, m: int) -> float:
    """A loose textbook bound on the operator-norm error of AQFT_m: the sum of the dropped rotation angles / 2
    (each controlled phase of angle t is within t/2 of the identity, up to phase). For context only."""
    return sum(math.pi / 2 ** k / 2 * (n - k) for k in range(m + 1, n))
