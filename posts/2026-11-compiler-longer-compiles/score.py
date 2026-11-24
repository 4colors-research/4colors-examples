"""The one scorer: every arm's final circuit is scored here, the same way (see the post).

* `score(qc, target)`: two-qubit gates, two-qubit depth, depth (`circuitkit.circuit_metrics`) and the estimated
  success probability (`circuitkit.estimated_esp`: the product of (1 - error) over every gate and measurement
  in the published calibration, rz free, a gate on a failed element gives 0). ESP is a model, never a measurement.
* `approximation(original, compiled)`: how far the compiled circuit is from the input as an operator, for
  circuits small enough to check exactly (at most MAX_EXACT_QUBITS physical qubits touched). It follows the
  compiled circuit's own layout: virtual qubit i starts on initial[i] and ends on final[i]; unused physical qubits
  start and end in |0>. Returns 1 - |Tr(U^dagger M)| / 2^n, which is 0 for an exact compile up to a global phase.
  Larger circuits return None (not measured); the post says what the 4colors arm gets then.

The Qiskit-long driver keeps its own copy of the metrics only to pick its incumbents; `tests/` checks that the two
agree on the 2q count, the 2q depth, the depth and ESP.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import circuitkit as ck  # noqa: E402

MAX_EXACT_QUBITS = 12
SKIP = frozenset({"barrier", "measure", "delay"})


def score(qc, target=None) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = dict(ck.circuit_metrics(qc))
    out["estimated_esp"] = ck.estimated_esp(qc, target) if target is not None else None
    return out


def layouts(qc, n_virtual: int) -> Optional[Dict[str, List[int]]]:
    """{"initial": [...], "final": [...]}: the physical qubit of each virtual qubit at the start and at the end.
    None if the circuit carries no layout (then it is read as the identity)."""
    lay = getattr(qc, "layout", None)
    if lay is None:
        return None
    return {"initial": list(lay.initial_index_layout(filter_ancillas=True))[:n_virtual],
            "final": list(lay.final_index_layout(filter_ancillas=True))[:n_virtual]}


def _embed(n: int, positions: List[int], m: int):
    """The 2^m x 2^n isometry that puts virtual qubit i on compact qubit positions[i], the rest in |0>."""
    import numpy as np

    E = np.zeros((2 ** m, 2 ** n), dtype=complex)
    for x in range(2 ** n):
        y = 0
        for i, p in enumerate(positions):
            if (x >> i) & 1:
                y |= 1 << p
        E[y, x] = 1.0
    return E


def approximation(original, compiled, lay: Optional[Dict[str, List[int]]] = None) -> Optional[float]:
    """1 - |Tr(U_in^dagger M)| / 2^n with M the compiled operator restricted to the layout's input and output
    positions (see the module docstring); None if more than MAX_EXACT_QUBITS physical qubits are touched."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    n = original.num_qubits
    lay = lay or layouts(compiled, n) or {"initial": list(range(n)), "final": list(range(n))}
    touched = set(lay["initial"]) | set(lay["final"])
    for inst in compiled.data:
        if inst.operation.name not in SKIP:
            touched |= {compiled.find_bit(q).index for q in inst.qubits}
    order = list(lay["initial"]) + sorted(touched - set(lay["initial"]))
    m = len(order)
    if m > MAX_EXACT_QUBITS:
        return None
    idx = {p: k for k, p in enumerate(order)}
    small = QuantumCircuit(m, global_phase=compiled.global_phase)
    for inst in compiled.data:
        if inst.operation.name in SKIP:
            continue
        small.append(inst.operation, [idx[compiled.find_bit(q).index] for q in inst.qubits])
    U = Operator(small).data
    E_in = _embed(n, [idx[p] for p in lay["initial"]], m)
    E_out = _embed(n, [idx[p] for p in lay["final"]], m)
    M = E_out.conj().T @ U @ E_in
    plain = original.remove_final_measurements(inplace=False)
    V = Operator(plain).data
    f = abs(np.trace(V.conj().T @ M)) / 2 ** n
    return max(0.0, 1.0 - float(f))
