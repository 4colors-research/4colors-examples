"""The approximation pre-pass (rule D2): how far is Qiskit's compiled circuit from the input?

The comparison is only fair at the same approximation. Qiskit's `transpile` is exact by default
(`approximation_degree` 1.0), but "exact by default" is a claim, so the pre-pass measures it where that is cheap:

* the circuit without its final measurements is transpiled with the same seed the Qiskit arm used (the final
  measurements only let Qiskit drop diagonal gates in front of them, which changes no measured outcome);
* the compiled circuit is cut down to the physical qubits it touches; when that is at most `max_active` qubits,
  both circuits are run on `n_states` random product input states (statevector), the input with each virtual qubit
  placed where the layout says it ends, and d_q = 1 - (lowest fidelity).

d_q below `tol` counts as exact. Circuits with resets or mid-circuit measurements, and circuits wider than
`max_active`, are UNMEASURED (d_q = None): the 4colors arm then stays exact, which is conservative against it.
This is a local check of Qiskit's output, not a compiler: it never changes a circuit.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

TOL = 1e-9


def _strip_measure(qc):
    out = qc.copy()
    out.remove_final_measurements(inplace=True)
    return out


def has_nonunitary(qc) -> bool:
    """Reset, or a measurement that is not at the end, anywhere in the circuit."""
    core = _strip_measure(qc)
    return any(inst.operation.name in ("reset", "measure") for inst in core.data)


def _random_u(rng):
    a, b, c = rng.uniform(0, 2 * np.pi, 3)
    return float(np.arccos(1 - 2 * rng.uniform())), float(b), float(c)


def fidelities(qc_in, compiled, n_states: int = 8, seed: int = 7, max_active: int = 20) -> Optional[List[float]]:
    """Fidelity of `compiled` (a transpiled circuit with a `layout`) against `qc_in` (no final measurements) on
    random product states, or None when the active width is above `max_active` or the circuit is not unitary."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    if has_nonunitary(qc_in):
        return None
    comp = _strip_measure(compiled)
    n = qc_in.num_qubits
    lay = comp.layout
    if lay is None:
        init = list(range(n)); final = list(range(n))
    else:
        init = lay.initial_index_layout(filter_ancillas=True)
        final = lay.final_index_layout(filter_ancillas=True)
    active = sorted({comp.find_bit(q).index for inst in comp.data for q in inst.qubits} | set(init) | set(final))
    if len(active) > max_active:
        return None
    pos = {p: k for k, p in enumerate(active)}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_states):
        angles = [_random_u(rng) for _ in range(n)]
        actual = QuantumCircuit(len(active))
        for v, (t, p, l) in enumerate(angles):
            actual.u(t, p, l, pos[init[v]])
        for inst in comp.data:
            actual.append(inst.operation, [pos[comp.find_bit(q).index] for q in inst.qubits])
        ideal = QuantumCircuit(len(active))
        for v, (t, p, l) in enumerate(angles):
            ideal.u(t, p, l, pos[final[v]])
        ideal.compose(qc_in, qubits=[pos[final[v]] for v in range(n)], inplace=True)
        a, b = Statevector(actual), Statevector(ideal)
        out.append(float(abs(np.vdot(b.data, a.data)) ** 2))
    return out


def measure_dq(qc, target, qiskit_seed: int, **kw) -> dict:
    """The pre-pass row fields for one circuit and one Qiskit seed: `dq` (None = unmeasured), `dq_method`,
    `dq_states`."""
    from qiskit import transpile

    core = _strip_measure(qc)
    if has_nonunitary(qc):
        return {"dq": None, "dq_method": "unmeasured: reset or mid-circuit measurement", "dq_states": 0}
    out = transpile(core, target=target, optimization_level=3, seed_transpiler=qiskit_seed)
    f = fidelities(core, out, **kw)
    if f is None:
        return {"dq": None, "dq_method": "unmeasured: too wide for the statevector check", "dq_states": 0}
    dq = max(0.0, 1.0 - min(f))
    return {"dq": 0.0 if dq < TOL else dq, "dq_method": "statevector, random product states", "dq_states": len(f)}


def ours_options(dq: Optional[float]) -> dict:
    """The 4colors arm's approximation options for a cell (D2): exact unless Qiskit measurably approximated, then
    `max_approximation_error` = d_q (capped at the service's 0.1)."""
    if dq is None or dq == 0.0:
        return {"approximation_mode": "exact"}
    return {"max_approximation_error": min(float(dq), 0.1)}
