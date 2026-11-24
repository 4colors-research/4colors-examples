"""A cheap, exact check for compiled Bernstein-Vazirani circuits: Clifford (stabilizer tableau) simulation.

BV is a Clifford circuit, so a 151-qubit output can be simulated exactly in milliseconds on a stabilizer tableau,
with no statevector. `returns_secret(compiled, secret)` runs the compiled circuit from |0...0>, reads each
measured qubit through the circuit's own measure instructions (so routing and the final layout are covered), and
checks that every measured bit is deterministic and that together they spell the secret.

Two paths: the exact tableau path for circuits whose single-qubit runs are Cliffords, and, for outputs whose
re-synthesized blocks are Clifford only as a whole, Aer's matrix-product-state simulator (no truncation; BV's ideal
state is a product state, so it stays cheap at 151 qubits). Each row records which path ran.

Compilers re-synthesize runs of single-qubit gates into rotations whose angles are not multiples of pi/2 one by
one, although each run as a whole is a Clifford. So each run is first merged into one 2x2 matrix and only then
matched to one of the 24 single-qubit Cliffords.

This checks the circuit as run (from |0...0>, measured in the computational basis), which is what BV does. It is not
a proof of unitary equivalence on every input state; the post says so.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


_ONE_QUBIT_CLIFFORDS = None


def _one_qubit_cliffords():
    """The 24 single-qubit Cliffords up to phase, each as (gate names, 2x2 matrix), from words in h and s."""
    global _ONE_QUBIT_CLIFFORDS
    if _ONE_QUBIT_CLIFFORDS is None:
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        S = np.diag([1, 1j])
        found, frontier = [((), np.eye(2))], [((), np.eye(2))]
        while frontier:
            nxt = []
            for word, m in frontier:
                for name, g in (("h", H), ("s", S)):
                    m2 = g @ m
                    if not any(abs(abs(np.trace(f.conj().T @ m2)) - 2) < 1e-9 for _, f in found):
                        found.append((word + (name,), m2))
                        nxt.append((word + (name,), m2))
            frontier = nxt
        assert len(found) == 24
        _ONE_QUBIT_CLIFFORDS = found
    return _ONE_QUBIT_CLIFFORDS


def _as_clifford_word(m: np.ndarray, tol: float = 1e-6):
    """The h/s word equal to the 2x2 unitary `m` up to a global phase, or None if `m` is not a Clifford."""
    for word, c in _one_qubit_cliffords():
        if abs(abs(np.trace(c.conj().T @ m)) - 2) < tol:
            return word
    return None


def _merged(compiled):
    """`compiled` without measurements and barriers, with every run of single-qubit gates on a qubit merged into
    one Clifford (an h/s word); returns (circuit, {clbit index: physical qubit index})."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    n = compiled.num_qubits
    out = QuantumCircuit(n)
    pending: Dict[int, np.ndarray] = {}
    measured: Dict[int, int] = {}

    def flush(q):
        m = pending.pop(q, None)
        if m is None:
            return
        word = _as_clifford_word(m)
        if word is None:
            raise ValueError(f"a run of single-qubit gates on qubit {q} is not a Clifford")
        for name in word:            # the word lists gates in the order they are applied
            getattr(out, name)(q)

    for inst in compiled.data:
        op = inst.operation
        qs = [compiled.find_bit(q).index for q in inst.qubits]
        if op.name == "barrier":
            continue
        if op.name == "measure":
            flush(qs[0])
            measured[compiled.find_bit(inst.clbits[0]).index] = qs[0]
            continue
        if op.name in ("reset", "delay"):
            raise ValueError(f"unsupported operation in a BV check: {op.name}")
        if len(qs) == 1:
            pending[qs[0]] = Operator(op).data @ pending.get(qs[0], np.eye(2))
            continue
        for q in qs:
            flush(q)
        out.append(op, qs)
    for q in list(pending):
        flush(q)
    return out, measured


def _tableau(compiled, secret: str) -> dict:
    """The Clifford-tableau path: exact, but only for circuits whose single-qubit runs are Cliffords."""
    from qiskit.quantum_info import StabilizerState

    bare, measured = _merged(compiled)          # raises ValueError on a non-Clifford run
    state = StabilizerState(bare)
    if sorted(measured) != list(range(len(secret))):
        return {"ok": False, "reason": f"measures {len(measured)} clbits, expected {len(secret)}"}
    got = []
    for c in range(len(secret)):
        p = state.probabilities([measured[c]])
        if max(p) < 1 - 1e-9:
            return {"ok": False, "reason": f"clbit {c} is not deterministic"}
        got.append("1" if p[1] > 0.5 else "0")
    ok = "".join(got) == secret
    return {"ok": ok, "reason": None if ok else "measures a different bit string"}


def _mps(compiled, secret: str, shots: int) -> dict:
    """The fallback: Aer's matrix-product-state simulator (no truncation), `shots` shots of the circuit as compiled.
    BV's ideal state is a product state, so the bond dimension stays small even at 151 qubits."""
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="matrix_product_state", max_parallel_threads=1, seed_simulator=1)
    counts = sim.run(compiled, shots=shots).result().get_counts()
    want = secret[::-1]                              # Qiskit prints clbit 0 last
    ok = counts == {want: shots}
    return {"ok": ok, "reason": None if ok else f"{len(counts)} distinct outcomes, {counts.get(want, 0)} of {shots} "
                                                f"shots return the secret"}


def returns_secret(compiled, secret: str, *, shots: int = 64) -> dict:
    """{"ok", "method", "reason"}: does `compiled`, run from |0...0>, measure `secret`?

    First the exact Clifford-tableau path ("clifford_tableau"). Compilers that re-synthesize two-qubit blocks
    (Qiskit at level 3 does) leave single-qubit runs that are not Cliffords one by one although the whole circuit
    is; those circuits go to the matrix-product-state path ("mps_<shots>_shots"). For a Clifford circuit a
    non-deterministic outcome would miss the secret on each shot with probability at least 1/2, so 64 of 64 shots
    leave a chance below 1e-19 that a non-deterministic circuit passes. `secret[i]` is classical bit i."""
    try:
        return {**_tableau(compiled, secret), "method": "clifford_tableau"}
    except ValueError:
        pass
    try:
        return {**_mps(compiled, secret, shots), "method": f"mps_{shots}_shots"}
    except Exception as e:  # never skipped: a check that cannot run is a failed check, with its reason
        return {"ok": False, "method": "none", "reason": f"check could not run: {str(e)[:200]}"}


def check_fields(compiled, secret: Optional[str]) -> dict:
    """Row fields for one compiled circuit: `clifford_check` is "returns_secret" or "wrong", with the method."""
    if compiled is None or secret is None:
        return {"clifford_check": None, "clifford_check_method": None, "clifford_check_reason": "no circuit"}
    r = returns_secret(compiled, secret)
    return {"clifford_check": "returns_secret" if r["ok"] else "wrong", "clifford_check_method": r["method"],
            "clifford_check_reason": r["reason"]}
