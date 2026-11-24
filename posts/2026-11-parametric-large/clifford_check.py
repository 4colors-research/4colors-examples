"""A correctness check that works at 150 qubits: bind at Clifford angles, compare stabilizer states.

A 150-qubit statevector does not fit in any computer's memory, so the statevector check of the compile-once post
stops at about 24 active qubits. This module checks a compiled parametric circuit at full size instead:

1. Draw values for every parameter from {0, pi/2, pi, 3 pi/2}. In this post's ansatze every rotation angle is a
   whole multiple of one parameter (ry, rz, rx(2 b), rzz(-g)), so the bound input is a Clifford circuit.
2. Bind the compiled circuit with the same values (by your own Parameter objects). If its fixed gates are Clifford
   too (rz, sx, x, cz with angles that are multiples of pi/2), the bound output is a Clifford circuit as well.
3. Simulate both from |0...0> as stabilizer states (a tableau: memory grows with n^2, not 2^n) and compare. The
   compiled circuit's `layout` says where each input qubit ends; every other qubit it touches must return to |0>
   (the same bookkeeping as the statevector check, `qaoakit.apikit.reduce_active`).

Some compilers (Qiskit at optimization level 3) re-synthesize gate blocks into pieces that are Clifford only as a
whole (rotations that are not multiples of pi/2 around a pair of CZ). When the direct simulation refuses a gate,
the check merges the bound circuit into blocks of at most 2, then 3, then 4 qubits (Qiskit's CollectMultiQBlocks and
ConsolidateBlocks: an exact rewrite of the bound circuit, used for the check only, never for counting), reads each
block's matrix as a Clifford (it must be one, up to a global phase) and simulates that
(`method: "stabilizer_blocks<k>"`, k the block size that worked). Both compilers' outputs go through the same
function.

A match at several random Clifford points is strong evidence, not a proof, that the compiled circuit is the input
circuit for every value: a wrong gate, a dropped parameter, a sign flip or a bad layout shows up at almost every
Clifford point. If a bound circuit is not Clifford even block by block, the check says so (`method:
"not_clifford"`, `ok: None`) and makes no claim.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, StabilizerState

from qaoakit import apikit as ak

BLOCK_SIZES = (2, 3, 4)


def clifford_values(n_params: int, seed: int) -> List[float]:
    """`n_params` values, each a random whole multiple of pi/2 in [0, 2 pi)."""
    rng = np.random.default_rng(seed)
    return [float(k * np.pi / 2) for k in rng.integers(0, 4, n_params)]


def _direct(qc: QuantumCircuit) -> Optional[StabilizerState]:
    try:
        return StabilizerState(qc)
    except QiskitError:
        return None


def _blocks(qc: QuantumCircuit, k: int) -> Optional[StabilizerState]:
    """Merge into blocks of at most k qubits, read each block's matrix as a Clifford, simulate. None if a block is
    not Clifford."""
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import CollectMultiQBlocks, ConsolidateBlocks

    merged = PassManager([CollectMultiQBlocks(max_block_size=k), ConsolidateBlocks(force_consolidate=True)]).run(qc)
    out = QuantumCircuit(merged.num_qubits)
    try:
        for ins in merged.data:
            qubits = [merged.find_bit(q).index for q in ins.qubits]
            if ins.operation.name == "unitary":
                out.compose(Clifford.from_matrix(ins.operation.to_matrix()).to_circuit(), qubits, inplace=True)
            else:
                out.append(ins.operation, qubits)
        return StabilizerState(out)
    except QiskitError:
        return None


def stabilizer_state(qc: QuantumCircuit) -> Tuple[Optional[StabilizerState], str]:
    """(stabilizer state of qc from |0...0>, method), or (None, "not_clifford")."""
    s = _direct(qc)
    if s is not None:
        return s, "stabilizer"
    for k in BLOCK_SIZES:
        s = _blocks(qc, k)
        if s is not None:
            return s, f"stabilizer_blocks{k}"
    return None, "not_clifford"


def stabilizer_match(input_qc: QuantumCircuit, compiled_qc: QuantumCircuit, values: Sequence[float],
                     compiled_values: Optional[Sequence[float]] = None) -> Dict:
    """Does `compiled_qc`, bound with `compiled_values` (default `values`), prepare the stabilizer state of
    `input_qc` bound with `values`? Values are bound by `input_qc.parameters` objects, in that order. A compiled
    circuit that lost a parameter is still bound (the rest of its parameters) and compared: a lost gate shows up as
    a mismatch.

    Returns dict(ok, method, active_qubits): ok True or False with method "stabilizer" or "stabilizer_blocks<k>"
    (the compiled side's method); ok None with method "not_clifford" (and `side`, "input" or "compiled") when a
    bound circuit is not Clifford even block by block."""
    params = list(input_qc.parameters)
    cv = values if compiled_values is None else compiled_values
    bound = compiled_qc.assign_parameters(dict(zip(params, cv)), strict=False)
    red, where = ak.reduce_active(bound, compiled_qc.layout, input_qc.num_qubits)
    expected = QuantumCircuit(red.num_qubits)
    expected.compose(ak.strip_final_ops(input_qc.assign_parameters(dict(zip(params, values)))), where, inplace=True)
    if red.parameters:                                   # a parameter the input does not have: cannot bind it
        return dict(ok=False, method="unbound_parameters", active_qubits=red.num_qubits)
    (want, _), (got, method) = stabilizer_state(expected), stabilizer_state(red)
    if want is None or got is None:
        return dict(ok=None, method="not_clifford", side="input" if want is None else "compiled",
                    active_qubits=red.num_qubits)
    return dict(ok=bool(got.equiv(want)), method=method, active_qubits=red.num_qubits)


def check_many(input_qc: QuantumCircuit, compiled_qc: QuantumCircuit, draws: int, seed: int) -> Dict:
    """`draws` random Clifford points. ok is True only if every point matches, False if any point differs, and None
    if any point could not be checked (not Clifford). Returns dict(clifford_ok, clifford_method, clifford_draws,
    clifford_mismatches, active_qubits); clifford_method is the largest block merge any point needed."""
    n = len(input_qc.parameters)
    results = [stabilizer_match(input_qc, compiled_qc, clifford_values(n, 1000 * seed + d)) for d in range(draws)]
    if not results:
        return dict(clifford_ok=None, clifford_method=None, clifford_draws=0, clifford_mismatches=None,
                    active_qubits=None)
    if any(r["ok"] is None for r in results):
        ok, method = None, "not_clifford"
    else:
        ok, method = all(r["ok"] for r in results), max((r["method"] for r in results), key=_method_rank)
    return dict(clifford_ok=ok, clifford_method=method, clifford_draws=draws,
                clifford_mismatches=sum(1 for r in results if r["ok"] is False),
                active_qubits=results[0]["active_qubits"])


def _method_rank(m: str) -> int:
    if m == "stabilizer":
        return 0
    if m.startswith("stabilizer_blocks"):
        return int(m[len("stabilizer_blocks"):])
    return 99
