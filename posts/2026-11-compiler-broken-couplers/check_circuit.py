"""Check your own circuits: how many two-qubit gates sit on a coupler the calibration marks as failed?

Plain Qiskit, no 4colors account needed. Works on any circuit that is already laid out on the device (the output of
`transpile`, of a 4colors compile via `result.to_qiskit()`, or a circuit you wrote by hand on physical qubits):

    from qiskit_ibm_runtime.fake_provider import FakeFez
    from check_circuit import gates_on_dead_couplers

    backend = FakeFez()                      # or your real backend: backend.target has today's calibration
    report = gates_on_dead_couplers(transpiled, backend.target)
    print(report["dead_couplers"], report["gates_on_dead"], report["by_coupler"])

A coupler counts as failed when its two-qubit error is at least 0.999 in either direction (IBM publishes an error of
1.0 for couplers that do not work). The threshold is a parameter.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

Pair = Tuple[int, int]

FAILED_ERROR = 0.999


def dead_couplers(target, threshold: float = FAILED_ERROR) -> Set[Pair]:
    """Couplers (sorted qubit pairs) whose two-qubit error in `target` (a Qiskit `Target`) is >= threshold."""
    dead: Set[Pair] = set()
    for name in target.operation_names:
        for qargs, props in (target[name] or {}).items():
            if qargs is None or len(qargs) != 2 or props is None or props.error is None:
                continue
            if float(props.error) >= threshold:
                dead.add(tuple(sorted(qargs)))
    return dead


def gates_on_dead_couplers(circuit, target, threshold: float = FAILED_ERROR) -> Dict:
    """Count the two-qubit gates of a laid-out `circuit` that act on a failed coupler of `target`.

    Returns {"dead_couplers": sorted pairs, "two_qubit_gates": int, "gates_on_dead": int,
    "by_coupler": {"a-b": count}}. Barriers and other directives are ignored."""
    dead = dead_couplers(target, threshold)
    total, on_dead = 0, 0
    by: Dict[str, int] = {}
    for inst in circuit.data:
        op = inst.operation
        if getattr(op, "_directive", False) or op.name in ("barrier", "delay", "measure", "reset"):
            continue
        if len(inst.qubits) != 2:
            continue
        total += 1
        pair = tuple(sorted(circuit.find_bit(q).index for q in inst.qubits))
        if pair in dead:
            on_dead += 1
            key = f"{pair[0]}-{pair[1]}"
            by[key] = by.get(key, 0) + 1
    return {"dead_couplers": [list(p) for p in sorted(dead)], "two_qubit_gates": total,
            "gates_on_dead": on_dead, "by_coupler": dict(sorted(by.items()))}


def main(argv: List[str] | None = None) -> int:
    """`python check_circuit.py circuit.qasm` checks a QASM file (OpenQASM 2 or 3) against FakeFez."""
    import argparse
    import json

    from qiskit import qasm2, qasm3
    from qiskit_ibm_runtime.fake_provider import FakeFez

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("qasm", help="a laid-out circuit as OpenQASM 2 or 3")
    a = p.parse_args(argv)
    text = open(a.qasm).read()
    qc = qasm3.loads(text) if "OPENQASM 3" in text else qasm2.loads(text)
    rep = gates_on_dead_couplers(qc, FakeFez().target)
    print(json.dumps(rep, indent=1))
    return 1 if rep["gates_on_dead"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
