"""The stated noise model: Qiskit Aer's device noise model built from ONE published calibration snapshot (the one
shipped with the fake backend), applied the same way to every compiled circuit.

    sim = simulator(FakeFez())
    dist, meta = noisy_distribution(compiled_qc, sim, FakeFez().target, shots=100_000, seed=1)

The model (Aer `NoiseModel.from_backend`): depolarizing error per gate at the calibrated gate error, thermal
relaxation (T1, T2) for each gate's duration, readout error per qubit. Idle time is charged too: each circuit is
scheduled as late as possible with the calibrated durations and every idle gap on a qubit the circuit uses becomes a
delay, which Aer gives thermal relaxation. Qubits the circuit never touches get no delays and drop out of the
simulation. This is a simulation with a noise model, not a hardware run: crosstalk, leakage, drift and
non-Markovian noise are not in it.
"""
from __future__ import annotations

from typing import Dict, Tuple

#: Density-matrix simulation up to this many active qubits (16 MB at 10, 1 GB at 13); trajectories above.
MAX_DM_QUBITS = 13


def simulator(backend, method: str = "density_matrix"):
    from qiskit_aer import AerSimulator

    return AerSimulator.from_backend(backend, method=method)


def used_qubits(qc) -> set:
    """Physical qubits touched by any operation other than a delay or a barrier."""
    return {qc.find_bit(q).index for inst in qc.data if inst.operation.name not in ("delay", "barrier")
            for q in inst.qubits}


def schedule(qc, target):
    """ALAP schedule with the target's durations; idle gaps become delays, only on qubits the circuit uses."""
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay

    s = PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)]).run(qc)
    used = used_qubits(s)
    out = s.copy_empty_like()
    for inst in s.data:
        if inst.operation.name == "delay" and not {s.find_bit(q).index for q in inst.qubits} <= used:
            continue
        out.append(inst)
    return out


def noisy_distribution(qc, sim, target, *, shots: int, seed: int, idle: bool = True, sv_shots: int = 20_000) -> Tuple[Dict[str, float], dict]:
    """Sampled output distribution (qubit 0 first, see observables.from_counts) of a compiled circuit under the
    noise model, and what ran (method, active qubits, shots, scheduled duration). Up to MAX_DM_QUBITS active
    qubits the density matrix is exact and `shots` samples are cheap; above it each shot is a noisy trajectory, and
    at most `sv_shots` are run (the row records the shots actually used)."""
    import observables

    run_qc = schedule(qc, target) if idle else qc
    n_active = len(used_qubits(run_qc))
    method = "density_matrix" if n_active <= MAX_DM_QUBITS else "statevector"
    shots = shots if method == "density_matrix" else min(shots, sv_shots)
    res = sim.run(run_qc, shots=shots, seed_simulator=seed, method=method).result()
    counts = res.get_counts()
    try:
        dur = run_qc.estimate_duration(target, unit="s")
    except Exception:                                           # an older Qiskit, or an unscheduled circuit
        dur = None
    return observables.from_counts(counts), {"sim_method": method, "active_qubits": n_active, "shots": shots,
                                             "sim_seed": seed, "idle_noise": idle, "duration_s": dur}
