"""Compile early or compile late? A parametric compile against bound compiles along an optimizer trajectory.

For each ansatz, a noiseless COBYLA run gives a trajectory of angle vectors; `--points` of them, evenly spaced
from the first evaluation after COBYLA's initial simplex to the last, are compiled four ways. The start point is
never compiled: for the UCCSD-style ansatz it is the Hartree-Fock point (all angles zero), where every excitation
block is an identity and a bound compile trivially removes almost everything; and the first n + 1 evaluations
(the simplex) move one angle at a time from the start. The trajectory row records which evaluations were chosen.
  - parametric: the circuit with free parameters, compiled once (4colors and Qiskit O3), the same gates at
    every point;
  - bound, exact: the circuit with the point's angles, `compile()` with `approximation_mode="exact"` (sent
    explicitly, never left to the service's default);
  - bound, per-block: the same with `approximation_mode="per_block"`; the verification label and its measured
    distance say what the check found;
  - bound, Qiskit O3 (matched budget; `approximation_degree` 1.0, the transpile default: exact).
Every bound compile is also checked here on a noiseless statevector (fidelity with the bound input), the same
check for both compilers (one scorer). Comparisons (D2): ours exact against Qiskit exact; per_block only against
ours exact (what the approximation buys), never against Qiskit exact.

Usage:
    python run.py --out runs/p3 [--ansatz qaoa-12-p2,uccsd-6-2] [--points 8] [--maxiter 120] [--budget 10]
                  [--seed 1] [--wait-verification 300]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
import examplekit as bk  # noqa: E402
from qaoakit import ansatz as az, apikit as ak, core  # noqa: E402


def toy_hamiltonian(n, seed=7):
    """A toy number-conserving fermionic Hamiltonian (Jordan-Wigner): random on-site energies, hopping between
    neighboring orbitals, random density-density terms. Chemistry-shaped, not a molecule."""
    rng = np.random.default_rng(seed)
    terms = []
    for i in range(n):                                       # eps_i n_i, n_i = (1 - Z_i) / 2
        e = rng.uniform(-1, 1)
        terms += [("", [], e / 2), ("Z", [i], -e / 2)]
    for i in range(n - 1):                                   # t (XX + YY) / 2 between neighbors
        t = rng.uniform(0.2, 0.6)
        terms += [("XX", [i, i + 1], t / 2), ("YY", [i, i + 1], t / 2)]
    for i in range(n):
        for j in range(i + 1, n):                            # U n_i n_j
            u = rng.uniform(0, 0.5)
            terms += [("", [], u / 4), ("Z", [i], -u / 4), ("Z", [j], -u / 4), ("ZZ", [i, j], u / 4)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()


def objective(spec, qc, inst):
    """Function to minimize, and a description: minus the expected cut (QAOA) or the toy energy (UCCSD-style)."""
    params = list(qc.parameters)
    if inst is not None:
        cost = core.cost_vector(inst)
        return (lambda x: -float(Statevector(qc.assign_parameters(dict(zip(params, x)))).probabilities() @ cost),
                "minus expected cut")
    H = toy_hamiltonian(qc.num_qubits)
    return (lambda x: float(Statevector(qc.assign_parameters(dict(zip(params, x)))).expectation_value(H).real),
            "toy energy")


def chosen_indices(n_evals, n_params, points):
    """Evaluation indices to compile: `points` evenly spaced from the first evaluation after COBYLA's initial
    simplex (index n_params + 1) to the last. Never index 0 (the start point). When the run is too short for
    that, start right after index 0."""
    first = n_params + 1 if n_evals - (n_params + 1) >= points else 1
    first = min(first, n_evals - 1)
    return sorted(set(np.linspace(first, n_evals - 1, points).round().astype(int).tolist()))


def trajectory(qc, f, x0, maxiter, points):
    """COBYLA from x0; returns (`points` (evaluation, x, f) chosen by `chosen_indices`, the number of
    evaluations)."""
    log = []

    def g(x):
        v = f(x)
        log.append((len(log), [float(t) for t in x], v))
        return v

    minimize(g, x0, method="COBYLA", options=dict(maxiter=maxiter, rhobeg=0.3))
    idx = chosen_indices(len(log), len(x0), points)
    return [log[i] for i in idx], len(log)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ansatz", default="qaoa-12-p2,qaoa-16-p3,uccsd-6-2,uccsd-8-2")
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wait-verification", type=float, default=300.0)
    ap.add_argument("--max-sim-qubits", type=int, default=20)
    ap.add_argument("--skip-qiskit", action="store_true", help="4colors arms only (the API record)")
    ap.add_argument("--skip-4colors", action="store_true", help="Qiskit arm only (the timed record; no API calls)")
    a = ap.parse_args(argv)
    res = bk.Results(Path(a.out) / "results.jsonl")
    client = None if a.skip_4colors else bk.client()
    arms = [x for x in ("4colors", "qiskit-o3")
            if not (x == "4colors" and a.skip_4colors) and not (x == "qiskit-o3" and a.skip_qiskit)]
    for spec in a.ansatz.split(","):
        qc, inst = az.build(spec)
        params = list(qc.parameters)
        f, what = objective(spec, qc, inst)
        x0 = np.full(len(params), 0.3) if inst is not None else np.zeros(len(params))   # UCCSD starts at HF
        t = time.process_time()
        pts, n_evals = trajectory(qc, f, x0, a.maxiter, a.points)
        base = dict(ansatz=spec, n_qubits=qc.num_qubits, input_n_parameters=len(params), objective=what,
                    instance=inst["id"] if inst else None, edges=inst["edges"] if inst else None,
                    exact_optimum=float(core.cost_vector(inst).max()) if inst else None,
                    optimum_certificate="brute-force" if inst else None, seed=a.seed, budget_s=a.budget,
                    input_two_qubit_gates=ak.gate_stats(qc)["two_qubit_gates"])
        res.add(kind="trajectory", n_evaluations=n_evals, start="hartree-fock (all zero)" if inst is None else "0.3",
                start_compiled=False, points=[dict(eval=e, f=v, max_abs_angle=float(np.max(np.abs(x))))
                                           for e, x, v in pts], optimizer_cpu_s=round(time.process_time() - t, 2),
                **base)
        for arm in arms:                                     # parametric: once, the same at every point
            if arm == "4colors":
                out, rec = ak.compile_4colors(qc, budget_s=a.budget, seed=a.seed, client=client,
                                              approximation_mode="exact", wait_verification_s=a.wait_verification)
            else:
                out, rec = ak.compile_qiskit(qc, budget_s=a.budget, seed=a.seed)
            res.add(kind="compile", mode="parametric", **{**base, **rec})
            print(spec, arm, "parametric", rec.get("two_qubit_gates"), flush=True)
        for e, x, v in pts:
            bound = qc.assign_parameters(dict(zip(params, x)))
            small = int(np.sum(np.abs(np.mod(np.asarray(x) + np.pi, 2 * np.pi) - np.pi) < 1e-2))
            point = dict(eval=e, f=v, x=x, max_abs_angle=float(np.max(np.abs(x))), angles_below_0_01=small)
            for arm, mode in (("4colors", "exact"), ("4colors", "per_block"), ("qiskit-o3", "exact")):
                if arm not in arms:
                    continue
                if arm == "4colors":
                    out, rec = ak.compile_4colors(bound, budget_s=a.budget, seed=a.seed, client=client,
                                                  approximation_mode=mode, wait_verification_s=a.wait_verification)
                else:
                    out, rec = ak.compile_qiskit(bound, budget_s=a.budget, seed=a.seed)
                    rec["qiskit_approximation_degree"] = 1.0           # transpile default: no approximation
                fid = None
                if out is not None:
                    chk = ak.bound_states_match(bound, out, {}, max_qubits=a.max_sim_qubits)
                    fid = chk["fidelity"]
                res.add(kind="compile", mode=f"bound-{mode}", fidelity_here=fid,
                        **point, **{**base, **rec})
                print(spec, "eval", e, arm, mode, rec.get("two_qubit_gates"), rec.get("verification"),
                      rec.get("distance"), flush=True)


if __name__ == "__main__":
    main()
