"""Parametric circuits at 100+ qubits: compile once, check at full size, bind many times.

For each ansatz, budget and seed, the 4colors Quantum Compiler (through the public API) and Qiskit O3 (local, a
matched budget of transpile seeds) compile the same parametric circuit. Each compiled circuit is then

  1. checked to keep every parameter (count, names, and the input's own Parameter objects);
  2. checked at full size (clifford_check.py): bound at --clifford-draws random Clifford points (every parameter a
     multiple of pi/2), simulated from |0...0> as a stabilizer state, and compared with the bound input, routing
     and the final layout included. No statevector: a 150-qubit one does not fit in any memory;
  3. bound --bindings times by object to time binding (local, untimed), and once by position, compared gate by
     gate with the bind by object.

Usage:
    python run.py --out runs/large [--ansatz ma-100,ma-150,hea-ring-100x2,hea-ring-150x2,qaoa-150-p2]
                  [--budgets 10,60] [--seeds 1,2,3,4,5] [--bindings 200] [--clifford-draws 3]
                  [--arms 4colors,qiskit-o3]
Rows go to OUT/results.jsonl (one JSON object per line, see README). `--arms 4colors` and `--arms qiskit-o3` split
the run into an API record and a timed local record.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
import examplekit as bk  # noqa: E402
import clifford_check as cc  # noqa: E402
from qaoakit import ansatz as az, apikit as ak  # noqa: E402

DEFAULT_ANSATZ = "ma-100,ma-150,hea-ring-100x2,hea-ring-150x2,qaoa-150-p2"
ARMS = ("4colors", "qiskit-o3")


def binding(qc, out, k, seed):
    """Bind `out` k times by object: median and total seconds (local, untimed); plus: does a bind by position give
    the same gates as the bind by object?"""
    params = list(qc.parameters)
    rng = np.random.default_rng(seed)
    times = []
    for _ in range(k):
        v = rng.uniform(-np.pi, np.pi, len(params))
        t = time.perf_counter()
        out.assign_parameters(dict(zip(params, v)))
        times.append(time.perf_counter() - t)
    v = rng.uniform(-np.pi, np.pi, len(params))
    a, b = out.assign_parameters(dict(zip(params, v))), out.assign_parameters(v)
    same = len(a.data) == len(b.data) and all(
        x.operation.name == y.operation.name and x.qubits == y.qubits
        and np.allclose([float(p) for p in x.operation.params], [float(p) for p in y.operation.params])
        for x, y in zip(a.data, b.data))
    total = float(np.sum(times)) if times else 0.0
    return dict(bindings=k, bind_s_median=float(np.median(times)) if times else None, bind_s_total=total,
                bindings_per_s=float(k / total) if total > 0 else None, position_equals_object=bool(same),
                local_times="untimed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ansatz", default=DEFAULT_ANSATZ)
    ap.add_argument("--budgets", default="10,60")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--bindings", type=int, default=200)
    ap.add_argument("--clifford-draws", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args(argv)
    arms = [x for x in a.arms.split(",") if x]
    unknown = set(arms) - set(ARMS)
    if unknown:
        ap.error(f"unknown arms: {sorted(unknown)}")
    res = bk.Results(Path(a.out) / "results.jsonl")
    client = bk.client() if "4colors" in arms else None
    for spec in a.ansatz.split(","):
        qc, inst = az.build(spec)
        base = dict(ansatz=spec, n_qubits=qc.num_qubits, input_n_parameters=len(qc.parameters),
                    instance=inst["id"] if inst else None, input_two_qubit_gates=ak.gate_stats(qc)["two_qubit_gates"],
                    simulated=False)
        input_check = cc.stabilizer_match(qc, qc, cc.clifford_values(len(qc.parameters), 0))
        if input_check["ok"] is not True:            # every spec in this post binds to a Clifford circuit
            raise SystemExit(f"{spec}: the input does not bind to a Clifford circuit ({input_check})")
        for budget in [float(b) for b in a.budgets.split(",")]:
            for seed in [int(s) for s in a.seeds.split(",")]:
                for arm in arms:
                    if arm == "4colors":
                        out, rec = ak.compile_4colors(qc, budget_s=budget, seed=seed, client=client,
                                                      approximation_mode="exact")
                    else:
                        out, rec = ak.compile_qiskit(qc, budget_s=budget, seed=seed)
                    row = dict(base, **rec)
                    if out is not None:
                        row.update(parameters_preserved=len(out.parameters) == len(qc.parameters),
                                   same_parameter_objects=list(out.parameters) == list(qc.parameters),
                                   res_parameters_count=len(rec["parameters"]) if rec.get("parameters") else None,
                                   **cc.check_many(qc, out, a.clifford_draws, seed),
                                   **binding(qc, out, a.bindings, seed))
                    res.add(kind="compile", **row)
                    print(spec, budget, seed, arm, rec.get("two_qubit_gates"), rec.get("refusal_code"),
                          row.get("parameters_preserved"), row.get("clifford_ok"), flush=True)


if __name__ == "__main__":
    main()
