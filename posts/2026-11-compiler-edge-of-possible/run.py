"""Where does a compiled circuit stop working on a real chip? Compile, then score.

    python run.py compile --out DIR [--backend ibm_kingston|fakefez] [--sizes 32,48,64,80,96,112,128]
                          [--budget 60] [--seed 1]
    python run.py score   --out DIR --counts DIR/counts.json

compile: one calibration snapshot of the device (saved), then for every size n a Bernstein-Vazirani circuit
with a random (n-1)-bit secret fixed by n, compiled by both arms against that same snapshot:

* qiskit: transpile(optimization_level=3) with new seeds until --budget seconds are spent, the seed with the
  highest estimated success (ESP, from the same snapshot) kept: Qiskit's best settings plus a noise-aware pick;
* 4colors: fourcolors.quantum.compiler.compile(qc, target=targets.from_qiskit(backend),
  time_budget_s=<Qiskit's measured wall>, seed=s): the public API, as any customer calls it.

Every compiled circuit is checked noiselessly (MPS simulation: the secret in every shot) and written as QASM 3,
and pubs.json lists the PUBs of ONE job (both arms, two repetitions each, shuffled with a fixed seed). This
script never submits anything to hardware: the job goes out through your own IBM account (README step 3).

score: reads counts.json ({pub key: counts}) and writes cells.jsonl, pairs.jsonl and edges.json with the
rules of edge.py.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit  # noqa: E402
import circuitkit as ck  # noqa: E402
import edge  # noqa: E402

POST = "compiler-edge-of-possible"
ARMS = ("qiskit", "4colors")
REPS = 2


def bv_circuit(n: int):
    """BV on n qubits: an (n-1)-bit secret fixed by n (seed n), at least one 1 bit. Returns qc, fields, answer
    (the counts key Qiskit prints for the secret: bit 0 last)."""
    secret = ck.random_secret(n - 1, seed=n)
    if "1" not in secret:
        secret = secret[:-1] + "1"
    qc = ck.bernstein_vazirani(secret)
    return qc, ck.circuit_fields(qc, f"bv_{n}", "bernstein_vazirani"), secret[::-1]


def get_backend(name: str):
    if name == "fakefez":
        return ck.fez()
    from qiskit_ibm_runtime import QiskitRuntimeService      # your saved IBM account

    return QiskitRuntimeService().backend(name)


def noiseless_ok(qc, answer: str):
    """True if the compiled circuit, run without noise, returns the secret with probability 1. BV compiles to a
    Clifford circuit, so a stabilizer computation checks it at any width (156 qubits included). None if the
    circuit is not recognized as Clifford (then the check is recorded as not done, never as passed)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford, StabilizerState

    meas = {qc.find_bit(i.clbits[0]).index: qc.find_bit(i.qubits[0]).index
            for i in qc.data if i.operation.name == "measure"}
    body = QuantumCircuit(qc.num_qubits)
    for i in qc.data:
        if i.operation.name not in ("measure", "barrier", "delay"):
            body.append(i.operation, [qc.find_bit(q).index for q in i.qubits])
    try:
        state = StabilizerState(Clifford(body))
    except Exception:                                  # not Clifford (an rz angle off pi/2): no check
        return None
    qargs = [meas[c] for c in range(len(answer))]       # clbit 0 first; Qiskit prints it last
    return abs(state.probabilities_dict_from_bitstring(answer, qargs=qargs)[answer] - 1.0) < 1e-9


def qiskit_arm(qc, target, budget_s: float, seed: int, failed):
    trials = ck.qiskit_trials(qc, target, budget_s, seed, failed, keep_circuits=True, esp_target=target)
    pick = max(trials, key=lambda t: ((t["estimated_esp"] or 0.0), -t["two_qubit_gates"]))
    wall = trials[-1]["cumulative_s"]
    return pick, wall, len(trials)


def compile_cmd(a):
    from qiskit import qasm3

    os.makedirs(a.out, exist_ok=False)
    out = examplekit.Results(os.path.join(a.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, a))
    backend = get_backend(a.backend)
    target = backend.target                                    # Qiskit gets the published calibration as-is
    failed = ck.failed_couplers(target)
    snapshot = ck.calibration_from_target(target, source=f"{a.backend} snapshot")
    with open(os.path.join(a.out, "snapshot.json"), "w") as f:
        json.dump(snapshot, f)
    out.add(kind="device", post=POST, name=a.backend, n_qubits=target.num_qubits, failed=sorted(failed))
    from fourcolors.quantum import targets

    ours_target = targets.from_qiskit(backend)                 # the same description and calibration
    client = None if a.skip_4colors else examplekit.client()
    pubs = []
    for n in a.sizes:
        qc, circ, answer = bv_circuit(n)
        pick, wall, n_trials = qiskit_arm(qc, target, a.budget, a.seed, failed)
        qk = pick.pop("circuit")
        compiled = {"qiskit": qk}
        qrow = ck.row(kind="result", post=POST, **circ, arm="qiskit", coupler_set=a.backend, seed=a.seed,
                      budget_s=a.budget, wall_s=wall)
        qrow.update({k: pick[k] for k in ("qiskit_seed", "estimated_esp", "two_qubit_gates", "two_qubit_depth",
                                          "depth", "gates_on_failed_couplers")}, trials=n_trials, answer=answer)
        out.add(**qrow)
        t_ours = max(1, math.ceil(wall))                       # same time as Qiskit used, whole seconds
        res, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ours_target, coupler_set=a.backend,
                                   seed=a.seed, budget_s=t_ours, failed=failed, client=client,
                                   with_counts=False, approximation_mode="exact")
        if res is not None and res.circuit is not None:
            compiled["4colors"] = res.to_qiskit()
            r["estimated_esp"] = ck.estimated_esp(compiled["4colors"], target)
        out.add(**r, answer=answer)
        for arm, c in compiled.items():
            ok = noiseless_ok(c, answer)
            path = os.path.join(a.out, f"bv_{n}.{arm}.qasm")
            with open(path, "w") as f:
                f.write(qasm3.dumps(c))
            out.add(kind="check", post=POST, circuit_id=circ["circuit_id"], arm=arm, noiseless_ok=ok, qasm=path)
            for rep in range(REPS):
                pubs.append({"key": f"bv_{n}|{arm}|{rep}", "n": n, "arm": arm, "rep": rep, "qasm": path,
                             "answer": answer, "shots": a.shots, "noiseless_ok": ok})
        print(f"bv_{n}: qiskit {pick['two_qubit_gates']} 2q in {wall:.1f}s, 4colors "
              f"{r.get('two_qubit_gates')} 2q at {t_ours}s {r.get('refusal_code') or ''}", flush=True)
    order = edge.pub_order([p["key"] for p in pubs])
    by_key = {p["key"]: p for p in pubs}
    with open(os.path.join(a.out, "pubs.json"), "w") as f:
        json.dump({"backend": a.backend, "order_seed": edge.ORDER_SEED, "pubs": [by_key[k] for k in order]}, f,
                  indent=1)
    print(f"wrote {out.path} and {len(pubs)} PUBs to pubs.json")


def score_cmd(a):
    with open(os.path.join(a.out, "pubs.json")) as f:
        pubs = json.load(f)["pubs"]
    with open(a.counts) as f:
        counts = json.load(f)
    cells, pairs = [], []
    for n in sorted({p["n"] for p in pubs}):
        per_arm = {}
        for arm in ARMS:
            reps = [p for p in pubs if p["n"] == n and p["arm"] == arm]
            if not reps or any(p["key"] not in counts for p in reps):
                continue
            answer = reps[0]["answer"]
            rc = [counts[p["key"]] for p in reps]
            s = edge.score_counts(edge.merge_counts(rc), answer)
            s.update(n=n, arm=arm, aa_z=edge.aa_z(rc[0], rc[1], answer) if len(rc) == 2 else None,
                     majority_ok=edge.bitwise_majority(edge.merge_counts(rc)) == answer)
            s["aa_ok"] = s["aa_z"] is not None and s["aa_z"] < edge.AA_Z
            per_arm[arm] = s
            cells.append(s)
        if len(per_arm) == 2:
            v = edge.pair_verdict(per_arm["4colors"], per_arm["qiskit"], per_arm["4colors"]["aa_z"],
                                  per_arm["qiskit"]["aa_z"])
            pairs.append({"n": n, **v})
    for name, rows in (("cells.jsonl", cells), ("pairs.jsonl", pairs)):
        with open(os.path.join(a.out, name), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    edges = {arm: edge.edge(cells, arm) for arm in ARMS}
    with open(os.path.join(a.out, "edges.json"), "w") as f:
        json.dump({"edges": edges, "shown": [p["n"] for p in edge.shown(pairs)]}, f, indent=1)
    for p in pairs:
        print(f"bv_{p['n']}: {p['verdict']} z {p['z']:.2f} edge_flip {p['edge_flip']}")
    print("edges:", {k: v["edge_n"] for k, v in edges.items()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--out", required=True)
    c.add_argument("--backend", default="fakefez")
    c.add_argument("--sizes", type=ck.ints, default=ck.ints("32,48,64,80,96,112,128"))
    c.add_argument("--budget", type=float, default=60.0, help="Qiskit's seed budget in seconds; ours gets its wall")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--shots", type=int, default=2000, help="shots per PUB; each arm runs twice (A/A)")
    c.add_argument("--skip-4colors", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--counts", required=True)
    a = p.parse_args(argv)
    (compile_cmd if a.cmd == "compile" else score_cmd)(a)


if __name__ == "__main__":
    main()
