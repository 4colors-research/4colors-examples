"""Check every compiled output of a device dir against its input, and measure how far it is from it.

    python check.py out/<device> [--max-active 24] [--states 3] [--qcec]

Writes out/<device>/checks.jsonl (or --out DIR/checks.jsonl), one row per output file: {circuit, stem, status, method, distance, ...}.
  status   PASS (equivalent within 1e-9), APPROX (a measured distance above that, at most --fail-above),
           FAIL (above it, or a wrong secret), UNKNOWN (no method could decide: the cell is not shown)
  method   clifford_secret   Bernstein-Vazirani: run from |0...0> (Aer matrix-product-state, 64 shots), the measured
                             bits must spell the secret with certainty. Exact for BV; it is the circuit as run.
           random_states_k   k random product states on the input's qubits, every other active qubit in |0>: the
                             output must be the input's state at the measured final positions, other qubits back
                             in |0>. distance = 1 - the lowest state fidelity. A sampled measurement, not a proof.
           qcec              optional (--qcec, needs `pip install mqt.qcec`): a formal equivalence check; PASS only.
The distance a rival output measures here is the bound ours gets (run.py, rule D2). Whether it is the same
quantity as the service's `max_approximation_error` must be confirmed before the run (a check made before the run).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import circuits  # noqa: E402
import sbs  # noqa: E402

PASS_TOL = 1e-9


def _load(text):
    from qiskit import qasm3

    return qasm3.loads(text)


def _measures(qc):
    """{clbit index: qubit index} of the measure instructions."""
    return {qc.find_bit(i.clbits[0]).index: qc.find_bit(i.qubits[0]).index
            for i in qc.data if i.operation.name == "measure"}


def bv_secret(inp) -> str:
    n = inp.num_qubits - 1
    ones = {inp.find_bit(i.qubits[0]).index for i in inp.data if i.operation.name == "cx"}
    return "".join("1" if q in ones else "0" for q in range(n))


def clifford_secret(out, secret: str, shots: int = 64) -> dict:
    from qiskit_aer import AerSimulator

    counts = AerSimulator(method="matrix_product_state").run(out, shots=shots, seed_simulator=1).result().get_counts()
    want = secret[::-1]                                   # Qiskit prints clbit 0 last
    keys = {k.replace(" ", "")[-len(secret):] for k in counts}
    ok = keys == {want}
    return {"status": "PASS" if ok else "FAIL", "method": "clifford_secret", "distance": 0.0 if ok else None,
            "shots": shots}


def random_states(inp, out, initial, k: int = 3, max_active: int = 24, seed: int = 1) -> dict:
    """See the module docstring. `initial[l]` is the physical qubit logical qubit l starts on."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    n = inp.num_qubits
    in_meas = _measures(inp)                              # clbit -> logical
    out_meas = _measures(out)                             # clbit -> physical
    final = {in_meas[c]: out_meas[c] for c in in_meas if c in out_meas}
    if initial is None or len(final) != n:
        return {"status": "UNKNOWN", "method": f"random_states_{k}", "distance": None,
                "note": "initial or final layout unknown"}
    gates = [i for i in out.data if i.operation.name not in ("measure", "barrier", "delay")]
    active = sorted({out.find_bit(q).index for i in gates for q in i.qubits} | set(initial) | set(final.values()))
    if len(active) > max_active:
        return {"status": "UNKNOWN", "method": f"random_states_{k}", "distance": None, "active_qubits": len(active),
                "note": f"more than {max_active} active qubits"}
    pos = {p: i for i, p in enumerate(active)}
    logical = inp.remove_final_measurements(inplace=False)
    rng = np.random.default_rng(seed)
    x = np.arange(2 ** n)
    emb = np.zeros_like(x)
    for l in range(n):
        emb |= ((x >> l) & 1) << pos[final[l]]
    worst = 1.0
    for _ in range(k):
        angles = rng.uniform(0, 2 * np.pi, size=(n, 3))
        prep_l = QuantumCircuit(n)
        prep_c = QuantumCircuit(len(active))
        for l, (t, p, lam) in enumerate(angles):
            prep_l.u(t, p, lam, l)
            prep_c.u(t, p, lam, pos[initial[l]])
        psi = Statevector(prep_l.compose(logical)).data
        for i in gates:
            prep_c.append(i.operation, [pos[out.find_bit(q).index] for q in i.qubits])
        phi = Statevector(prep_c).data
        worst = min(worst, float(abs(np.vdot(psi, phi[emb])) ** 2))
    d = max(0.0, 1.0 - worst)
    return {"status": "PASS" if d < PASS_TOL else "APPROX", "method": f"random_states_{k}", "distance": d,
            "active_qubits": len(active)}


def qcec_check(inp, out) -> dict:
    from mqt import qcec

    res = qcec.verify(inp, out)
    eq = str(res.equivalence)
    ok = eq in ("equivalent", "equivalent_up_to_global_phase", "equivalent_up_to_phase")
    return {"status": "PASS" if ok else "UNKNOWN", "method": "qcec", "distance": 0.0 if ok else None, "qcec": eq}


def check_one(inp_text: str, out_text: str, layouts: dict, family_check: str, *, states: int = 3,
              max_active: int = 24, fail_above: float = 1e-2, use_qcec: bool = False) -> dict:
    inp, out = _load(inp_text), _load(out_text)
    if family_check == "clifford":
        return clifford_secret(out, bv_secret(inp))
    if use_qcec:
        try:
            r = qcec_check(inp, out)
            if r["status"] == "PASS":
                return r
        except ImportError:
            pass
    initial = sbs.initial_layout(sbs.parse_qasm(out_text), layouts, inp.num_qubits)
    r = random_states(inp, out, initial, k=states, max_active=max_active)
    if r["distance"] is not None and r["distance"] > fail_above:
        r["status"] = "FAIL"
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("device_dir")
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--max-active", type=int, default=24)
    ap.add_argument("--fail-above", type=float, default=1e-2)
    ap.add_argument("--qcec", action="store_true")
    ap.add_argument("--out", help="dir for checks.jsonl (default: the device dir)")
    a = ap.parse_args(argv)
    os.makedirs(a.out or a.device_dir, exist_ok=True)
    path = os.path.join(a.out or a.device_dir, "checks.jsonl")
    done = {(r["circuit"], r["stem"], r["qasm_sha256"]) for r in
            (json.loads(x) for x in open(path))} if os.path.exists(path) else set()
    for name in ("rivals.jsonl", "ours.jsonl"):
        p = os.path.join(a.device_dir, name)
        for row in ([json.loads(x) for x in open(p)] if os.path.exists(p) else []):
            if not row.get("stem") or (row["circuit"], row["stem"], row["qasm_sha256"]) in done:
                continue
            cell = os.path.join(a.device_dir, row["circuit"])
            lj = os.path.join(cell, row["stem"] + ".layouts.json")
            r = check_one(open(os.path.join(cell, "in.qasm")).read(),
                          open(os.path.join(cell, row["stem"] + ".qasm")).read(),
                          json.load(open(lj)) if os.path.exists(lj) else None,
                          circuits.CIRCUITS[row["circuit"]]["check"], states=a.states, max_active=a.max_active,
                          fail_above=a.fail_above, use_qcec=a.qcec)
            r.update(circuit=row["circuit"], stem=row["stem"], arm=row["arm"], qasm_sha256=row["qasm_sha256"])
            with open(path, "a") as f:
                f.write(json.dumps(r, sort_keys=True) + "\n")
            print(f"{row['circuit']} {row['stem']}: {r['status']} {r['method']} distance {r['distance']}", flush=True)


if __name__ == "__main__":
    main()
