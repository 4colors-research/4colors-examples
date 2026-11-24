"""Approximate QFT, measured like for like: two-qubit gates against measured distance from the exact QFT.

    python run.py --out results/aqft [--sizes 8,12,16,20,24,28,32] [--aqft-m 6,10] [--seeds 1,2,3] [--budgets 10]
                  [--qiskit-degrees 0.999999,0.9999] [--check-inputs 8] [--check-method matrix_product_state]

Inputs per size n (at most 32 qubits: above that the post has no exact QFT input; owner read pending):
  qft        the QFT (h and controlled phases, no final swaps), exact;
  aqft<m>    the textbook approximate QFT keeping controlled phases of angle at least pi / 2**m (skipped when it
             equals the QFT).

Arms (same 10 s budget, same device: FakeFez with its published calibration):
  4colors exact       on qft and on each aqft<m>       (approximation_mode="exact", sent explicitly)
  4colors per_block   on qft                           (the service may approximate each block within its budget)
  Qiskit level 3      on qft at approximation_degree 1.0 and at each --qiskit-degrees value (its own approximate
                      two-qubit synthesis), and on each aqft<m> at 1.0; seeds until the budget is spent, best kept.

Every returned circuit is scored by ONE local check for both compilers (check.infidelity): 1 - mean state fidelity
with the exact QFT over random Pauli-eigenstate product inputs (`infidelity_vs_qft`). The service's verification
label, its measured distance and `approximation_spent` are kept beside it for 4colors rows.
"""
import os
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import examplekit
import check
import circuitkit as ck

POST = "compiler-approximate-qft"
#: The largest QFT in this post (brief 2 Nov: "no QFT figure above 32 qubits"; whether that rule binds an
#: approximation post is an owner read, OD marker in the README).
QFT_MAX = 32


def inputs_for(n, ms):
    """[(input_id, circuit)] for size n: the QFT and each distinct AQFT."""
    if n > QFT_MAX:
        raise ValueError(f"QFT above {QFT_MAX} qubits is out of scope for this post")
    out = [("qft", ck.qft(n))]
    for m in ms:
        a = check.aqft(n, m)
        if a is not None:
            out.append((f"aqft{m}", a))
    return out


def qiskit_best(qc, target, budget_s, seed, degree, max_trials=None):
    """Qiskit level 3 with `approximation_degree=degree`, seeds 1000*seed + k until `budget_s` of wall time is
    spent (at least one); the fewest two-qubit gates (then depth) kept. Returns (circuit, fields)."""
    from qiskit import transpile

    best, spent, k, first_s = None, 0.0, 0, None
    while True:
        t0 = time.perf_counter()
        out = transpile(qc, target=target, optimization_level=3, seed_transpiler=1000 * seed + k,
                        approximation_degree=degree)
        dt = time.perf_counter() - t0
        first_s = dt if first_s is None else first_s
        spent += dt
        m = ck.circuit_metrics(out)
        key = (m["two_qubit_gates"], m["two_qubit_depth"], m["depth"])
        if (best is None or key < best[0]) and (spent <= budget_s or best is None):
            best = (key, out, m, 1000 * seed + k)
        k += 1
        if spent >= budget_s or (max_trials is not None and k >= max_trials):
            break
    return best[1], {**best[2], "qiskit_seed": best[3], "trials": k, "wall_s": spent,
                     "over_budget": first_s > budget_s, "approximation_degree": degree,
                     "engine_version": ck.qiskit_version(), "status": "finished"}


def score(qc_out, n, args):
    return check.infidelity(qc_out, ck.qft(n), n_inputs=args.check_inputs, seed=args.check_seed,
                            method=args.check_method, max_active=args.check_max_active)


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10", seeds="1,2,3", sizes="8,12,16,20,24,28,32")
    p.add_argument("--aqft-m", type=ck.ints, default=ck.ints("6,10"), help="AQFT: keep phases >= pi/2**m")
    p.add_argument("--qiskit-degrees", type=ck.floats, default=ck.floats("0.999999,0.9999"),
                   help="extra Qiskit approximation_degree values on the exact QFT (1.0 is always run)")
    p.add_argument("--verify", default="wait", help="4colors verify mode (default wait; waits are not billed)")
    p.add_argument("--check-inputs", type=int, default=8)
    p.add_argument("--check-seed", type=int, default=1)
    p.add_argument("--check-method", default="matrix_product_state")
    p.add_argument("--check-max-active", type=int, default=None, help="skip the check above this many qubits")
    args = p.parse_args(argv)
    if max(args.sizes) > QFT_MAX:
        p.error(f"sizes above {QFT_MAX} are out of scope")
    budget = args.budgets[0]
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    client = None if args.skip_4colors else examplekit.client()
    sets, failed = ck.heavy_hex_coupler_sets()

    for n in args.sizes:
        for input_id, qc in inputs_for(n, args.aqft_m):
            circ = {**ck.circuit_fields(qc, f"{input_id}_{n}", "qft"), "input_id": input_id}
            for seed in args.seeds:
                if not args.skip_4colors:
                    for mode in (("exact", "per_block") if input_id == "qft" else ("exact",)):
                        res, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                                   coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed, budget_s=budget,
                                                   failed=failed, client=client, verify=args.verify,
                                                   approximation_mode=mode)
                        r["mode"] = mode
                        if res is not None and res.circuit is not None:
                            qout = res.to_qiskit()
                            meta = (qout.metadata or {}).get("fourcolors", {})
                            r["approximation_spent"] = meta.get("approximation_spent")
                            r["verification_method"] = getattr(res.verification, "method", None)
                            r.update(score(qout, n, args))
                        out.add(**r)
                        print(f"{circ['circuit_id']} seed {seed} 4colors {mode}: {r['two_qubit_gates']} 2q, "
                              f"label {r['verification_label']}, check {r.get('infidelity')}", flush=True)
                if not args.skip_qiskit:
                    degrees = [1.0] + ([d for d in args.qiskit_degrees if d != 1.0] if input_id == "qft" else [])
                    for d in degrees:
                        qout, f = qiskit_best(qc, sets["full"], budget, seed, d, args.max_qiskit_trials)
                        r = ck.row(kind="result", post=POST, **circ, arm="qiskit", coupler_set="full", seed=seed,
                                   budget_s=budget, mode=f"degree {d:g}",
                                   **{**f, "gates_on_failed_couplers": ck.circuit_metrics(qout, failed)
                                      ["gates_on_failed_couplers"]})
                        r.update(score(qout, n, args))
                        out.add(**r)
                        print(f"{circ['circuit_id']} seed {seed} qiskit degree {d:g}: {r['two_qubit_gates']} 2q, "
                              f"check {r.get('infidelity')}", flush=True)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
