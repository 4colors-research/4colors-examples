"""Verified, approximate, unverified: what the label on a compiled circuit means.

    python run.py --out results/verify [--sizes 6,20,100] [--modes exact,per_block] [--seeds 1,2,3] [--budgets 10]

The QFT (h and controlled phases, many of them tiny) at up to 32 qubits, and one larger circuit that is exact as
input (QAOA p = 1 on a random 3-regular graph: no QFT above 32 qubits, whose inputs are not exact), at three sizes:

  6 qubits    QFT on a 2 x 3 grid chip (custom target): small enough to check here too, with Qiskit's Operator
  20 qubits   QFT on FakeFez (targets.from_qiskit(FakeFez()): its description and published calibration)
  100 qubits  QAOA on FakeFez: larger than an exact check covers

each with approximation_mode "exact" and "per_block", verify="wait" (the result comes back with its label
decided). Every row keeps the label, the measured distance and the reason. For circuits of at most
--local-check-max qubits both arms' outputs are also compared with the input here (Operator.from_circuit,
which follows the layout): `local_equivalent` and `local_infidelity` (1 - average gate fidelity).

One more job shows the default flow: verify="background" returns with the label "pending";
res.wait_verification() then waits for the check ("background" rows).
"""
import os
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck

POST = "compiler-verification"
SMALL_CHIP = "grid_2x3"


def small_chip():
    q = lambda r, c: r * 3 + c
    edges = [(q(r, c), q(r, c + 1)) for r in range(2) for c in range(2)] + [(q(0, c), q(1, c)) for c in range(3)]
    return ck.custom_target(6, edges)


def target_for(n):
    """The 4colors target and the Qiskit coupler sets for a QFT on n qubits."""
    if n <= 6:
        chip = small_chip()
        return chip, SMALL_CHIP, {SMALL_CHIP: ck.qiskit_target_from_custom(chip)}
    sets, _ = ck.heavy_hex_coupler_sets()
    return ck.heavy_hex_target(), ck.HEAVY_HEX_DEVICE, {"full": sets["full"]}     # Qiskit: the published map as-is


#: The largest QFT the post uses: above it the QFT input itself is not exact (brief, 2 Nov).
QFT_MAX = 32


def circuit_for(n):
    """(circuit, its fields): the QFT up to QFT_MAX qubits, QAOA p = 1 on a 3-regular graph above it."""
    if n <= QFT_MAX:
        qc = ck.qft(n)
        return qc, ck.circuit_fields(qc, f"qft_{n}", "qft")
    qc = ck.qaoa_maxcut(n, examplekit.random_regular(n, 3, seed=n), p=1)
    return qc, ck.circuit_fields(qc, f"qaoa3r_{n}", "qaoa")


def local_check(compiled, logical):
    """Compare a compiled circuit with its input here, on the input's qubits with every other chip qubit
    starting in |0> (Operator.from_circuit puts the input's qubits first and follows the layout). Returns
    (equivalent up to global phase, 1 - process fidelity). Only for small circuits: the matrix has
    4**(chip qubits) entries."""
    import numpy as np
    from qiskit.quantum_info import Operator

    full = Operator.from_circuit(compiled).data
    want = Operator(logical.remove_final_measurements(inplace=False)).data
    d = want.shape[0]
    block = full[:d, :d]                    # rows and columns where the extra qubits are |0>
    fidelity = abs(np.trace(want.conj().T @ block)) ** 2 / d ** 2
    infidelity = max(0.0, 1.0 - float(fidelity))
    return bool(infidelity < 1e-9), infidelity


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10", seeds="1,2,3", sizes="6,20,100")
    p.add_argument("--modes", default="exact,per_block", help="approximation modes (default exact,per_block)")
    p.add_argument("--local-check-max", type=int, default=10, help="largest circuit checked here (default 10)")
    p.add_argument("--wait-timeout", type=float, default=900, help="seconds to wait for a background check")
    p.add_argument("--no-background", action="store_true", help="skip the verify=background demo")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    client = None if args.skip_4colors else examplekit.client()
    budget = args.budgets[0]

    for n in args.sizes:
        qc, circ = circuit_for(n)
        target, cset, qiskit_sets = target_for(n)
        check = n <= args.local_check_max
        for seed in args.seeds:
            if not args.skip_qiskit:
                for r in ck.qiskit_rows(qc, qiskit_sets, [budget], seed, post=POST, circuit=circ,
                                        max_trials=args.max_qiskit_trials):
                    if check and r["kind"] == "result":
                        from qiskit import transpile
                        best = transpile(qc, target=qiskit_sets[r["coupler_set"]], optimization_level=3,
                                         seed_transpiler=r["qiskit_seed"])
                        r["local_equivalent"], r["local_infidelity"] = local_check(best, qc)
                    out.add(**r)
            if args.skip_4colors:
                continue
            for mode in args.modes.split(","):
                res, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=target, coupler_set=cset, seed=seed,
                                           budget_s=budget, client=client, verify="wait", approximation_mode=mode)
                if res is not None and res.circuit is not None:
                    meta = res.to_qiskit().metadata.get("fourcolors", {})
                    r["approximation_spent"] = meta.get("approximation_spent")
                    r["verification_method"] = res.verification.method
                    if check:
                        r["local_equivalent"], r["local_infidelity"] = local_check(res.to_qiskit(), qc)
                out.add(**r)
                print(f"{circ['circuit_id']} seed {seed} {mode}: {r['verification_label']} distance "
                      f"{r['verification_distance']} reason {r['verification_reason']!r} {r['refusal_code'] or ''}",
                      flush=True)

    if not args.skip_4colors and not args.no_background:
        n = args.sizes[min(1, len(args.sizes) - 1)]
        qc, circ = circuit_for(n)
        target, cset, _ = target_for(n)
        res, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=target,
                                   coupler_set=cset, seed=args.seeds[0], budget_s=budget, client=client,
                                   verify="background", approximation_mode="exact")
        r["kind"] = "background"
        r["label_at_return"] = r["verification_label"]
        if res is not None:
            t0 = time.monotonic()
            v = res.wait_verification(timeout=args.wait_timeout)
            r.update(verification_label=v.label, verification_distance=v.distance, verification_reason=v.reason,
                     waited_s=time.monotonic() - t0)
        out.add(**r)
        print(f"background: {r['label_at_return']} at return, then {r['verification_label']}", flush=True)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
