"""Bernstein-Vazirani on a heavy-hex chip: a star-shaped circuit on a sparse device.

    python run.py --out results/bv [--sizes 50,75,100,125,150] [--budgets 10,60] [--seeds 1,2,3,4,5]
                  [--secret random|ones] [--skip-check]

For each size n the circuit is BV with an n-bit secret: a random secret fixed by n by default (about n/2 CNOTs
into one ancilla), or all ones (`--secret ones`: n CNOTs, the full star, the worst case). Both arms compile the
same circuit, measurements included:

* qiskit: transpile(optimization_level=3) for FakeFez, seeds until the budget is spent, on the working couplers
  and on the full device map (failed couplers included);
* 4colors: fourcolors.quantum.compiler.compile(qc, target=targets.from_qiskit(FakeFez()), time_budget_s=T, seed=s,
  verify="skip").

Every row also stores where the two-qubit gates landed (coupler_counts), for the heat map in plot.py, and the
result of a Clifford check of the compiled circuit (bv_check.py: run from |0...0>, does it measure the secret with
certainty?) in `clifford_check` / `clifford_check_method`, and the circuit's two-qubit blocks by class (blocks.py:
stand-alone SWAPs, CNOT-and-move blocks, plain CNOTs) in `block_counts`. Qiskit's picked circuit is rebuilt for the check from its
recorded `qiskit_seed` (transpile is deterministic for a fixed seed); the check is never part of either arm's time.
"""
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import blocks
import examplekit
import bv_check
import circuitkit as ck

POST = "compiler-bv-routing"


def bv_circuit(n: int, secret_kind: str):
    secret = "1" * n if secret_kind == "ones" else ck.random_secret(n, seed=n)
    qc = ck.bernstein_vazirani(secret)
    return qc, ck.circuit_fields(qc, f"bv_{n}_{secret_kind}", "bernstein_vazirani"), secret


def output_fields(compiled, secret):
    """What the post reads from a compiled circuit beyond the counts: the Clifford check (bv_check.py) and the
    block classes (blocks.py: SWAP blocks, CNOT-and-move blocks, ...)."""
    f = bv_check.check_fields(compiled, secret)
    f["block_counts"] = blocks.block_counts(compiled) if compiled is not None else None
    return f


def qiskit_check(qc, row, coupler_sets, secret):
    """`output_fields` of the circuit a Qiskit result row picked, rebuilt from its seed on its coupler set."""
    from qiskit import transpile

    if row.get("qiskit_seed") is None or row["coupler_set"] not in coupler_sets:
        return output_fields(None, secret)
    out = transpile(qc, target=coupler_sets[row["coupler_set"]], optimization_level=3,
                    seed_transpiler=row["qiskit_seed"])
    return output_fields(out, secret)


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10,60", sizes="50,75,100,125,150")
    p.add_argument("--secret", choices=["ones", "random"], default="random",
                   help="a random secret fixed by n (default) or all ones (the full star, the worst case)")
    p.add_argument("--skip-check", action="store_true", help="skip the Clifford check and block classes of the compiled circuits")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))

    coupler_sets, failed = ck.heavy_hex_coupler_sets()
    full = coupler_sets["full"]
    out.add(kind="device", post=POST, name="FakeFez", n_qubits=full.num_qubits,
            couplers=ck.device_couplers(full), failed=sorted(failed), named_target=ck.heavy_hex_target())
    client = None if args.skip_4colors else examplekit.client()

    for n in args.sizes:
        qc, circ, secret = bv_circuit(n, args.secret)
        out.add(kind="interaction", post=POST, **circ, secret=secret, edges=ck.interaction_edges(qc))
        for seed in args.seeds:
            if not args.skip_qiskit:
                checks = {}
                for r in ck.qiskit_rows(qc, coupler_sets, args.budgets, seed, post=POST, circuit=circ,
                                        failed=failed, max_trials=args.max_qiskit_trials, with_counts=True):
                    if r["kind"] == "result" and not args.skip_check:
                        cset = r.get("from_coupler_set") or r["coupler_set"]
                        key = (cset, r.get("qiskit_seed"))
                        if key not in checks:
                            checks[key] = qiskit_check(qc, {**r, "coupler_set": cset}, coupler_sets, secret)
                        r = {**r, **checks[key]}
                    out.add(**r)
            if not args.skip_4colors:
                for budget in args.budgets:
                    res, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                               coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed, budget_s=budget,
                                               failed=failed, with_counts=True, client=client,
                                               approximation_mode="exact")
                    if not args.skip_check:
                        r.update(output_fields(res.to_qiskit() if res is not None else None, secret))
                    out.add(**r)
                    print(f"{circ['circuit_id']} seed {seed} budget {budget:g}s: 4colors "
                          f"{r['two_qubit_gates']} two-qubit gates {r['refusal_code'] or ''} "
                          f"{r.get('clifford_check') or ''}", flush=True)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
