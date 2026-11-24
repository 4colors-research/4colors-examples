"""Compile the post's circuits for FakeFez with its published calibration, both arms, and keep every circuit.

    python run.py --out results/obs [--circuits uccsd_8,...] [--budgets 10] [--seeds 1,2,3]

Arms (the same device, the same published calibration, both exact):

* 4colors: fourcolors.quantum.compiler.compile(qc, target=targets.from_qiskit(FakeFez()), time_budget_s=T,
  seed=s, approximation_mode="exact", verify="skip");
* qiskit: transpile(optimization_level=3) on FakeFez().target as shipped (its calibration as-is, approximation
  degree 1.0, i.e. exact), new seeds until the budget is spent, best kept (fewest two-qubit gates).

Every compiled circuit is written to <out>/circuits/<circuit>__<arm>__s<seed>__b<budget>.qpy (Qiskit's pick is
rebuilt from its recorded seed and checked to have the same counts), so `simulate.py` can run the noise model on
exactly what each arm returned. Rows also carry `estimated_esp`, a model from the calibration snapshot (never a
measurement) and `coupler_counts` (two-qubit gates per physical coupler, from the saved circuit; the arms check
compares it). Refusals are rows, never dropped.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit
import domain_circuits as circuits
import circuitkit as ck

POST = "compiler-noisy-observables"


def circuit_path(out_dir, circuit_id, arm, seed, budget):
    return os.path.join(out_dir, "circuits", f"{circuit_id}__{arm}__s{seed}__b{budget:g}.qpy")


save, load = circuits.save_qpy, circuits.load_qpy


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10", seeds="1,2,3")
    p.add_argument("--circuits", default=",".join(circuits.POST_CIRCUITS), help="circuit ids (domain_circuits.py)")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    fez = ck.fez()
    target4 = None if args.skip_4colors else ck.heavy_hex_target()
    failed = ck.failed_couplers(fez.target)
    out.add(kind="device", post=POST, device="FakeFez", n_qubits=fez.target.num_qubits,
            failed_couplers=sorted(failed), calibration="published snapshot shipped with qiskit-ibm-runtime",
            backend_version=getattr(fez, "backend_version", None))
    client = None if args.skip_4colors else examplekit.client()
    for cid in [c for c in args.circuits.split(",") if c]:
        qc, spec = circuits.build(cid)
        circ = ck.circuit_fields(qc, cid, spec["family"])
        out.add(kind="circuit", post=POST, **circ, spec=spec, trivial=circuits.trivial_reason(qc))
        for seed in args.seeds:
            if not args.skip_4colors:
                for budget in args.budgets:
                    res, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=target4, coupler_set="fakefez",
                                               seed=seed, budget_s=budget, failed=failed, client=client,
                                               approximation_mode="exact")
                    if res is not None and res.circuit is not None:
                        compiled = res.to_qiskit()
                        r["circuit_file"] = save(compiled, circuit_path(args.out, cid, "4colors", seed, budget))
                        r["estimated_esp"] = ck.estimated_esp(compiled, fez.target)
                        r["coupler_counts"] = ck.coupler_counts(compiled)
                    out.add(**r)
                    print(f"{cid} seed {seed} {budget:g}s: 4colors {r['two_qubit_gates']} "
                          f"{r.get('refusal_code') or ''}", flush=True)
            if not args.skip_qiskit:
                for r in ck.qiskit_rows(qc, {"full": fez.target}, args.budgets, seed, post=POST, circuit=circ,
                                        failed=failed, max_trials=args.max_qiskit_trials):
                    if r["kind"] == "result":
                        compiled = rebuild(qc, fez.target, r)
                        r["circuit_file"] = save(compiled, circuit_path(args.out, cid, "qiskit", seed, r["budget_s"]))
                        r["estimated_esp"] = ck.estimated_esp(compiled, fez.target)
                        r["coupler_counts"] = ck.coupler_counts(compiled)
                    out.add(**r)
    print(f"wrote {out.path}")


def rebuild(qc, target, row):
    """Qiskit's picked circuit, rebuilt from its recorded seed (transpile is deterministic for a fixed seed)."""
    from qiskit import transpile

    compiled = transpile(qc, target=target, optimization_level=3, seed_transpiler=row["qiskit_seed"])
    if ck.circuit_metrics(compiled)["two_qubit_gates"] != row["two_qubit_gates"]:
        raise RuntimeError(f"rebuilt Qiskit circuit for {row['circuit_id']} differs from its trial")
    return compiled


if __name__ == "__main__":
    main()
