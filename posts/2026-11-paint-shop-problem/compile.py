"""Compile the paint shop QAOA circuit for a heavy-hex chip: 4colors against Qiskit O3, same seconds.

    python compile.py --out runs/qaoa [--sizes 20,50,100] [--layers 1,2] [--budgets 10,30] [--seeds 1,2]

For each car count n (instance seed 1), the circuit is `paintshop.qaoa_circuit(word, p)`: one rzz per coupled car
pair per layer, measurements included. Both arms compile the same circuit for FakeFez with its published
calibration:

* qiskit: transpile(optimization_level=3), seeds until the budget is spent, best kept (circuitkit.qiskit_rows), on
  the full device map as shipped (primary) and on the working couplers (labeled extra);
* 4colors: fourcolors.quantum.compiler.compile(qc, target=targets.from_qiskit(FakeFez()), time_budget_s=T, seed=s).

The approximation mode is the same for both: "exact" (Qiskit O3 does not approximate unless asked). Every row
also records the modeled ESP on the published calibration (circuitkit.estimated_esp; a model, never a measurement).
"""
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import examplekit
import circuitkit as ck
import paintshop as ps

POST = "paint-shop-problem"


def circuit(n: int, p: int, instance: int = 1):
    word = ps.random_word(n, seed=instance)
    qc = ps.qaoa_circuit(word, p=p)
    J, _ = ps.ising_couplings(word)
    info = ck.circuit_fields(qc, f"bpsp_n{n}_i{instance}_p{p}", "paint_shop_qaoa")
    return qc, {**info, "cars": n, "layers": p, "instance": instance, "couplings": len(J)}


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10,30", seeds="1,2", sizes="20,50,100")
    p.add_argument("--layers", type=ck.ints, default=[1, 2], help="QAOA layers p (default 1,2)")
    p.add_argument("--instance", type=int, default=1, help="instance seed of the word")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    coupler_sets, failed = ck.heavy_hex_coupler_sets()
    full = coupler_sets["full"]
    out.add(kind="device", post=POST, name="FakeFez", n_qubits=full.num_qubits, couplers=ck.device_couplers(full),
            failed=sorted(failed))
    client = None if args.skip_4colors else examplekit.client()
    for n in args.sizes:
        for layers in args.layers:
            qc, circ = circuit(n, layers, args.instance)
            for seed in args.seeds:
                if not args.skip_qiskit:
                    for r in ck.qiskit_rows(qc, coupler_sets, args.budgets, seed, post=POST, circuit=circ,
                                            failed=failed, max_trials=args.max_qiskit_trials, esp_target=full):
                        out.add(**r)
                if not args.skip_4colors:
                    for budget in args.budgets:
                        res, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                                   coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed, budget_s=budget,
                                                   failed=failed, client=client, approximation_mode="exact")
                        if res is not None and res.circuit is not None:
                            r["estimated_esp"] = ck.estimated_esp(res.to_qiskit(), full)
                        out.add(**r)
                        print(f"{circ['circuit_id']} seed {seed} budget {budget:g}s: 4colors "
                              f"{r['two_qubit_gates']} two-qubit gates {r['refusal_code'] or ''}", flush=True)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
