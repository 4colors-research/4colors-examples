"""Give it seconds: how the compiled circuit improves with the time budget.

    python run.py --out results/seconds [--budgets 1,3,10,30,60] [--seeds 1,2,3,4,5] [--sizes 15,21,27,50]

Circuits: QAOA with --layers layers (default 3) on Barabasi-Albert graphs (Benchpress style, --ba-m edges per
new node) of each size: the family whose margin over Qiskit grows with the budget, and the only one the post makes
claims about. --regular-n N adds QAOA on a random 3-regular graph with N nodes (an optional extra, off by default).

Three parts, all in one results.jsonl:

1. budget sweep: fourcolors.quantum.compiler.compile(qc, target=targets.from_qiskit(FakeFez()), time_budget_s=T, seed=s,
   verify="skip") for every budget and seed ("result" rows);
2. the stream: one submit() per circuit at the largest budget, recording every incumbent from
   job.incumbents() with the time it arrived ("incumbent" rows); --no-stream skips it;
3. Qiskit: transpile(level 3) on FakeFez as published (coupler set "full": the published calibration as-is, the
   only Qiskit arm) with new seeds until the largest budget is spent, each seed a "trial" row with the time spent
   so far, and the best within each budget as a "result" row (`over_budget` when even the first seed took longer).
"""
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck

POST = "compiler-give-it-seconds"


def build_circuits(sizes, ba_m, layers, regular_n):
    out = []
    for n in sizes:
        qc = ck.qaoa_maxcut(n, ck.barabasi_albert(n, ba_m, seed=n), p=layers)
        out.append((qc, ck.circuit_fields(qc, f"qaoa_ba{ba_m}_{n}_p{layers}", "qaoa_barabasi_albert")))
    if regular_n:
        qc = ck.qaoa_maxcut(regular_n, examplekit.random_regular(regular_n, 3, seed=regular_n), p=layers)
        out.append((qc, ck.circuit_fields(qc, f"qaoa_3reg_{regular_n}_p{layers}", "qaoa_3_regular")))
    return out


def stream(qc, circ, budget, seed, failed, client, out):
    """Submit once and record each better circuit as it arrives (then the final result, always last)."""
    for r in ck.stream_rows(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(), budget_s=budget, seed=seed,
                            failed=failed, client=client, approximation_mode="exact"):
        out.add(**r)
        print(f"  {circ['circuit_id']} {r['t_client_s']:6.1f}s {r['status']}: {r['two_qubit_gates']} two-qubit gates",
              flush=True)


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="1,3,10,30,60", sizes="15,21,27,50")
    p.add_argument("--ba-m", type=int, default=3, help="Barabasi-Albert edges per new node (default 3)")
    p.add_argument("--layers", type=int, default=3, help="QAOA layers p (default 3)")
    p.add_argument("--regular-n", type=int, default=0, help="optional 3-regular QAOA size (default 0: none)")
    p.add_argument("--no-stream", action="store_true", help="skip the incumbent stream")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    coupler_sets, failed = ck.heavy_hex_coupler_sets()
    client = None if args.skip_4colors else examplekit.client()

    for qc, circ in build_circuits(args.sizes, args.ba_m, args.layers, args.regular_n):
        print(f"{circ['circuit_id']}: {circ['qubits']} qubits, {circ['input_2q']} two-qubit gates in", flush=True)
        for seed in args.seeds:
            if not args.skip_qiskit:
                for r in ck.qiskit_rows(qc, {"full": coupler_sets["full"]}, args.budgets, seed, post=POST, circuit=circ,
                                        failed=failed, max_trials=args.max_qiskit_trials):
                    out.add(**r)
            if not args.skip_4colors:
                for budget in args.budgets:
                    _, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                             coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed, budget_s=budget, failed=failed,
                                             client=client, approximation_mode="exact")
                    out.add(**r)
                    print(f"  seed {seed} {budget:g}s: {r['two_qubit_gates']} {r['refusal_code'] or ''}", flush=True)
        if not args.skip_4colors and not args.no_stream:
            stream(qc, circ, max(args.budgets), args.seeds[0], failed, client, out)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
