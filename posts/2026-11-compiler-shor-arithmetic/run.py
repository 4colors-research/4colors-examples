"""Ripple-carry adders up to 100 qubits on a heavy-hex chip: 4colors and Qiskit O3, the same seconds, every
compiled adder checked on random sums.

    python run.py --out results/adders [--kinds cuccaro,vbe] [--budgets 10] [--largest-budget 60] [--seeds 1,2,3]

For each adder (adders.py: Cuccaro and VBE, three sizes each, the largest 100 qubits), both arms compile the same
block (no measurements) for FakeFez:

* qiskit: transpile(optimization_level=3) on FakeFez as shipped (published calibration as-is), `--qiskit-streams`
  streams of new seeds until the budget is spent per stream; arms "qiskit-o3-best" and "qiskit-o3-once";
* 4colors: compile(qc, target=targets.from_qiskit(FakeFez()), time_budget_s=T, seed=s, approximation_mode="exact",
  verify="wait").

Every compiled circuit (Qiskit's picks rebuilt from their recorded seed) gets `adders.check`: random a, b and carry
in, placed by the compiled circuit's layout, simulated, compared with the classical sum. The check is never part of
either arm's time. `--largest-budget` adds a second, longer budget for the 100-qubit adders only.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import adders
import examplekit
import circuitkit as ck

POST = "compiler-shor-arithmetic"


def toffolis(qc) -> int:
    return sum(1 for i in qc.data if i.operation.name == "ccx")


def qiskit_streams(qc, target, failed, budget, seed, streams, max_trials=None):
    """Streams one after another, each with its own clock (= `streams` cores side by side)."""
    out = []
    for s in range(streams):
        out += [{**t, "stream": s} for t in ck.qiskit_trials(qc, target, budget, 100 * seed + s, failed,
                                                             max_trials=max_trials)]
    return out


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10", seeds="1,2,3")
    p.add_argument("--kinds", type=lambda s: [x for x in s.split(",") if x], default=["cuccaro", "vbe"])
    p.add_argument("--bits", type=ck.ints, default=None, help="override the sizes (n bits) for every kind")
    p.add_argument("--largest-budget", type=float, default=60.0, help="extra budget for the largest size (0: none)")
    p.add_argument("--qiskit-streams", type=int, default=2, help="the plan's compile threads [CHECK /v1/license]")
    p.add_argument("--check-trials", type=int, default=8)
    p.add_argument("--skip-check", action="store_true")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    sets, failed = ck.heavy_hex_coupler_sets()
    full = sets["full"]
    client = None if args.skip_4colors else examplekit.client()
    for kind in args.kinds:
        sizes = args.bits or adders.SIZES[kind]
        for n in sizes:
            qc = adders.circuit(kind, n)
            circ = ck.circuit_fields(qc, qc.name, kind)
            tof = toffolis(qc)
            common = dict(bits=n, toffolis=tof, toffoli_counted_input=circ["input_2q"] + 6 * tof)
            out.add(kind="circuit", post=POST, **circ, **common, registers={k: len(v) for k, v in
                                                                             adders.registers(kind, n).items()})
            budgets = list(args.budgets) + ([args.largest_budget] if args.largest_budget and n == max(sizes) else [])
            for seed in args.seeds:
                for budget in budgets:
                    if not args.skip_qiskit:
                        from qiskit import transpile

                        trials = qiskit_streams(qc, full, failed, budget, seed, args.qiskit_streams,
                                                args.max_qiskit_trials)
                        best = ck.best_within(trials, budget)
                        first = next(t for t in trials if t["stream"] == 0)
                        for arm, pick in (("qiskit-o3-best", best), ("qiskit-o3-once", first)):
                            r = ck.row(kind="result", post=POST, **circ, arm=arm, coupler_set="full", seed=seed,
                                       budget_s=budget, status="finished", engine_version=ck.qiskit_version(),
                                       wall_s=pick["cumulative_s"] if arm.endswith("best") else pick["trial_s"],
                                       over_budget=best["over_budget"] if arm.endswith("best") else False,
                                       qiskit_seed=pick["qiskit_seed"],
                                       **{k: pick[k] for k in ("two_qubit_gates", "two_qubit_depth", "depth",
                                                               "gates_on_failed_couplers")})
                            if not args.skip_check:
                                rebuilt = transpile(qc, target=full, optimization_level=3,
                                                    seed_transpiler=pick["qiskit_seed"])
                                r.update(adders.check(rebuilt, kind, n, trials=args.check_trials, seed=seed))
                            out.add(**r, **common)
                    if not args.skip_4colors:
                        res, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                                   coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed, budget_s=budget,
                                                   failed=failed, client=client, approximation_mode="exact",
                                                   verify="wait")
                        if res is not None and not args.skip_check:
                            r.update(adders.check(res.to_qiskit(), kind, n, trials=args.check_trials, seed=seed))
                        out.add(**r, **common)
                        print(f"{circ['circuit_id']} seed {seed} {budget:g}s: 4colors {r['two_qubit_gates']} 2q "
                              f"{r.get('adder_check') or ''} ({r['verification_label']}) {r['refusal_code'] or ''}",
                              flush=True)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
