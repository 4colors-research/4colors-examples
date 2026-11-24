"""A 2D transverse-field Ising Trotter step on a heavy-hex chip: 4colors and Qiskit O3, the same seconds.

    python run.py --out results/magnet [--grids 4x4,6x6,7x8,8x8] [--steps 1,2] [--budgets 10] [--seeds 1,2,3]
                  [--qiskit-streams 2]

For each open grid and step count (tfim.py), both arms compile the same measured circuit for FakeFez:

* qiskit: transpile(optimization_level=3) on FakeFez as shipped (published calibration as-is), `--qiskit-streams`
  streams of new seeds until the budget is spent per stream; "qiskit-o3-best" is the best trial within the budget
  over all streams, "qiskit-o3-once" the first trial (one default call);
* 4colors: fourcolors.quantum.compiler.compile(qc, target=targets.from_qiskit(FakeFez()), time_budget_s=T, seed=s,
  approximation_mode="exact", verify="wait").

Qiskit's transpile is exact on this circuit family (its default approximation_degree is 1.0); the report card's D2
pre-pass measures it on the grids small enough for a statevector check. Rows carry `per_step` metrics
(two-qubit gates / steps) and the logical floor (2 per bond per step).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit
import circuitkit as ck
import tfim

POST = "compiler-2d-magnet"


def qiskit_streams(qc, target, failed, budget, seed, streams, max_trials=None):
    """Streams run one after another here; each keeps its own clock, so the result is the same as running them
    side by side on `streams` cores (each stream gets `budget` seconds)."""
    out = []
    for s in range(streams):
        for t in ck.qiskit_trials(qc, target, budget, 100 * seed + s, failed, max_trials=max_trials):
            out.append({**t, "stream": s})
    return out


def per_step(r: dict, steps: int) -> dict:
    g, d = r.get("two_qubit_gates"), r.get("two_qubit_depth")
    return {"two_qubit_gates_per_step": None if g is None else g / steps,
            "two_qubit_depth_per_step": None if d is None else d / steps}


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10", seeds="1,2,3")
    p.add_argument("--grids", type=lambda s: [tfim.parse_grid(x) for x in s.split(",") if x],
                   default=[(4, 4), (6, 6), (7, 8), (8, 8)])
    p.add_argument("--steps", type=ck.ints, default=[1, 2])
    p.add_argument("--qiskit-streams", type=int, default=2, help="the plan's compile threads [CHECK /v1/license]")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args), boundary="open")
    sets, failed = ck.heavy_hex_coupler_sets()
    client = None if args.skip_4colors else examplekit.client()
    for rows, cols in args.grids:
        for steps in args.steps:
            qc = tfim.circuit(rows, cols, steps)
            circ = ck.circuit_fields(qc, qc.name, "tfim2d")
            common = dict(grid=[rows, cols], steps=steps, boundary="open",
                          logical_floor=tfim.logical_floor(rows, cols, steps))
            out.add(kind="circuit", post=POST, **circ, **common, bonds=len(tfim.bonds(rows, cols)),
                    max_degree=max(tfim.degrees(rows, cols).values()))
            for seed in args.seeds:
                for budget in args.budgets:
                    if not args.skip_qiskit:
                        trials = qiskit_streams(qc, sets["full"], failed, budget, seed, args.qiskit_streams,
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
                            out.add(**r, **common, **per_step(r, steps), trials=len(trials))
                    if not args.skip_4colors:
                        _, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                                 coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed, budget_s=budget,
                                                 failed=failed, client=client, approximation_mode="exact",
                                                 verify="wait")
                        out.add(**r, **common, **per_step(r, steps))
                        print(f"{circ['circuit_id']} seed {seed} {budget:g}s: 4colors {r['two_qubit_gates']} 2q "
                              f"({r['verification_label']}) {r['refusal_code'] or ''}", flush=True)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
