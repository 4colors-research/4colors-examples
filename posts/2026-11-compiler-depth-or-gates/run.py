"""Depth or gates: what objective="depth" changes.

    python run.py --out results/depth [--budgets 30] [--seeds 1,2,3,4,5] [--circuits qaoa_3reg_50_p2,adder_16b,qaoa_ba3_27_p3]

Each circuit is compiled twice per seed, both through fourcolors.quantum.compiler.submit with every incumbent
recorded as it arrives:

  4colors / two_qubit   objective="two_qubit" (the default, sent explicitly): the fewest two-qubit gates first,
                        then depth
  4colors / depth       objective="depth": the smallest depth first (the service leaves virtual rz gates out of
                        this depth), then two-qubit gates

Both send approximation_mode="exact" (rule D2: ours is never more approximate than Qiskit).

The last incumbent of each job is the final result (also written as a "result" row). Qiskit gets the same seconds
(level 3, seeds until the budget is spent) on FakeFez as published (coupler set "full": the published calibration
as-is, failed couplers included, the primary Qiskit arm). From the same series of seeds two picks are reported, one
per objective, so each is paired with ours at the matching objective:

  qiskit / full                     the fewest two-qubit gates of its seeds   <->  4colors / two_qubit
  qiskit / full, by two-qubit depth the smallest two-qubit depth of its seeds <->  4colors / depth
"""
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck
from fourcolors.errors import InvalidRequest, JobFailed

POST = "compiler-depth-or-gates"
OBJECTIVES = {"two_qubit": {"objective": "two_qubit"}, "depth": {"objective": "depth"}}
QISKIT_SET = "full"                                   # FakeFez as published: the primary Qiskit arm
BY_DEPTH = "full, by two-qubit depth"
#: (Qiskit pick, our matching objective, the metric the pair is judged on)
PAIRS = [(QISKIT_SET, "two_qubit", "two_qubit_gates"), (BY_DEPTH, "depth", "two_qubit_depth")]


def build(name):
    """qaoa_3reg_<n>_p<p>, qaoa_ba<m>_<n>_p<p> or adder_<bits>b."""
    if name.startswith("adder_"):
        qc = ck.draper_adder(int(name[6:-1]))
        return qc, ck.circuit_fields(qc, name, "draper_adder")
    head, n, p = name.split("_")[1], int(name.split("_")[2]), int(name.split("_p")[-1])
    if head == "3reg":
        edges = examplekit.random_regular(n, 3, seed=n)
    elif head.startswith("ba"):
        edges = ck.barabasi_albert(n, int(head[2:]), seed=n)
    else:
        raise SystemExit(f"unknown circuit {name}")
    qc = ck.qaoa_maxcut(n, edges, p=p)
    return qc, ck.circuit_fields(qc, name, "qaoa")


def by_depth_rows(rows, budgets):
    """From Qiskit's trial rows on "full": the trial with the smallest two-qubit depth (then fewest two-qubit
    gates) within each budget; the first trial if even that one overran (then `over_budget` is True)."""
    out = []
    trials = [r for r in rows if r["kind"] == "trial" and r["coupler_set"] == QISKIT_SET]
    for b in budgets:
        ok = [t for t in trials if t["cumulative_s"] <= b]
        pool = ok or trials[:1]
        if pool:
            t = min(pool, key=lambda t: (t["two_qubit_depth"], t["two_qubit_gates"], t["depth"]))
            out.append({**t, "kind": "result", "coupler_set": BY_DEPTH, "budget_s": b, "over_budget": not ok,
                        "trials_in_budget": len(ok), "status": "finished", "engine_version": ck.qiskit_version()})
    return out


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="30")
    p.add_argument("--circuits", default="qaoa_3reg_50_p2,adder_16b,qaoa_ba3_27_p3")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    coupler_sets, failed = ck.heavy_hex_coupler_sets()
    client = None if args.skip_4colors else examplekit.client()

    for name in args.circuits.split(","):
        qc, circ = build(name)
        for seed in args.seeds:
            if not args.skip_qiskit:
                rows = ck.qiskit_rows(qc, {QISKIT_SET: coupler_sets[QISKIT_SET]}, args.budgets, seed, post=POST,
                                      circuit=circ, failed=failed, max_trials=args.max_qiskit_trials)
                for r in rows + by_depth_rows(rows, args.budgets):
                    out.add(**r)
            if args.skip_4colors:
                continue
            for budget in args.budgets:
                for label, opts in OBJECTIVES.items():
                    last = None
                    try:
                        for r in ck.stream_rows(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                                budget_s=budget, seed=seed, failed=failed, client=client,
                                                coupler_set=label, approximation_mode="exact", **opts):
                            out.add(**r)
                            last = r
                    except (JobFailed, InvalidRequest) as err:     # a refusal: kept as a row
                        last = ck.row(kind="incumbent", post=POST, **circ, arm="4colors", coupler_set=label,
                                      seed=seed, budget_s=budget, **ck.refusal_fields(err))
                        out.add(**last)
                    out.add(**{**last, "kind": "result"})
                    print(f"{name} seed {seed} {label}: {last['two_qubit_gates']} two-qubit gates, depth "
                          f"{last['depth']} {last['refusal_code'] or ''}", flush=True)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
