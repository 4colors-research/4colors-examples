"""A qLDPC code end to end: bound its distance with the 4colors Optimizer for QUBO, compile its syndrome round with
the 4colors Quantum Compiler onto the code's own chip and onto heavy-hex.

    python run.py --out results/scout --stage scout
    python run.py --out results/full --stage full

Distance arm: for each code and each of its k X-type logicals l, one constrained job: the least-weight Z-type error
that commutes with every X check and anticommutes with l (codes.distance_model, exact parity rows with slack bits).
Every returned answer is decoded and certified in plain Python (codes.certify); a certified answer of weight w proves
d_Z <= w. The least over the k logicals equals d_Z when every job returns its optimum (a nontrivial Z logical
anticommutes with at least one basis X logical); the post never claims to compute d, it compares with the distance
the code's paper reports. --escalate re-solves at the longer budget only the codes whose short-budget bound is
above the reported distance.

Compile arm: one syndrome-extraction round (codes.syndrome_round, measurements included) of each code that fits
156 qubits, on
* the Tanner-graph chip: one coupler per check-data incidence (targets.build_description, cz, no calibration),
  a do-no-harm check: no routing should be needed;
* heavy-hex: the FakeFez description with its published calibration (targets.from_qiskit); Qiskit gets the same
  device as-is (transpile O3, seeds until the same budget).
Both exact. Every compiled round of both arms is checked by Pauli propagation with injected errors
(codes.syndrome_check): the deterministic syndrome bits must equal H e.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit
import circuitkit as ck
import codes as cd

POST = "qldpc-end-to-end"
STAGES = {
    "scout": {"distance": ["toric_3", "bb_72_12_6"], "compile": ["bb_72_12_6"], "max_logicals": 2},
    "full": {"distance": ["bb_72_12_6", "bb_90_8_10", "bb_108_8_10", "bb_144_12_12", "bb_288_12_18"],
             "compile": ["bb_72_12_6"], "max_logicals": None},
}


def solve(model, time_limit, seed, client):
    import fourcolors
    import fourcolors.qubo

    try:
        return fourcolors.qubo.solve(model, time_limit_s=time_limit, seed=seed, client=client), None
    except fourcolors.errors.FourColorsError as exc:
        return None, {"error_code": getattr(exc, "code", None), "error": str(exc)[:400]}


def distance_rows(out, name, budget, args, client):
    """One job per logical at `budget`; returns the least certified weight (None if none)."""
    hx, hz, reported = cd.code(name)
    logicals = cd.x_logicals(hx, hz)
    if args.max_logicals is not None:
        logicals = logicals[:args.max_logicals]
    best = None
    for li, l in enumerate(logicals):
        model = cd.distance_model(hx, l)
        common = dict(kind="distance", post=POST, code=name, n=hx.shape[1], k=cd.n_logical(hx, hz),
                      reported_distance=reported, logical_index=li, logical_weight=int(l.sum()), n_vars=model.n,
                      n_rows=model.m, time_limit_s=budget, seed=args.seed, arm="4colors-qubo")
        r, err = solve(model, budget, args.seed, client)
        if r is None:
            out.add(**common, **err)
            continue
        e = None if r.solution is None else cd.error_of(r.solution, hx.shape[1])
        ok = e is not None and cd.certify(hx, l, e)
        w = int(e.sum()) if ok else None
        out.add(**common, certified=ok, weight=w, error_support=None if e is None else [int(j) for j in e.nonzero()[0]],
                model_feasible=None if r.solution is None else model.feasible(r.solution), **examplekit.result_fields(r))
        print(f"{name} logical {li} at {budget:g}s: certified={ok} weight={w} (reported d={reported})", flush=True)
        if ok and (best is None or w < best):
            best = w
    out.add(kind="distance_bound", post=POST, code=name, time_limit_s=budget, upper_bound=best,
            reported_distance=reported, logicals=len(logicals))
    return best, reported


def compile_rows(out, name, args, client):
    hx, hz, _ = cd.code(name)
    qc = cd.syndrome_round(hx, hz)
    circ = ck.circuit_fields(qc, f"syndrome_{name}", "qldpc_syndrome_round")
    out.add(kind="interaction", post=POST, **circ, code=name, tanner_couplers=len(cd.tanner_couplers(hx, hz)))
    tanner_desc = cd.tanner_description(hx, hz)
    tanner_qiskit = ck.qiskit_target_from_custom(ck.custom_target(tanner_desc["n_qubits"], tanner_desc["couplers"]))
    coupler_sets, failed = ck.heavy_hex_coupler_sets()
    chips = {"tanner": (tanner_desc, {"tanner": tanner_qiskit}, ()),
             "heavy_hex": (ck.heavy_hex_target(), {"full": coupler_sets["full"]}, failed)}
    for chip, (ours, theirs, bad) in chips.items():
        for seed in args.compile_seeds:
            if not args.skip_qiskit:
                for r in ck.qiskit_rows(qc, theirs, [args.budget], seed, post=POST, circuit=circ, failed=bad,
                                        max_trials=args.max_qiskit_trials, with_counts=False):
                    if r["kind"] == "result":
                        from qiskit import transpile
                        cset = r["coupler_set"]
                        pick = transpile(qc, target=theirs[cset], optimization_level=3,
                                         seed_transpiler=r["qiskit_seed"])
                        r = {**r, **cd.syndrome_check(pick, hx, hz)}
                    out.add(**r, chip=chip, code=name)
            if not args.skip_4colors:
                res, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ours,
                                           coupler_set="fakefez" if chip == "heavy_hex" else "tanner", seed=seed,
                                           budget_s=args.budget, failed=bad, client=client,
                                           approximation_mode="exact", verify="skip")
                r.update(cd.syndrome_check(res.to_qiskit(), hx, hz) if res is not None and res.circuit is not None
                         else {"syndrome_check": "not_checked", "detail": "no circuit"})
                out.add(**r, chip=chip, code=name)
                print(f"{name} on {chip} seed {seed}: 4colors {r.get('two_qubit_gates')} two-qubit gates "
                      f"{r.get('refusal_code') or ''} {r['syndrome_check']}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", choices=sorted(STAGES), default="full")
    p.add_argument("--codes", nargs="*", default=None, help="distance codes (default: the stage's)")
    p.add_argument("--compile-codes", nargs="*", default=None)
    p.add_argument("--budgets", type=float, nargs="+", default=[10.0, 60.0], help="QUBO seconds per job")
    p.add_argument("--escalate", action="store_true",
                   help="run the later budgets only for codes whose earlier bound is above the reported distance")
    p.add_argument("--max-logicals", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--budget", type=float, default=60.0, help="compile seconds, both compilers")
    p.add_argument("--compile-seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-qiskit-trials", type=int, default=None)
    p.add_argument("--skip-qiskit", action="store_true")
    p.add_argument("--skip-4colors", action="store_true")
    p.add_argument("--skip-distance", action="store_true")
    p.add_argument("--skip-compile", action="store_true")
    args = p.parse_args(argv)
    stage = STAGES[args.stage]
    if args.max_logicals is None:
        args.max_logicals = stage["max_logicals"]
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    client = None if args.skip_4colors else examplekit.client()
    if not args.skip_distance and not args.skip_4colors:
        for name in args.codes or stage["distance"]:
            for i, budget in enumerate(args.budgets):
                best, reported = distance_rows(out, name, budget, args, client)
                if args.escalate and best is not None and reported is not None and best <= reported:
                    break
    if not args.skip_compile:
        for name in args.compile_codes or stage["compile"]:
            compile_rows(out, name, args, client)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
