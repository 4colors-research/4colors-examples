"""The compiler report card: 29 families x 3 sizes x 2 instances, compiled for a heavy-hex chip in the same seconds.

    python run.py --out results/card                          # everything (522 compiles at 10 s, plus Qiskit)
    python run.py --out results/smoke --families tfim2d --sizes-index 0 --instances 101 --seeds 1
    python run.py --out results/card --skip-4colors           # the Qiskit side only (no API calls)
    python run.py --out results/card --skip-qiskit --dq-from results/qiskit/results.jsonl

For every circuit (families.py, lowered to standard gates, measurements included):

* a "circuit" row: id, family, domain, size, instance, qubits, input two-qubit gates, generator version and the
  sha256 of the exact OpenQASM 3 text compiled (`families.instance_sha256`);
* the D2 pre-pass (dq.py): Qiskit's measured distance from the input, d_q, or "unmeasured";
* Qiskit 2.x `transpile(optimization_level=3)` on FakeFez as shipped (its published calibration as-is), on
  `--qiskit-streams` parallel streams, each trying new seeds until T seconds are spent: arm "qiskit-o3-best" is the
  best trial within T over all streams, arm "qiskit-o3-once" is the first trial of stream 0 (one default call);
* 4colors through the public API: `fourcolors.quantum.compiler.compile(qc, target=targets.from_qiskit(FakeFez()),
  time_budget_s=T, seed=s, verify="wait")`, exact unless d_q > 0 (then `max_approximation_error=d_q`).

Every row has the same keys (circuitkit.ROW_KEYS); a refusal is a row, never dropped.
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit
import circuitkit as ck
import dq
import families as F

POST = "compiler-report-card"


def generator_commit() -> str | None:
    """The last commit that changed families.py (pinned in every run's meta row), or None outside git."""
    try:
        return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", "families.py"], cwd=HERE,
                                       text=True, stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        return None


def _stream(job):
    """One Qiskit stream (runs in its own process): seeds 1000 * stream_seed + k until `budget` is spent."""
    family, size, inst, stream_seed, budget, max_trials = job
    qc, _ = F.make(family, size, inst)
    sets, failed = ck.heavy_hex_coupler_sets()
    return ck.qiskit_trials(qc, sets["full"], budget, stream_seed, failed, max_trials=max_trials)


def qiskit_streams(family, size, inst, seed, budget, streams, max_trials=None, pool=None):
    """Trials of every stream; stream s uses stream seed 100 * seed + s."""
    jobs = [(family, size, inst, 100 * seed + s, budget, max_trials) for s in range(streams)]
    results = list(pool.map(_stream, jobs)) if pool is not None else [_stream(j) for j in jobs]
    return [{**t, "stream": s} for s, trials in enumerate(results) for t in trials]


def qiskit_rows(qc, circ, seed, budget, trials):
    rows = []
    for t in trials:
        rows.append(ck.row(kind="trial", post=POST, **circ, arm="qiskit", coupler_set="full", seed=seed,
                           budget_s=budget, wall_s=t["trial_s"], **{k: v for k, v in t.items() if k != "trial_s"}))
    in_budget = [t for t in trials if t["cumulative_s"] <= budget]   # each stream has its own clock
    best = ck.best_within(trials, budget)
    first = next(t for t in trials if t["stream"] == 0)
    for arm, pick, over in (("qiskit-o3-best", best, not in_budget), ("qiskit-o3-once", first, False)):
        rows.append(ck.row(kind="result", post=POST, **circ, arm=arm, coupler_set="full", seed=seed,
                           budget_s=budget, status="finished", engine_version=ck.qiskit_version(),
                           wall_s=pick["cumulative_s"] if arm == "qiskit-o3-best" else pick["trial_s"],
                           over_budget=over, trials_in_budget=len(in_budget), qiskit_seed=pick["qiskit_seed"],
                           stream=pick["stream"], two_qubit_gates=pick["two_qubit_gates"],
                           two_qubit_depth=pick["two_qubit_depth"], depth=pick["depth"],
                           gates_on_failed_couplers=pick["gates_on_failed_couplers"]))
    return rows


def load_dq(paths):
    """{circuit_id: d_q} from earlier "dq" rows (the Qiskit record), so the API record uses the same d_q."""
    out = {}
    for r in examplekit.load_rows(paths):
        if r.get("kind") == "dq":
            out[r["circuit_id"]] = r.get("dq")
    return out


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10", seeds="1,2,3")
    p.add_argument("--families", type=lambda s: [x for x in s.split(",") if x], default=list(F.FAMILIES),
                   help="comma list (default: all 29)")
    p.add_argument("--sizes-index", type=ck.ints, default=[0, 1, 2], help="which of each family's 3 sizes")
    p.add_argument("--instances", type=ck.ints, default=list(F.INSTANCES))
    p.add_argument("--qiskit-streams", type=int, default=2,
                   help="parallel Qiskit streams = the plan's compile threads (default 2) [CHECK /v1/license]")
    p.add_argument("--skip-dq", action="store_true", help="no D2 pre-pass (every 4colors compile exact)")
    p.add_argument("--dq-from", nargs="*", default=[], help="reuse d_q rows from these records")
    args = p.parse_args(argv)
    unknown = set(args.families) - set(F.FAMILIES)
    if unknown:
        p.error(f"unknown families: {sorted(unknown)}")
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args), generator_version=F.GENERATOR_VERSION, generator_changes=F.GENERATOR_CHANGES,
            generator_commit=generator_commit(), families=len(args.families))
    sets, failed = ck.heavy_hex_coupler_sets()
    known_dq = load_dq(args.dq_from)
    client = None if args.skip_4colors else examplekit.client()
    pool = None if args.skip_qiskit or args.qiskit_streams < 2 else ProcessPoolExecutor(args.qiskit_streams)
    try:
        for family, size, inst in F.plan(args.families, args.sizes_index, args.instances):
            qc, meta = F.make(family, size, inst)
            circ = ck.circuit_fields(qc, qc.name, family)
            out.add(kind="circuit", post=POST, **circ, domain=meta["domain"], size=size, instance=inst,
                    sha256=F.instance_sha256(qc), generator_version=F.GENERATOR_VERSION,
                    meta={k: v for k, v in meta.items() if k not in ("graph",)})
            d = known_dq.get(circ["circuit_id"])
            if circ["circuit_id"] not in known_dq and not args.skip_dq and not args.skip_qiskit:
                d_row = dq.measure_dq(qc, sets["full"], qiskit_seed=100000)
                d = d_row["dq"]
                out.add(kind="dq", post=POST, **circ, **d_row)
            for seed in args.seeds:
                for budget in args.budgets:
                    if not args.skip_qiskit:
                        trials = qiskit_streams(family, size, inst, seed, budget, args.qiskit_streams,
                                                args.max_qiskit_trials, pool)
                        for r in qiskit_rows(qc, circ, seed, budget, trials):
                            out.add(**r)
                    if not args.skip_4colors:
                        opts = dq.ours_options(d)
                        _, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                                 coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed, budget_s=budget,
                                                 failed=failed, client=client, verify="wait", **opts)
                        out.add(**r, dq=d, instance=inst, size=size)
                        print(f"{circ['circuit_id']} seed {seed} {budget:g}s: 4colors {r['two_qubit_gates']} 2q "
                              f"{r['verification_label'] or ''} {r['refusal_code'] or ''}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
