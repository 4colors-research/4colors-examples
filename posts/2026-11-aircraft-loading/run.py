"""Aircraft loading, both sides: the classical solve with exact rows, and the compile cost of the QAOA formulation.

    python run.py --out results/aero                       # both parts
    python run.py --out results/aero --part qubo           # solve + check + HiGHS reference
    python run.py --out results/aero --part compile        # QAOA circuits of the studied sizes on FakeFez

QUBO part: holds at the studied sizes (models of at most 12, 20 and 28 variables, `loading.paper_scale`) and two
larger holds (20 x 20 and 50 x 50 container-position grids), depot seeds `--holds`; each solved with
`fourcolors.qubo.solve` (exact rows, no penalty), decoded and checked in plain Python (`loading.check`), beside a
HiGHS reference on the same rows.

Compile part: `loading.qaoa_circuit` (textbook QAOA on the penalty form, p = 1 and 2) for the studied sizes,
compiled for FakeFez by 4colors (public API, `time_budget_s`, exact, verify="wait") and by Qiskit O3 (best of seeds
within the same seconds on `--qiskit-streams` streams, and one default call), published calibration as-is.
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import fourcolors
import fourcolors.qubo

import examplekit
import circuitkit as ck
import loading

POST = "aircraft-loading"
PAPER_QUBITS = (12, 20, 28)
LARGE = ((20, 20), (50, 50))


def holds(seeds):
    for s in seeds:
        for q in PAPER_QUBITS:
            yield f"paper{q}", loading.paper_scale(q, seed=s), s
        for nc, npos in LARGE:
            yield f"grid{nc}x{npos}", loading.generate(nc, npos, seed=s), s


def qubo_part(a, out, client):
    for name, h, hseed in holds(a.holds):
        m = loading.model(h)
        common = dict(hold=name, hold_seed=hseed, n_vars=h.n, n_rows=m.m, containers=len(h.weights),
                      positions=len(h.arms), payload_kg=h.payload)
        out.add(kind="instance", post=POST, **common)
        if not a.skip_reference:
            t0 = time.perf_counter()
            ref = loading.milp_reference(h, a.reference_time)
            rx = ref.pop("reference_x")
            out.add(kind="reference", post=POST, **common, method="HiGHS via scipy.optimize.milp", **ref,
                    reference_valid=None if rx is None else loading.check(h, rx)["valid"],
                    wall_s=time.perf_counter() - t0, untimed=True)
        if a.skip_4colors:
            continue
        for tl in a.time_limits:
            for seed in a.seeds:
                try:
                    r = fourcolors.qubo.solve(m, time_limit_s=tl, seed=seed, client=client)
                except fourcolors.errors.FourColorsError as exc:
                    out.add(kind="solve", post=POST, **common, time_limit_s=tl, seed=seed,
                            error_code=getattr(exc, "code", None), error=str(exc)[:300])
                    continue
                x = r.solution
                chk = loading.check(h, x) if x is not None else {"valid": False}
                out.add(kind="solve", post=POST, **common, time_limit_s=tl, seed=seed, **chk,
                        plan=loading.decode(h, x) if x is not None and h.n <= 60 else None,
                        model_feasible=None if x is None else m.feasible(x),
                        service_feasible=(r.constraints or {}).get("feasible"), **examplekit.result_fields(r))
                print(f"{name} hold {hseed} {tl:g}s seed {seed}: {chk.get('load_kg')} kg valid {chk['valid']}",
                      flush=True)


def compile_part(a, out, client):
    sets, failed = ck.heavy_hex_coupler_sets()
    for q in PAPER_QUBITS:
        h = loading.paper_scale(q, seed=a.holds[0])
        for p in a.layers:
            qc = loading.qaoa_circuit(h, p=p)
            circ = ck.circuit_fields(qc, qc.name, "aircraft_qaoa")
            extra = dict(hold=f"paper{q}", p=p, n_vars=h.n)
            out.add(kind="circuit", post=POST, **circ, **extra,
                    interacting_pairs=len(loading.penalty_qubo(h)[0]), all_pairs=h.n * (h.n - 1) // 2)
            for seed in a.seeds:
                for budget in a.budgets:
                    if not a.skip_qiskit:
                        trials = []
                        for s in range(a.qiskit_streams):
                            trials += [{**t, "stream": s} for t in ck.qiskit_trials(qc, sets["full"], budget,
                                                                                     100 * seed + s, failed)]
                        best = ck.best_within(trials, budget)
                        first = next(t for t in trials if t["stream"] == 0)
                        for arm, pick in (("qiskit-o3-best", best), ("qiskit-o3-once", first)):
                            out.add(**ck.row(kind="result", post=POST, **circ, arm=arm, coupler_set="full",
                                             seed=seed, budget_s=budget, status="finished",
                                             engine_version=ck.qiskit_version(),
                                             wall_s=pick["cumulative_s"] if arm.endswith("best") else pick["trial_s"],
                                             over_budget=best["over_budget"] if arm.endswith("best") else False,
                                             qiskit_seed=pick["qiskit_seed"],
                                             **{k: pick[k] for k in ("two_qubit_gates", "two_qubit_depth", "depth",
                                                                     "gates_on_failed_couplers")}), **extra)
                    if not a.skip_4colors:
                        _, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                                 coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed, budget_s=budget,
                                                 failed=failed, client=client, approximation_mode="exact",
                                                 verify="wait")
                        out.add(**r, **extra)
                        print(f"{circ['circuit_id']} seed {seed} {budget:g}s: 4colors {r['two_qubit_gates']} 2q",
                              flush=True)


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10", seeds="1,2,3")
    p.add_argument("--part", choices=["both", "qubo", "compile"], default="both")
    p.add_argument("--holds", type=ck.ints, default=[1, 2], help="hold seeds (the compile part uses the first)")
    p.add_argument("--time-limits", type=ck.floats, default=[1.0], help="QUBO seconds per solve")
    p.add_argument("--layers", type=ck.ints, default=[1, 2], help="QAOA layers p")
    p.add_argument("--qiskit-streams", type=int, default=2, help="the plan's compile threads [CHECK /v1/license]")
    p.add_argument("--reference-time", type=float, default=60.0)
    p.add_argument("--skip-reference", action="store_true")
    a = p.parse_args(argv)
    out = examplekit.Results(os.path.join(a.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, a))
    client = None if a.skip_4colors else examplekit.client()
    if a.part in ("both", "qubo"):
        qubo_part(a, out, client)
    if a.part in ("both", "compile"):
        compile_part(a, out, client)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
