"""The referee: best-known and certified Max-Cut values for the benchmark families, and their circuits compiled.

    python run.py --out results/ref --part qubo          # reference values (API + exhaustive certificates)
    python run.py --out results/ref --part compile       # LR-QAOA circuits on FakeFez, 4colors vs Qiskit O3
    python run.py --out results/ref                      # both

QUBO part, per instance (referee.py): Q-score's G(n, 1/2) graphs and fully connected weighted graphs, 3 instance
seeds per size:
* "instance": n, edges, total weight, Q-score's asymptotic reference n^2/8 + 0.178 n^1.5 (G(n, 1/2) only);
* "exact" (n <= --exact-max, default 24): the exhaustive maximum, a certificate;
* "solve": the 4colors Optimizer for QUBO at each --time-limits value, the cut recomputed in plain Python.

Compile part: LR-QAOA (`referee.lr_qaoa`) on fully connected weighted graphs (p = 3) and G(n, 1/2) graphs (p = 1),
the non-native shapes where routing dominates; for FakeFez with its published calibration, 4colors through the API
(`time_budget_s`, exact, verify="wait") and Qiskit O3 (best of seeds within the same seconds on `--qiskit-streams`
streams, and one default call). Native-layout and 1D-chain circuits are not compiled: no compiler can win there.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import fourcolors
import fourcolors.qubo

import examplekit
import circuitkit as ck
import referee as R

POST = "referee-qpu-benchmark"
ER_SIZES = (10, 15, 20, 24, 30, 40, 60, 80, 100, 120, 156)
FC_SIZES = (16, 24, 40, 56, 100, 156)
COMPILE = (("fc", 16, 3), ("fc", 24, 3), ("fc", 32, 3), ("fc", 40, 3), ("fc", 56, 3),
           ("er", 16, 1), ("er", 24, 1), ("er", 32, 1), ("er", 40, 1), ("er", 56, 1))


def instance(family, n, seed):
    if family == "er":
        return R.er_half(n, seed), None
    return R.fc_weighted(n, seed)


def qubo_part(a, out, client):
    for family, sizes in (("er", a.er_sizes), ("fc", a.fc_sizes)):
        for n in sizes:
            for iseed in a.instances:
                edges, w = instance(family, n, iseed)
                common = dict(family=family, n=n, instance_seed=iseed, m_edges=len(edges),
                              total_weight=sum(w.values()) if w else len(edges))
                out.add(kind="instance", post=POST, **common,
                        qscore_reference=R.expected_optimum(n) if family == "er" else None)
                if n <= a.exact_max:
                    best, x = R.exact_maxcut(n, edges, w)
                    assert R.cut_value(edges, x, w) == best
                    out.add(kind="exact", post=POST, **common, optimum=best, method="exhaustive", certified=True)
                if a.skip_4colors:
                    continue
                m = R.maxcut_model(n, edges, w)
                for tl in a.time_limits:
                    for seed in a.qubo_seeds:
                        try:
                            r = fourcolors.qubo.solve(m, time_limit_s=tl, seed=seed, client=client)
                        except fourcolors.errors.FourColorsError as exc:
                            out.add(kind="solve", post=POST, **common, time_limit_s=tl, seed=seed,
                                    error_code=getattr(exc, "code", None), error=str(exc)[:300])
                            continue
                        cut = None if r.solution is None else R.cut_value(edges, r.solution, w)
                        out.add(kind="solve", post=POST, **common, time_limit_s=tl, seed=seed, cut=cut,
                                energy_python=None if r.solution is None else m.energy(r.solution),
                                **examplekit.result_fields(r))
                        print(f"{family} n={n} inst {iseed} {tl:g}s seed {seed}: cut {cut}", flush=True)


def compile_part(a, out, client):
    sets, failed = ck.heavy_hex_coupler_sets()
    for family, n, p in COMPILE:
        if family not in a.compile_graphs or (a.compile_n and n not in a.compile_n):
            continue
        edges, w = instance(family, n, a.instances[0])
        qc = R.lr_qaoa(n, edges, w, p=p)
        circ = ck.circuit_fields(qc, qc.name + f"_{family}", f"lrqaoa_{family}")
        extra = dict(graph=family, n=n, p=p, m_edges=len(edges))
        out.add(kind="circuit", post=POST, **circ, **extra)
        for seed in a.seeds:
            for budget in a.budgets:
                if not a.skip_qiskit:
                    trials = []
                    for s in range(a.qiskit_streams):
                        trials += [{**t, "stream": s} for t in ck.qiskit_trials(qc, sets["full"], budget,
                                                                                 100 * seed + s, failed,
                                                                                 max_trials=a.max_qiskit_trials)]
                    best = ck.best_within(trials, budget)
                    first = next(t for t in trials if t["stream"] == 0)
                    for arm, pick in (("qiskit-o3-best", best), ("qiskit-o3-once", first)):
                        out.add(**ck.row(kind="result", post=POST, **circ, arm=arm, coupler_set="full", seed=seed,
                                         budget_s=budget, status="finished", engine_version=ck.qiskit_version(),
                                         wall_s=pick["cumulative_s"] if arm.endswith("best") else pick["trial_s"],
                                         over_budget=best["over_budget"] if arm.endswith("best") else False,
                                         qiskit_seed=pick["qiskit_seed"],
                                         **{k: pick[k] for k in ("two_qubit_gates", "two_qubit_depth", "depth",
                                                                 "gates_on_failed_couplers")}), **extra)
                if not a.skip_4colors:
                    _, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                             coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed, budget_s=budget,
                                             failed=failed, client=client, approximation_mode="exact", verify="wait")
                    out.add(**r, **extra)
                    print(f"{circ['circuit_id']} seed {seed} {budget:g}s: 4colors {r['two_qubit_gates']} 2q",
                          flush=True)


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="60", seeds="1,2,3")
    p.add_argument("--part", choices=["both", "qubo", "compile"], default="both")
    p.add_argument("--er-sizes", type=ck.ints, default=list(ER_SIZES))
    p.add_argument("--fc-sizes", type=ck.ints, default=list(FC_SIZES))
    p.add_argument("--instances", type=ck.ints, default=[1, 2, 3], help="instance seeds (compile uses the first)")
    p.add_argument("--time-limits", type=ck.floats, default=[1.0, 3.0, 10.0], help="QUBO seconds per solve")
    p.add_argument("--compile-graphs", type=lambda t: [x for x in t.split(",") if x], default=["fc", "er"],
                   help="which COMPILE graphs (fc, er)")
    p.add_argument("--compile-n", type=ck.ints, default=[], help="only these COMPILE sizes (default: all)")
    p.add_argument("--qubo-seeds", type=ck.ints, default=[1], help="solver seeds for the QUBO part")
    p.add_argument("--exact-max", type=int, default=24, help="largest n for the exhaustive certificate")
    p.add_argument("--qiskit-streams", type=int, default=2, help="the plan's compile threads [CHECK /v1/license]")
    a = p.parse_args(argv)
    out = examplekit.Results(os.path.join(a.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, a), qscore_lambda=R.LAMBDA, qscore_pass=R.PASS_BETA)
    client = None if a.skip_4colors else examplekit.client()
    if a.part in ("both", "qubo"):
        qubo_part(a, out, client)
    if a.part in ("both", "compile"):
        compile_part(a, out, client)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
