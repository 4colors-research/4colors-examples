"""The QAOA zoo, priced: QAOA variants compiled by Qiskit O3 and by 4colors, scored under noise against the exact
optimum, next to the 4colors Optimizer for QUBO at short budgets.

Instances: the 21 graphs of ../../qaoa-blog (15 Max-Cut: unweighted 3-regular and weighted G(n, 0.3), 12-20 nodes;
6 max-3-cut graphs, one-hot, 12-18 qubits). Exact optimum of each by enumerating every assignment.

Variants (the logical circuit, then the angles):
  std-p1..3       standard QAOA, angles optimized noiselessly (COBYLA, INTERP start for p > 1)
  fixed-p1..3     the same circuits with the tree fixed angles (Wurtz and Lykov 2021), 3-regular graphs only
  ma-p1           multi-angle QAOA (one angle per edge and per node), optimized from the std-p1 angles
  warm-local-p1   warm-start QAOA (Egger et al. 2021, eps 0.25) from a 1-flip local optimum
  warm-qubo-p1    the same from the QUBO optimizer's answer at the shortest budget (seed 1)
  rqaoa-p1        recursive QAOA, p = 1, eliminating to 8 nodes, then enumeration. Two rows, never mixed:
                  - the `variant` row: "recursion simulated classically" (exact p = 1 correlations, which are
                    classically computable); no circuit is compiled for it and it carries no gate price;
                  - `rqaoa_noisy` rows (--rqaoa-noisy-seeds): every recursion step's circuit compiled by the arm,
                    simulated under noise, and its edge correlations estimated from --rqaoa-shots samples; priced
                    over all steps (two-qubit gates summed over the n - 8 compiled circuits, shots summed).
  xy-p1, xy-p2    XY-mixer QAOA on max-3-cut (one-hot states kept by construction)

One parametric circuit per structure (std-pk, ma-p1, warm-p1, xy-pk) is compiled per arm and seed; every variant
that shares it binds its own angles. Same logical circuit for both compilers, same budget, seeds 1-5.

Usage:
    python run.py --out runs/zoo [--only reg3-n12] [--budget 5] [--seeds 1,2,3,4,5] [--e2 3e-3]
                  [--trajectories 100] [--batches 5] [--qubo-limits 0.01,1] [--qubo-seeds 1,2,3,4,5]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
import examplekit as bk  # noqa: E402
from qaoakit import ansatz as az, apikit as ak, core  # noqa: E402

VARIANTS = ["std-p1", "std-p2", "std-p3", "fixed-p1", "fixed-p2", "fixed-p3", "ma-p1", "warm-local-p1",
            "warm-qubo-p1", "rqaoa-p1", "xy-p1", "xy-p2"]


def structure_circuit(inst, struct):
    """The parametric input circuit both compilers get (INPUT_BASIS, measure_all) and its parameter groups."""
    if struct == "warm-p1":
        qc, th, tw = core.build_warm_parametric(inst, 1)
        groups = dict(t=th, w=tw)
    else:
        qc, th = core.build(inst, struct)
        groups = dict(t=th)
    qc = az.to_input(qc)
    qc.measure_all()
    return qc, groups


def classical_bar(inst, a, client, res, base):
    """The QUBO optimizer on the same problem at each short budget and seed. Returns the answer at the shortest
    budget, seed 1 (the warm start of warm-qubo-p1), and its ratio."""
    n, opt = inst["nodes"], base["exact_optimum"]
    model = ak.maxcut_matrix(n, inst["edges"]) if inst["kind"] == "maxcut" else ak.max3cut_model(n, inst["edges"])
    first = (None, None)
    for limit in a.qubo_limits:
        for s in a.qubo_seeds:
            r, rec = ak.solve_qubo(model, time_limit_s=limit, seed=s, client=client)
            ratio = -rec["objective"] / opt if rec.get("objective") is not None else None
            res.add(kind="classical", classical_ratio=ratio, **base, **rec)
            if first[0] is None and r is not None and r.solution is not None:
                first = (list(r.solution), ratio)
            print(inst["id"], "qubo", limit, s, ratio, rec.get("refusal_code"), flush=True)
    return first


def logical_variants(inst, cost, a, x_qubo):
    """Angles and noiseless ratio for every variant that applies: {variant: dict(struct, values, ideal_ratio, ...)}."""
    n, opt = core.n_qubits(inst), float(cost.max())
    out = {}
    want = set(a.variants)
    if inst["kind"] == "maxcut":
        prev = None
        for p in (1, 2, 3):
            t = time.process_time()
            ang, e, ev = core.tune_numpy(n, cost, p, init=core.interp(prev, p) if prev else None,
                                         restarts=1 if prev else 4, seed=7)
            prev = ang
            out[f"std-p{p}"] = dict(struct=f"std-p{p}", values=dict(t=ang), ideal_ratio=e / opt, angles="optimized",
                                    tune_cpu_s=time.process_time() - t, tune_evals=ev)
            if inst["id"].startswith("reg3"):
                fx = core.fixed_angles(p)
                e = float(np.abs(core.qaoa_statevector(n, cost, fx)) ** 2 @ cost)
                out[f"fixed-p{p}"] = dict(struct=f"std-p{p}", values=dict(t=fx), ideal_ratio=e / opt, angles="fixed",
                                          tune_cpu_s=0.0, tune_evals=0)
        if "ma-p1" in want:
            qc, th = core.build(inst, "ma-p1")
            g, b = out["std-p1"]["values"]["t"]
            t = time.process_time()
            ang, e, ev = core.tune_circuit(qc, th, cost, restarts=1, init=[g] * len(inst["edges"]) + [b] * n,
                                           maxiter=a.ma_maxiter)
            out["ma-p1"] = dict(struct="ma-p1", values=dict(t=ang), ideal_ratio=e / opt, angles="optimized",
                                tune_cpu_s=time.process_time() - t, tune_evals=ev)
        for name, x0 in (("warm-local-p1", core.local_optimum(inst)), ("warm-qubo-p1", x_qubo)):
            if x0 is None:
                continue
            t = time.process_time()
            ang, e, ev = core.tune_numpy(n, cost, 1, x0=x0, seed=7)
            out[name] = dict(struct="warm-p1", values=dict(t=ang, w=core.warm_thetas(x0)), ideal_ratio=e / opt,
                             angles="optimized", tune_cpu_s=time.process_time() - t, tune_evals=ev,
                             warm_start=list(x0), warm_start_ratio=float(cost[core.bits_index(x0)] / opt))
        t = time.process_time()
        x, steps = core.rqaoa(n, inst["edges"], seed=7)
        out["rqaoa-p1"] = dict(struct="std-p1", values=None, ideal_ratio=float(cost[core.bits_index(x)] / opt),
                               angles="optimized per step", tune_cpu_s=time.process_time() - t, steps=steps)
    else:
        prev = None
        for p in (1, 2):
            t = time.process_time()
            init = core.interp(prev, p) if prev else None
            ang, e = core.tune(inst, f"xy-p{p}", cost, restarts=3 if prev else 6, init=init, seed=7)
            prev = ang
            out[f"xy-p{p}"] = dict(struct=f"xy-p{p}", values=dict(t=ang), ideal_ratio=e / opt, angles="optimized",
                                   tune_cpu_s=time.process_time() - t)
    return {k: v for k, v in out.items() if k in want}


def score(bound, inst, cost, opt, v, a):
    """Noiseless check, then the noisy ratio at every e2 (mean, standard error over batches)."""
    nq = core.n_qubits(inst)
    active = core.compact(bound, nq)[0].num_qubits
    if active > a.max_active:
        return dict(active_qubits=active, simulated=False), []
    e0, _ = core.noisy_expectation(bound, nq, cost, 0.0, e_ro=0.0, threads=a.threads)
    chk = dict(active_qubits=active, simulated=True, check_ok=abs(e0 / opt - v["ideal_ratio"]) < 1e-6)
    noisy = []
    for e2 in a.e2:
        st, _ = core.noisy_ratio(bound, nq, cost, opt, e2, a.e_ro, a.trajectories, a.batches, a.threads)
        noisy.append(dict(e2=e2, e1=e2 / 10, e_ro=a.e_ro, noisy_ratio_mean=st["mean"], noisy_ratio_se=st["se"]))
    return chk, noisy


def sampled_zz(m, probs, pairs, shots, rng):
    """<Z_a Z_b> estimated from `shots` samples of the distribution `probs` (shots <= 0: the exact expectation)."""
    if shots <= 0:
        return core.zz_from_probs(m, probs, pairs)
    p = np.clip(np.asarray(probs, float), 0, None)
    counts = rng.multinomial(shots, p / p.sum())
    return core.zz_from_probs(m, counts / shots, pairs)


def rqaoa_noisy(inst, arm, seed, a, client, tot):
    """RQAOA where every step's correlations come from that step's circuit, compiled by `arm`, simulated with the
    first e2 of --e2 and sampled with --rqaoa-shots shots (angles tuned noiselessly per step). Returns
    (assignment, steps); per-step and total gates, shots and API use go to `tot`, so the price covers every step."""
    rng = np.random.default_rng(1000 * seed + 7)

    def corr(m, edges, step):
        sub = dict(id=f"{inst['id']}-step{step}", kind="maxcut", nodes=m, edges=edges)
        c = core.maxcut_cost(m, edges)
        ang, _, _ = core.tune_numpy(m, c, 1, restarts=4, seed=7)
        qc, groups = structure_circuit(sub, "std-p1")
        bound_in = qc.assign_parameters(dict(zip(groups["t"], ang)))
        out, rec = (ak.compile_4colors(bound_in, budget_s=a.budget, seed=seed, client=client,
                                       approximation_mode="exact") if arm == "4colors"
                    else ak.compile_qiskit(bound_in, budget_s=a.budget, seed=seed))
        tot["job_ids"].append(rec.get("job_id"))
        tot["billed_s"] += rec.get("billed_s") or 0.0
        tot["compiles"] += 1
        if out is None:
            tot["refusal_codes"].append(rec.get("refusal_code"))
            tot["refused_at_step"] = step
            raise RuntimeError(rec.get("refusal_code"))
        tot["two_qubit_gates_total"] += rec["two_qubit_gates"]
        tot["two_qubit_gates_per_step"].append(rec["two_qubit_gates"])
        probs, _ = core.noisy_probs(out, m, a.e2[0], a.e_ro, a.trajectories, seed=1, threads=a.threads)
        tot["shots_total"] += max(0, a.rqaoa_shots)
        return sampled_zz(m, probs, [(u, v) for u, v, _ in edges], a.rqaoa_shots, rng)

    return core.rqaoa(inst["nodes"], inst["edges"], correlations=corr, seed=7)


def main(argv=None, instances=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated instance id prefixes (default: all 21)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--e2", default="3e-3")
    ap.add_argument("--e-ro", type=float, default=0.01)
    ap.add_argument("--trajectories", type=int, default=100)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--max-active", type=int, default=24)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ma-maxiter", type=int, default=2000)
    ap.add_argument("--qubo-limits", default="0.01,1")
    ap.add_argument("--qubo-seeds", default="1,2,3,4,5")
    ap.add_argument("--rqaoa-noisy-seeds", default="", help="compiler seeds for the noisy recursion ('' = none)")
    ap.add_argument("--rqaoa-shots", type=int, default=1000,
                    help="shots per recursion step for the correlation estimates (0 = exact noisy expectation)")
    a = ap.parse_args(argv)
    a.variants = a.variants.split(",")
    a.e2 = [float(x) for x in a.e2.split(",")]
    a.qubo_limits = [float(x) for x in a.qubo_limits.split(",")]
    a.qubo_seeds = [int(x) for x in a.qubo_seeds.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    res = bk.Results(Path(a.out) / "results.jsonl")
    client = bk.client()
    for inst in instances or core.instances():
        if not inst["id"].startswith(tuple(a.only.split(","))):
            continue
        cost = core.cost_vector(inst)
        opt = float(cost.max())                        # enumeration of every assignment: a certificate
        base = dict(instance=inst["id"], problem=inst["kind"], n=inst["nodes"], n_qubits=core.n_qubits(inst),
                    edges=inst["edges"], exact_optimum=opt, optimum_certificate="brute-force",
                    # zero-gate baseline: a uniformly random cut, or a uniformly random coloring (2/3 of the weight)
                    random_ratio=(core.random_ratio(cost) if inst["kind"] == "maxcut"
                                  else 2 / 3 * sum(w for _, _, w in inst["edges"]) / opt))
        x_qubo, cl_ratio = classical_bar(inst, a, client, res, base)
        base_c = dict(base, classical_ratio=cl_ratio, classical_limit_s=a.qubo_limits[0])
        logical = logical_variants(inst, cost, a, x_qubo)
        for name, v in logical.items():
            res.add(kind="variant", pricing="recursion simulated classically" if v["values"] is None else None,
                    variant=name, p=int(name[-1]), structure=v["struct"], ideal_ratio=v["ideal_ratio"],
                    angles=v["angles"], tune_cpu_s=round(v["tune_cpu_s"], 3), tune_evals=v.get("tune_evals"),
                    values={k: list(map(float, x)) for k, x in v["values"].items()} if v["values"] else None,
                    warm_start_ratio=v.get("warm_start_ratio"), rqaoa_steps=v.get("steps"), **base_c)
        # rqaoa-p1 has no single circuit to price: it is scored by its recursion (rqaoa_noisy), never here
        priced = {k: v for k, v in logical.items() if v["values"] is not None}
        for struct in dict.fromkeys(v["struct"] for v in priced.values()):
            qc, groups = structure_circuit(inst, struct)
            users = {k: v for k, v in priced.items() if v["struct"] == struct}
            for seed in seeds:
                for arm in ("4colors", "qiskit-o3"):
                    out, rec = (ak.compile_4colors(qc, budget_s=a.budget, seed=seed, client=client,
                                                   approximation_mode="exact") if arm == "4colors"
                                else ak.compile_qiskit(qc, budget_s=a.budget, seed=seed))
                    for name, v in users.items():
                        row = dict(base_c, kind="qaoa", variant=name, p=int(name[-1]), structure=struct,
                                   ideal_ratio=v["ideal_ratio"], angles=v["angles"], **rec)
                        if out is None:                            # refused: a row with its code, no price
                            res.add(**row)
                            continue
                        binds = {p: x for g, vals in v["values"].items() for p, x in zip(groups[g], vals)}
                        chk, noisy = score(out.assign_parameters(binds), inst, cost, opt, v, a)
                        for nz in noisy or [{}]:
                            res.add(**row, **chk, **nz)
                    print(inst["id"], struct, arm, seed, rec.get("two_qubit_gates"), rec.get("refusal_code"),
                          flush=True)
        if "rqaoa-p1" in logical and a.rqaoa_noisy_seeds:
            for seed in [int(s) for s in a.rqaoa_noisy_seeds.split(",")]:
                for arm in ("4colors", "qiskit-o3"):
                    tot = dict(two_qubit_gates_total=0, two_qubit_gates_per_step=[], compiles=0, shots_total=0,
                               billed_s=0.0, job_ids=[], refusal_codes=[], refused_at_step=None)
                    try:
                        x, steps = rqaoa_noisy(inst, arm, seed, a, client, tot)
                        ratio = float(cost[core.bits_index(x)] / opt)
                    except RuntimeError:
                        steps, ratio = None, None
                    # priced over every step: two_qubit_gates is the sum over all compiled step circuits
                    res.add(kind="rqaoa_noisy", variant="rqaoa-p1", p=1, arm=arm, seed=seed, budget_s=a.budget,
                            e2=a.e2[0], e_ro=a.e_ro, trajectories=a.trajectories, shots_per_step=a.rqaoa_shots,
                            noisy_ratio_mean=ratio, noisy_ratio_se=None, rqaoa_steps=steps,
                            two_qubit_gates=tot["two_qubit_gates_total"] if ratio is not None else None,
                            pricing="all recursion steps", **tot, **base_c)


if __name__ == "__main__":
    main()
