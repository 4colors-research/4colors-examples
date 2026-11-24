"""Solve it, then compile it: the same Max-Cut graphs through the 4colors Optimizer for QUBO (the reference answer)
and through QAOA compiled by Qiskit O3 and by the 4colors Quantum Compiler, scored by simulation under a stated
noise model.

Panels of this post (--panels, default optimizer,qaoa):
  optimizer  quality against time: `fourcolors.qubo.solve` at several time limits and seeds (ratio against the
             enumerated optimum for n <= 20, against the best found over all runs for larger graphs).
  qaoa       QAOA p = 1-3 on every certified graph (p = 4, 5 on the --deep-n graphs: the ratio-vs-p panel), angles
             optimized noiselessly (evaluations recorded) or fixed (none); compile billed seconds per 4colors job,
             transpile seeds tried per Qiskit compile; ideal and noisy ratio, the probability of sampling a cut as
             good as the optimizer's, best of S shots for every S of --shots, at every e2 of --e2-sweep (the
             break-even sweep: a threshold derived from the stated noise model, never a hardware claim). A uniformly
             random sampler is recorded next to it (`random_success_S`).

Later (not part of this post; kept for the follow-up post on warm starts and density, run only on request):
  density    G(n, p) at n = 20 and 100: compiled two-qubit gates of p = 1 QAOA, the optimizer's time to its best
             answer, and the textbook SWAP-network count on a line.
  warm       warm-start QAOA from a classical answer, against the answer itself and more classical search.

Usage:
    python run.py --out runs/solve [--panels optimizer,qaoa] [--budget 5] [--seeds 1,2,3,4,5]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
import examplekit as bk  # noqa: E402
from qaoakit import ansatz as az, apikit as ak, core  # noqa: E402


def reg3(n, s):
    return az.reg3_instance(n, s)


def gnp_instance(n, p, s):
    g = nx.gnp_random_graph(n, p, seed=s)
    return dict(id=f"gnp-n{n}-p{p:g}-s{s}", kind="maxcut", nodes=n, edges=[(int(a), int(b), 1.0) for a, b in sorted(g.edges)])


def qaoa_input(inst, struct):
    if struct == "warm-p1":
        qc, th, tw = core.build_warm_parametric(inst, 1)
        groups = dict(t=th, w=tw)
    else:
        qc, th = core.build(inst, struct)
        groups = dict(t=th)
    qc = az.to_input(qc)
    qc.measure_all()
    return qc, groups


def compile_arm(arm, qc, budget, seed, client):
    if arm == "4colors":
        return ak.compile_4colors(qc, budget_s=budget, seed=seed, client=client, approximation_mode="exact")
    return ak.compile_qiskit(qc, budget_s=budget, seed=seed)


def swap_network_two_qubit_gates(n, p=1):
    """Textbook all-to-all QAOA on a line (Kivlichan et al. 2018; O'Gorman et al. 2019): n layers of
    nearest-neighbor ZZ+SWAP blocks, n(n-1)/2 blocks per QAOA layer, 3 two-qubit gates per block (a fused ZZ and
    SWAP is one two-qubit unitary that needs three CZ). A reference count for a complete graph, analytic."""
    return 3 * p * n * (n - 1) // 2


# ---- panel: optimizer ---------------------------------------------------------------------------------

def panel_optimizer(a, client, res, cert, large):
    for inst in cert + large:
        n = inst["nodes"]
        opt = float(core.cost_vector(inst).max()) if n <= 24 else None
        limits = a.qubo_limits + (a.qubo_limits_large if n > 24 else [])
        for limit in limits:
            for s in a.seeds:
                r, rec = ak.solve_qubo(ak.maxcut_matrix(n, inst["edges"]), time_limit_s=limit, seed=s, client=client)
                cut = -rec["objective"] if rec.get("objective") is not None else None
                res.add(kind="optimizer", instance=inst["id"], n=n, edges=inst["edges"], cut=cut, exact_optimum=opt,
                        optimum_certificate="brute-force" if opt is not None else None,
                        classical_ratio=cut / opt if (opt and cut is not None) else None, **rec)
                print("optimizer", inst["id"], limit, s, cut, rec.get("refusal_code"), flush=True)


# ---- panel: qaoa + break-even sweep -----------------------------------------------------------------

def classical_reference(inst, a, client):
    """The optimizer's cut at --reference-limit (seed 1): the answer quality QAOA has to match. "Reach the
    optimizer's answer" below means: sample a cut at least this large."""
    _, rec = ak.solve_qubo(ak.maxcut_matrix(inst["nodes"], inst["edges"]), time_limit_s=a.reference_limit, seed=1,
                           client=client)
    return (-rec["objective"] if rec.get("objective") is not None else None), rec


def success(probs, cost, target, shots):
    """Probability that at least one of `shots` samples has a cut >= target."""
    q = float(np.clip(np.asarray(probs)[cost >= target - 1e-9].sum(), 0, 1))
    return 1 - (1 - q) ** shots, q


def sweep_rows(out, inst, cost, opt, target, a, e2s=None):
    """Noisy metrics of one bound compiled circuit at every e2 of the sweep (or of `e2s`): ratio, probability of the
    optimum, best of S shots, and the probability that S shots reach `target` (a cut value)."""
    nq = inst["nodes"]
    active = core.compact(out, nq)[0].num_qubits
    if active > a.max_active:
        return [dict(active_qubits=active, simulated=False)]
    rows = []
    for e2 in e2s or a.e2_sweep:
        st, probs = core.noisy_ratio(out, nq, cost, opt, e2, a.e_ro, a.trajectories, a.batches, a.threads)
        row = dict(e2=e2, e1=e2 / 10, e_ro=a.e_ro, active_qubits=active, simulated=True,
                   noisy_ratio_mean=st["mean"], noisy_ratio_se=st["se"],
                   p_opt=float(np.asarray(probs)[cost >= opt - 1e-9].sum()))
        for S in a.shots:
            row[f"best_of_{S}_ratio"] = core.best_of_shots(probs, cost, S) / opt
            if target is not None:
                row[f"success_{S}"], row["p_reach_target"] = success(probs, cost, target, S)
                row["target_cut"] = target
        rows.append(row)
    return rows


def panel_qaoa(a, client, res, cert):
    for inst in cert:
        n, cost = inst["nodes"], core.cost_vector(inst)
        opt = float(cost.max())
        target, ref = classical_reference(inst, a, client)
        ps = a.ps + (a.deep_ps if n in a.deep_n else [])
        uniform = np.full(2 ** n, 2.0 ** -n)
        base = dict(instance=inst["id"], n=n, edges=inst["edges"], exact_optimum=opt,
                    optimum_certificate="brute-force", random_ratio=core.random_ratio(cost),
                    classical_cut=target, classical_ratio=target / opt if target is not None else None,
                    classical_limit_s=a.reference_limit, classical_job_id=ref.get("job_id"),
                    reference_is_optimum=target is not None and target >= opt - 1e-9,
                    classical_billed_s=ref.get("billed_s"),
                    **{f"random_success_{S}": success(uniform, cost, target, S)[0] if target is not None else None
                       for S in a.shots})
        prev = None
        for p in ps:
            t = time.process_time()
            ang, e, evals = core.tune_numpy(n, cost, p, init=core.interp(prev, p) if prev else None,
                                            restarts=1 if prev else 4, seed=7)
            tune_cpu = time.process_time() - t
            prev = ang
            angle_sets = [("optimized", ang, e, tune_cpu, evals)]
            if p in core.FIXED_ANGLES_3REG:
                fx = core.fixed_angles(p)
                angle_sets.append(("fixed", fx, float(np.abs(core.qaoa_statevector(n, cost, fx)) ** 2 @ cost), 0.0, 0))
            qc, groups = qaoa_input(inst, f"std-p{p}")
            for seed in a.compile_seeds:
                for arm in ("4colors", "qiskit-o3"):
                    out, rec = compile_arm(arm, qc, a.budget, seed, client)
                    for name, vals, e_id, cpu, ev in angle_sets:
                        row = dict(base, kind="qaoa", variant="std", p=p, angles=name, ideal_ratio=e_id / opt,
                                   tune_cpu_s=round(cpu, 3), tune_evals=ev, **rec)
                        if out is None:
                            res.add(**row)
                            continue
                        bound = out.assign_parameters(dict(zip(groups["t"], vals)))
                        for sw in sweep_rows(bound, inst, cost, opt, target, a):
                            res.add(**row, **sw)
                    print("qaoa", inst["id"], p, arm, seed, rec.get("two_qubit_gates"), rec.get("refusal_code"),
                          flush=True)


# ---- panel: density -------------------------------------------------------------------------------

def panel_density(a, client, res):
    for n in a.density_n:
        for dens in a.densities:
            inst = gnp_instance(n, dens, a.graph_seed)
            m = len(inst["edges"])
            base = dict(instance=inst["id"], n=n, density=dens, n_edges=m, edges=inst["edges"],
                        swap_network_two_qubit_gates=swap_network_two_qubit_gates(n),
                        swap_network_note="analytic, complete graph on a line, p = 1")
            qc, _ = qaoa_input(inst, "std-p1")
            base["input_two_qubit_gates"] = ak.gate_stats(qc)["two_qubit_gates"]
            for seed in a.compile_seeds:
                for arm in ("4colors", "qiskit-o3"):
                    out, rec = compile_arm(arm, qc, a.budget, seed, client)
                    res.add(kind="density_compile", **base, **rec)
                    print("density", inst["id"], arm, seed, rec.get("two_qubit_gates"), rec.get("refusal_code"),
                          flush=True)
            opt = float(core.cost_vector(inst).max()) if n <= 24 else None
            for s in a.seeds:
                r, rec = ak.solve_qubo(ak.maxcut_matrix(n, inst["edges"]), time_limit_s=a.density_limit, seed=s,
                                       client=client)
                cut = -rec["objective"] if rec.get("objective") is not None else None
                res.add(kind="density_qubo", **base, cut=cut, exact_optimum=opt,
                        optimum_certificate="brute-force" if opt is not None else None,
                        classical_ratio=cut / opt if (opt and cut is not None) else None, **rec)


# ---- panel: warm start ----------------------------------------------------------------------------

def panel_warm(a, client, res, cert):
    """Warm-start QAOA against its own seed and against more classical search at matched compute."""
    for inst in cert:
        n, cost = inst["nodes"], core.cost_vector(inst)
        opt = float(cost.max())
        Q = ak.maxcut_matrix(n, inst["edges"])
        sources = []
        r, rec = ak.solve_qubo(Q, time_limit_s=a.warm_seed_limit, seed=1, client=client)
        if r is not None and r.solution is not None:
            sources.append(("qubo-short", list(r.solution), rec))
        sources.append(("local-1flip", core.local_optimum(inst), {}))
        qc, groups = qaoa_input(inst, "warm-p1")
        compiled = {}
        for seed in a.compile_seeds:
            for arm in ("4colors", "qiskit-o3"):
                compiled[(arm, seed)] = compile_arm(arm, qc, a.budget, seed, client)
        for src, x0, srec in sources:
            seed_cut = float(cost[core.bits_index(x0)])
            t = time.process_time()
            ang, e, evals = core.tune_numpy(n, cost, 1, x0=x0, seed=7)
            tune_cpu = time.process_time() - t
            base = dict(instance=inst["id"], n=n, edges=inst["edges"], exact_optimum=opt,
                        optimum_certificate="brute-force", seed_source=src, seed_ratio=seed_cut / opt,
                        seed_is_optimal=seed_cut >= opt - 1e-9, seed_job_id=srec.get("job_id"),
                        ideal_ratio=e / opt, tune_cpu_s=round(tune_cpu, 3), tune_evals=evals)
            binds = {**dict(zip(groups["t"], ang)), **dict(zip(groups["w"], core.warm_thetas(x0)))}
            for (arm, seed), (out, rec) in compiled.items():
                row = dict(base, kind="warm_qaoa", **rec)
                if out is None:
                    res.add(**row)
                    continue
                # success_S here: S shots find a cut strictly better than the seed answer
                for sw in sweep_rows(out.assign_parameters(binds), inst, cost, opt, seed_cut + 1e-6, a, [a.warm_e2]):
                    res.add(**row, **sw)
                if seed != a.compile_seeds[0]:
                    continue
                # more classical search at matched compute (first compile seed): the optimizer, warm-started from the
                # same answer, for as many seconds as the QAOA step cost classically (angle optimization + compile)
                matched = min(a.match_cap, max(a.qubo_limits[0], tune_cpu + (rec.get("billed_s") or rec["wall_s"])))
                r2, rec2 = ak.solve_qubo(Q, time_limit_s=round(matched, 3), seed=seed, warm_start=x0, client=client)
                cut2 = -rec2["objective"] if rec2.get("objective") is not None else None
                res.add(kind="warm_classical", **base, matched_to_arm=arm, matched_to_seed=seed,
                        matched_s=round(matched, 3), classical_ratio=cut2 / opt if cut2 is not None else None, **rec2)
            print("warm", inst["id"], src, "seed ratio", round(seed_cut / opt, 4), flush=True)


def main(argv=None, cert=None, large=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", default="optimizer,qaoa", help="optimizer,qaoa (this post); density, warm: later")
    ap.add_argument("--n", default="12,16,20", help="certified QAOA graphs (3-regular)")
    ap.add_argument("--graph-seeds", default="1,2,3")
    ap.add_argument("--large-n", default="50,100,150", help="optimizer-only graphs (best found, no certificate)")
    ap.add_argument("--seeds", default="1,2,3,4,5", help="QUBO seeds")
    ap.add_argument("--compile-seeds", default="1,2,3,4,5")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--ps", default="1,2,3")
    ap.add_argument("--deep-ps", default="4,5", help="extra depths for the ratio-vs-p panel, on --deep-n graphs only")
    ap.add_argument("--deep-n", default="12,16")
    ap.add_argument("--qubo-limits", default="0.01,0.1,1")
    ap.add_argument("--qubo-limits-large", default="10")
    ap.add_argument("--reference-limit", type=float, default=1.0)
    ap.add_argument("--e2-sweep", default="1e-4,3e-4,1e-3,3e-3,1e-2")
    ap.add_argument("--e-ro", type=float, default=0.01)
    ap.add_argument("--shots", default="100,1000")
    ap.add_argument("--trajectories", type=int, default=100)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--max-active", type=int, default=24)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--density-n", default="20,100")
    ap.add_argument("--densities", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--density-limit", type=float, default=1.0)
    ap.add_argument("--graph-seed", type=int, default=1)
    ap.add_argument("--warm-seed-limit", type=float, default=0.01)
    ap.add_argument("--match-cap", type=float, default=10.0)
    ap.add_argument("--warm-e2", type=float, default=3e-3)
    a = ap.parse_args(argv)
    ints = lambda s: [int(x) for x in s.split(",") if x]
    floats = lambda s: [float(x) for x in s.split(",") if x]
    a.seeds, a.compile_seeds, a.ps, a.shots = ints(a.seeds), ints(a.compile_seeds), ints(a.ps), ints(a.shots)
    a.density_n, a.densities = ints(a.density_n), floats(a.densities)
    a.deep_ps, a.deep_n = ints(a.deep_ps), ints(a.deep_n)
    a.qubo_limits, a.qubo_limits_large, a.e2_sweep = floats(a.qubo_limits), floats(a.qubo_limits_large), floats(a.e2_sweep)
    res = bk.Results(Path(a.out) / "results.jsonl")
    client = bk.client()
    cert = cert or [reg3(n, s) for n in ints(a.n) for s in ints(a.graph_seeds)]
    large = large if large is not None else [reg3(n, 1) for n in ints(a.large_n)]
    panels = a.panels.split(",")
    if "optimizer" in panels:
        panel_optimizer(a, client, res, cert, large)
    if "qaoa" in panels:
        panel_qaoa(a, client, res, cert)
    if "density" in panels:
        panel_density(a, client, res)
    if "warm" in panels:
        panel_warm(a, client, res, cert)


if __name__ == "__main__":
    main()
