"""Compile once, bind many times: a parametric ansatz through the 4colors Quantum Compiler.

For each ansatz (10-500 parameters) this script
  1. compiles it once through the Qiskit plugin (`generate_pass_manager` on a Heron-class backend) and once
     through `compiler.compile()`, and once with Qiskit at optimization level 3 at the same budget; and, for the
     instant comparison (--instant, on by default), once through `compiler.compile()` with no time budget (the
     service's instant answer) and once with Qiskit's default `transpile` (one try, Qiskit's default level);
     4colors compiles wait up to --wait-verification seconds for the service's verification label;
  2. binds it to random values many times: two-qubit gates per binding (the same for every binding, by
     construction), binding time, and, up to --max-sim-qubits active qubits, a noiseless statevector check that
     the bound compiled circuit prepares the bound input's state (by Parameter object and by position);
  3. once (--text-demo): sends the circuit as OpenQASM 3 text, then binds the result by name, as the client
     docs say (and records whether a positional bind would have been right);
  4. runs a small COBYLA loop on the Aer estimator (noiseless, exact expectation) for one QAOA ansatz with the
     circuit compiled once (4colors, Qiskit) and compiled at every step (4colors, Qiskit), recording compile
     seconds and billed seconds per arm.

Usage:
    python run.py --out runs/p1 [--ansatz hea-8x2,qaoa-16-p3,...] [--budget 30] [--seed 1]
           [--bindings 20] [--check 5] [--loop-ansatz qaoa-12-p2] [--loop-maxiter 60] [--per-step-evals 20]
           [--wait-verification 60] [--no-instant]
Rows go to OUT/results.jsonl (one JSON object per line, see README).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
import examplekit as bk  # noqa: E402
from qaoakit import ansatz as az, apikit as ak, core  # noqa: E402

DEFAULT_ANSATZ = "hea-8x1,hea-12x4,hea-16x8,hea-20x12,qaoa-16-p1,qaoa-16-p5,ma-16,ma-20"


def backend():
    """IBM Heron-class fake backend. The plugin sends the backend's description and calibration unless given a
    `target`; the post passes `ak.device_target()` (the working couplers, no calibration) so both 4colors routes
    compile for the same map (no named targets). Qiskit gets the backend's published target (`ak.qiskit_target()`)."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    return FakeFez()


def wait_label(res, wait_s):
    """Wait up to `wait_s` seconds for the service's verification label to leave "pending" (the check runs in the
    background after the compile; it is not billed). Still pending after that: recorded as pending."""
    from fourcolors.errors import WaitTimeout

    if wait_s and res.verification.label == "pending":
        try:
            res.wait_verification(timeout=wait_s)
        except WaitTimeout:
            pass
    return res.verification


def compile_qiskit_default(qc, seed):
    """Qiskit's default: one `transpile` for FakeFez's published target (`ak.qiskit_target()`, as the O3 arm), no
    optimization_level given (Qiskit's default level), one seed. The instant comparison for the 4colors compile
    with no time budget."""
    import qiskit
    from qiskit import transpile

    t = time.perf_counter()
    c = transpile(qc, target=ak.qiskit_target(), seed_transpiler=seed)
    rec = dict(arm="qiskit-default", route="transpile-default", budget_s=None, seed=seed,
               target=ak.QISKIT_TARGET_LABEL, target_view=ak.QISKIT_TARGET_VIEW, tries=1,
               optimization_level="default", qiskit=qiskit.__version__,
               wall_s=round(time.perf_counter() - t, 3), **ak.gate_stats(c), **ak.device_score(c))
    return c, rec


def compile_pm(qc, budget, seed, client, wait_s=0.0):
    """The Qiskit plugin route: pm.run(ansatz). Returns (circuit or None, record)."""
    from fourcolors.errors import FourColorsError
    from fourcolors.quantum import compiler

    rec = dict(arm="4colors", route="generate_pass_manager", budget_s=budget, seed=seed, target=ak.TARGET_NAME,
               approximation_mode="exact")
    t = time.perf_counter()
    try:
        pm = compiler.generate_pass_manager(backend(), target=ak.device_target(), time_budget_s=budget, seed=seed,
                                            approximation_mode="exact", client=client)
        out = pm.run(qc)
    except FourColorsError as e:
        rec.update(refusal_code=getattr(e, "code", None), error=type(e).__name__, message=str(e)[:500],
                   wall_s=round(time.perf_counter() - t, 3))
        return None, rec
    res = pm.property_set["fourcolors_result"]
    v = wait_label(res, wait_s)
    rec.update(ak.gate_stats(out), **ak.device_score(out), verification=v.label, distance=getattr(v, "distance", None),
               parameters=res.parameters,
               job_id=res.job.id if res.job is not None else None, billed_s=res.billed_s,
               target_how=(pm.property_set["fourcolors_target"] or {}).get("how"),
               engine_version=(res.engine or {}).get("version"), client_version=bk.client_version(),
               wall_s=round(time.perf_counter() - t, 3))
    return out, rec


def binding_rows(spec, qc, out, rec, args, rng):
    """Bind `out` to random values: gate counts, binding time, statevector checks by object and by position."""
    params = list(qc.parameters)
    same_objects = list(out.parameters) == params
    twoq, bind_s, checks = set(), [], []
    for k in range(args.bindings):
        v = rng.uniform(-np.pi, np.pi, len(params))
        t = time.perf_counter()
        bound = out.assign_parameters(dict(zip(params, v)))        # by your own Parameter objects
        bind_s.append(time.perf_counter() - t)
        twoq.add(ak.gate_stats(bound)["two_qubit_gates"])
        if k < args.check:
            by_obj = ak.bound_states_match(qc, out, dict(zip(params, v)), max_qubits=args.max_sim_qubits)
            by_pos = ak.bound_states_match(qc, out, list(v), max_qubits=args.max_sim_qubits)
            checks.append(dict(by_object=by_obj["ok"], by_position=by_pos["ok"], fidelity=by_obj["fidelity"],
                               active_qubits=by_obj["active_qubits"]))
    oks = [c["by_object"] for c in checks] + [c["by_position"] for c in checks]
    return dict(kind="bindings", ansatz=spec, arm=rec["arm"], route=rec.get("route"), seed=rec["seed"],
                budget_s=rec["budget_s"], n_qubits=qc.num_qubits, n_parameters=len(params),
                compiled_n_parameters=len(out.parameters), same_parameter_objects=same_objects,
                res_parameters_match=(rec.get("parameters") == [p.name for p in params]
                                      if rec.get("parameters") is not None else None),
                bindings=args.bindings, two_qubit_gates_per_binding=sorted(twoq),
                bind_s_median=float(np.median(bind_s)), bind_s_max=float(np.max(bind_s)),
                checked=len(checks), checks_ok=None if None in oks else all(oks),
                min_fidelity=min((c["fidelity"] for c in checks if c["fidelity"] is not None), default=None),
                active_qubits=checks[0]["active_qubits"] if checks else None, job_id=rec.get("job_id"))


def text_demo(spec, args, client, res, rng):
    """OpenQASM 3 text in: the compiled circuit has new parameters, sorted by name; bind them by name."""
    from qiskit import QuantumCircuit, qasm3
    from qiskit.circuit import Parameter

    qc, _ = az.build(spec)
    plain = {p: Parameter(f"t{k}") for k, p in enumerate(qc.parameters)}   # t0, t1, ..., t10, ...
    qc = qc.assign_parameters(plain)
    text = qasm3.dumps(qc)
    out, rec = ak.compile_4colors(text, budget_s=args.budget, seed=args.seed, client=client, approximation_mode="exact")
    row = dict(kind="text_demo", ansatz=spec, input="openqasm3 text", **rec)
    if out is not None:
        v = rng.uniform(-np.pi, np.pi, len(plain))             # v[k] is the value of t{k}: your own order
        by_name = {f"t{k}": x for k, x in enumerate(v)}
        given = {p: by_name[p.name] for p in qc.parameters}
        names = [p.name for p in out.parameters]                # sorted by name: t0, t1, t10, t11, t2, ...
        row.update(compiled_parameter_names=names, res_parameters=rec.get("parameters"),
                   names_found=all(n in by_name for n in names))
        if row["names_found"]:
            ok = ak.bound_states_match(qc, out, given, compiled_values={p: by_name[p.name] for p in out.parameters},
                                       max_qubits=args.max_sim_qubits)
            pos = ak.bound_states_match(qc, out, given, compiled_values=list(v), max_qubits=args.max_sim_qubits)
            row.update(bind_by_name_ok=ok["ok"], bind_by_position_ok=pos["ok"])
    res.add(**row)
    print("text demo:", {k: row.get(k) for k in ("two_qubit_gates", "names_found", "bind_by_name_ok",
                                                  "bind_by_position_ok", "refusal_code")}, flush=True)


# ---- the COBYLA loop ---------------------------------------------------------------------------------

def cut_hamiltonian(inst):
    """H = -sum_edges w (1 - Z_a Z_b) / 2: its minimum is minus the maximum cut."""
    n = inst["nodes"]
    terms = [("", [], -sum(w for _, _, w in inst["edges"]) / 2)]
    terms += [("ZZ", [a, b], w / 2) for a, b, w in inst["edges"]]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def energy(compiled, values, H, n):
    """Exact noiseless <H> of the bound compiled circuit (Aer EstimatorV2, precision 0), on its active qubits."""
    from qiskit_aer.primitives import EstimatorV2

    red, where = ak.reduce_active(compiled.assign_parameters(values), compiled.layout, n)
    obs = H.apply_layout(where, num_qubits=red.num_qubits)
    est = EstimatorV2(options=dict(default_precision=0.0, backend_options=dict(max_parallel_threads=1)))
    return float(est.run([(red, obs)]).result()[0].data.evs)


def loop(args, client, res, compiled_once):
    """The same COBYLA run (same start, same maxiter) four ways: compiled once by 4colors (the plugin route) and by
    Qiskit, and compiled at every evaluation by 4colors and by Qiskit (the latter two capped at
    --per-step-evals evaluations: compile-per-step is the pattern this post argues against)."""
    from scipy.optimize import minimize

    qc, inst = compiled_once.get("input") or az.build(args.loop_ansatz)
    params = list(qc.parameters)                               # bind by these objects: the ones compiled
    H = cut_hamiltonian(inst)
    opt = float(core.cost_vector(inst).max())                  # brute force: the certified maximum cut
    x0 = np.full(len(params), 0.3)
    for arm, once in (("4colors", True), ("qiskit-o3", True), ("4colors", False), ("qiskit-o3", False)):
        log, comp_s, billed, jobs, refusals = [], 0.0, 0.0, [], []
        if once:
            out, rec = compiled_once.get((args.loop_ansatz, arm)) or (None, None)
            if out is None:
                out, rec = (compile_pm(qc, args.budget, args.seed, client) if arm == "4colors"
                            else ak.compile_qiskit(qc, budget_s=args.budget, seed=args.seed))
            if out is None:
                res.add(kind="loop", ansatz=args.loop_ansatz, arm=arm, mode="once", refusal_code=rec.get("refusal_code"))
                continue
            comp_s, billed, jobs = rec["wall_s"], rec.get("billed_s") or 0.0, [rec.get("job_id")]

        def f(x):
            nonlocal comp_s, billed
            t = time.perf_counter()
            if once:
                e = energy(out, dict(zip(params, x)), H, qc.num_qubits)
                gates = None
            else:
                bound = qc.assign_parameters(dict(zip(params, x)))
                if arm == "4colors":
                    c, r = ak.compile_4colors(bound, budget_s=args.per_step_budget, seed=args.seed, client=client,
                                              approximation_mode="exact")
                    jobs.append(r.get("job_id"))
                    billed += r.get("billed_s") or 0.0
                    if c is None:
                        refusals.append(r.get("refusal_code"))
                        raise RuntimeError(f"per-step compile refused: {r.get('refusal_code')}")
                else:
                    c, r = ak.compile_qiskit(bound, budget_s=0, seed=args.seed)
                comp_s += r["wall_s"]
                gates = r.get("two_qubit_gates")
                e = energy(c, {}, H, qc.num_qubits)
            log.append(dict(eval=len(log), energy=e, eval_s=round(time.perf_counter() - t, 4), two_qubit_gates=gates))
            return e

        maxiter = args.loop_maxiter if once else args.per_step_evals
        t0 = time.perf_counter()
        try:
            r = minimize(f, x0, method="COBYLA", options=dict(maxiter=maxiter, rhobeg=0.3))
            x_final = [float(v) for v in r.x]
        except RuntimeError as e:
            x_final, refusals = None, refusals or [str(e)]
        res.add(kind="loop", ansatz=args.loop_ansatz, arm=arm, mode="once" if once else "per_step", seed=args.seed,
                budget_s=args.budget if once else args.per_step_budget, n_parameters=len(params),
                evaluations=len(log), compile_wall_s=round(comp_s, 3), billed_s=round(billed, 3),
                loop_wall_s=round(time.perf_counter() - t0, 3), final_energy=log[-1]["energy"] if log else None,
                final_ratio=-log[-1]["energy"] / opt if log else None, exact_optimum=opt,
                optimum_certificate="brute-force", x_final=x_final, trace=log, job_ids=[j for j in jobs if j],
                refusal_codes=refusals)
        print("loop", arm, "once" if once else "per-step", len(log), "evals", round(comp_s, 2), "compile s",
              flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ansatz", default=DEFAULT_ANSATZ)
    ap.add_argument("--budget", type=float, default=30.0, help="time_budget_s, and Qiskit's matched budget")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bindings", type=int, default=20)
    ap.add_argument("--check", type=int, default=5, help="bindings checked on the statevector")
    ap.add_argument("--max-sim-qubits", type=int, default=22)
    ap.add_argument("--text-demo", default="hea-8x1", help="ansatz sent as OpenQASM 3 text ('' to skip)")
    ap.add_argument("--loop-ansatz", default="qaoa-12-p2", help="'' to skip the COBYLA loop")
    ap.add_argument("--loop-maxiter", type=int, default=60)
    ap.add_argument("--per-step-evals", type=int, default=20)
    ap.add_argument("--per-step-budget", type=float, default=1.0)
    ap.add_argument("--wait-verification", type=float, default=60.0,
                    help="seconds to wait for the service's verification label per compile (0: do not wait)")
    ap.add_argument("--no-instant", dest="instant", action="store_false",
                    help="skip the instant comparison (4colors with no budget, Qiskit's default transpile)")
    args = ap.parse_args(argv)
    res = bk.Results(Path(args.out) / "results.jsonl")
    client = bk.client()
    rng = np.random.default_rng(args.seed)
    compiled_once = {}
    specs = [s for s in args.ansatz.split(",") if s]
    if args.loop_ansatz and args.loop_ansatz not in specs:
        specs.append(args.loop_ansatz)
    for spec in specs:
        qc, _ = az.build(spec)
        routes = ["generate_pass_manager", "compile", "qiskit"]
        if args.instant:
            routes += ["compile-instant", "qiskit-default"]
        for route in routes:
            if route == "generate_pass_manager":
                out, rec = compile_pm(qc, args.budget, args.seed, client, wait_s=args.wait_verification)
            elif route in ("compile", "compile-instant"):
                out, rec = ak.compile_4colors(qc, budget_s=args.budget if route == "compile" else None,
                                              seed=args.seed, client=client, approximation_mode="exact",
                                              wait_verification_s=args.wait_verification)
                rec["route"] = route
            elif route == "qiskit":
                out, rec = ak.compile_qiskit(qc, budget_s=args.budget, seed=args.seed)
                rec["route"] = "transpile"
            else:
                out, rec = compile_qiskit_default(qc, args.seed)
            res.add(kind="compile", ansatz=spec, n_qubits=qc.num_qubits, input_n_parameters=len(qc.parameters),
                    input_two_qubit_gates=ak.gate_stats(qc)["two_qubit_gates"], **rec)
            if route in ("generate_pass_manager", "qiskit") and spec == args.loop_ansatz:
                compiled_once[(spec, rec["arm"])] = (out, rec)
                compiled_once["input"] = (qc, az.build(spec)[1])
            if out is not None:
                row = res.add(**binding_rows(spec, qc, out, rec, args, rng))
                print(spec, rec["arm"], rec["route"], rec.get("two_qubit_gates"), "checks_ok", row["checks_ok"],
                      flush=True)
            else:
                print(spec, rec["arm"], rec["route"], "refused", rec.get("refusal_code"), flush=True)
    if args.text_demo:
        text_demo(args.text_demo, args, client, res, rng)
    if args.loop_ansatz:
        loop(args, client, res, compiled_once)


if __name__ == "__main__":
    main()
