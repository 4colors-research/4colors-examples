"""Pauli grouping as graph coloring, priced in compiled gates.

    python run.py --out results/scout --stage scout
    python run.py --out results/full --hamlib-dir hamlib_txt --time-limit 10 --budget 3

For each Hamiltonian (built-in Fermi-Hubbard grids, plus any OpenFermion-text files in --hamlib-dir, one per file):
1. the general-commuting (GC) conflict graph: two terms conflict when they anticommute;
2. baselines (comparison arms, local): Sorted Insertion (Crawford et al. 2021) and Qiskit's `group_commuting`, both
   for GC and qubit-wise (QWC), each checked in plain Python;
3. the 4colors arm: the K-coloring model (exact one-hot rows, conflicts as the objective) solved by the 4colors
   Optimizer for QUBO at K = (best GC baseline) - 1, - 2, ... until no conflict-free answer comes back or
   --k-steps is reached; every answer decoded and checked in plain Python;
4. pricing: the diagonalization circuit of every distinct entangling group of the best valid 4colors grouping and of
   the Sorted Insertion grouping, compiled by the 4colors Quantum Compiler (target: the FakeFez description with its
   published calibration, `targets.from_qiskit`) and by Qiskit (transpile O3, seeds until the same budget), both
   exact. The modeled cost is two-qubit gates per shot averaged over the modeled shot shares (grouping.priced_cost).

Before any solve the script reads the plan limits from GET /v1/license (`license` row) and skips, as a recorded row,
any model whose rows exceed them. Every service refusal is a row, never dropped.
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
import grouping as gp

POST = "pauli-grouping-coloring"
BUILTIN = {"hubbard_2x2": (2, 2), "hubbard_3x2": (3, 2), "hubbard_4x2": (4, 2), "hubbard_3x3": (3, 3)}
SCOUT = ["hubbard_2x2", "hubbard_3x2"]


def instances(args):
    for name in args.instances:
        lx, ly = BUILTIN[name]
        yield name, "fermi_hubbard_jw_open_t1_u4", gp.fermi_hubbard(lx, ly)
    if args.hamlib_dir:
        for p in sorted(Path(args.hamlib_dir).glob("*.txt")):
            yield p.stem, "hamlib", gp.parse_openfermion(p.read_text())


def limits_row(client):
    """The plan's limits as GET /v1/license reports them (the fields whose names mention constraints, variables or
    terms are copied to the row; the whole `limits` object is kept as given)."""
    if client is None:
        return {"kind": "license", "post": POST, "limits": None, "note": "no client (offline)"}
    lic = client.license()
    lim = lic.get("limits") or (lic.get("license") or {}).get("limits") or {}
    return {"kind": "license", "post": POST, "limits": lim}


def over_limits(limits, n_vars, n_rows, nnz):
    """The first plan limit the model breaks, or None. Unknown limit names are not guessed [CHECK names on the
    live /v1/license]."""
    if not limits:
        return None
    for key, value in (("max_constraints", n_rows), ("max_variables", n_vars), ("max_nnz", nnz),
                       ("max_quadratic_terms", nnz)):
        cap = limits.get(key)
        if isinstance(cap, (int, float)) and value > cap:
            return key
    return None


def solve(model, args, seed, client):
    import fourcolors
    import fourcolors.qubo

    try:
        return fourcolors.qubo.solve(model, time_limit_s=args.time_limit, seed=seed, client=client), None
    except fourcolors.errors.FourColorsError as exc:
        return None, {"error_code": getattr(exc, "code", None), "error": str(exc)[:400]}


def grouping_row(name, arm, kind, terms, groups):
    labels = [t[0] for t in terms]
    return dict(kind="grouping", post=POST, instance=name, arm=arm, commute=kind, n_groups=len(groups),
                valid=gp.valid_grouping(labels, groups, kind), modeled_shot_factor=gp.modeled_shot_factor(terms, groups),
                groups=groups)


def coloring_sweep(out, name, terms, edges, start_k, args, client, limits):
    """K = start_k, start_k - 1, ...: one job per (K, seed). Returns the best valid grouping found (fewest groups)."""
    m = len(terms)
    labels = [t[0] for t in terms]
    best = None
    for step in range(args.k_steps):
        k = start_k - step
        if k < 1:
            break
        nnz = len(edges) * k
        common = dict(kind="solve", post=POST, instance=name, arm="4colors-qubo", commute="gc", k=k, n_terms=m,
                      n_vars=m * k, n_rows=m + 1, nnz=nnz, time_limit_s=args.time_limit)
        why = over_limits(limits, m * k, m + 1, nnz) or ("max_nnz_arg" if nnz > args.max_nnz else None)
        if why:
            out.add(**common, skipped=why)
            print(f"{name} K={k}: skipped ({why})", flush=True)
            break
        model = gp.coloring_model(m, edges, k)
        found = False
        for seed in args.seeds:
            r, err = solve(model, args, seed, client)
            if r is None:
                out.add(**common, seed=seed, **err)
                continue
            colors = None if r.solution is None else gp.decode(r.solution, m, k)
            conflicts = None if colors is None else gp.n_conflicts(colors, edges)
            groups = None if colors is None else gp.groups_from_colors(colors)
            valid = groups is not None and conflicts == 0 and gp.valid_grouping(labels, groups, "gc")
            row = out.add(**common, seed=seed, n_conflicts=conflicts, valid=valid,
                          n_groups=None if groups is None else len(groups), groups=groups if valid else None,
                          model_feasible=None if r.solution is None else model.feasible(r.solution),
                          modeled_shot_factor=gp.modeled_shot_factor(terms, groups) if valid else None,
                          **examplekit.result_fields(r))
            print(f"{name} K={k} seed={seed}: conflicts={conflicts} valid={valid}", flush=True)
            if valid:
                found = True
                if best is None or len(groups) < len(best) or (
                        len(groups) == len(best) and row["modeled_shot_factor"] < gp.modeled_shot_factor(terms, best)):
                    best = groups
        if not found:
            break
    return best


def price(out, name, arm_grouping, terms, groups, args, client, cache):
    """Compile every distinct entangling group of `groups` with both compilers; then one `priced` row per compiler."""
    labels = [t[0] for t in terms]
    per = {"4colors": [], "qiskit": []}
    coupler_sets, failed = ck.heavy_hex_coupler_sets()
    for gi, g in enumerate(groups):
        glabels = sorted(labels[i] for i in g)
        if not gp.needs_entangling(glabels):
            for a in per:
                per[a].append(0)                    # qubit-wise: one-qubit basis changes, no two-qubit gate
            continue
        key = tuple(glabels)
        if key not in cache:
            qc = gp.diagonalization_circuit(glabels)
            circ = ck.circuit_fields(qc, f"{name}_g{len(cache)}", "pauli_diagonalization")
            rows = {}
            if not args.skip_qiskit:
                q = [r for r in ck.qiskit_rows(qc, {"full": coupler_sets["full"]}, [args.budget], 1, post=POST,
                                               circuit=circ, failed=failed, max_trials=args.max_qiskit_trials)
                     if r["kind"] == "result"]
                rows["qiskit"] = q[0]
            if not args.skip_4colors:
                _, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                         coupler_set=ck.HEAVY_HEX_DEVICE, seed=1, budget_s=args.budget, failed=failed,
                                         client=client, approximation_mode="exact", verify=args.verify)
                rows["4colors"] = r
            for arm, r in rows.items():
                out.add(**r, instance=name, group_terms=len(glabels),
                        input_diagonalizes=gp.diagonalizes(qc, glabels))
            cache[key] = {a: r.get("two_qubit_gates") for a, r in rows.items()}
        for a in per:
            per[a].append(cache[key].get(a))
    for a, twoq in per.items():
        if (a == "qiskit" and args.skip_qiskit) or (a == "4colors" and args.skip_4colors):
            continue
        out.add(kind="priced", post=POST, instance=name, grouping=arm_grouping, compiler=a, n_groups=len(groups),
                budget_s=args.budget, group_two_qubit=twoq, modeled_shot_factor=gp.modeled_shot_factor(terms, groups),
                modeled_2q_per_shot=gp.priced_cost(terms, groups, twoq))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", choices=["scout", "full"], default="full")
    p.add_argument("--instances", nargs="*", default=None, choices=sorted(BUILTIN))
    p.add_argument("--hamlib-dir", default=None, help="OpenFermion-text files, one Hamiltonian each (hamlib_to_text.py)")
    p.add_argument("--time-limit", type=float, default=10.0, help="QUBO seconds per job")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--k-steps", type=int, default=3, help="how many K below the best baseline to try")
    p.add_argument("--max-nnz", type=int, default=20_000_000, help="skip models with more quadratic terms")
    p.add_argument("--budget", type=float, default=3.0, help="compile seconds per group circuit, both compilers")
    p.add_argument("--verify", default="skip")
    p.add_argument("--max-qiskit-trials", type=int, default=None)
    p.add_argument("--skip-qiskit", action="store_true")
    p.add_argument("--skip-4colors", action="store_true", help="baselines only (no API calls)")
    p.add_argument("--skip-compile", action="store_true")
    args = p.parse_args(argv)
    if args.instances is None:
        args.instances = SCOUT if args.stage == "scout" else sorted(BUILTIN)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    client = None if args.skip_4colors else examplekit.client()
    lim = out.add(**limits_row(client))["limits"]
    cache: dict = {}
    for name, source, terms in instances(args):
        labels = [t[0] for t in terms]
        edges = gp.conflict_edges(labels, "gc")
        out.add(kind="instance", post=POST, instance=name, source=source, n_qubits=len(labels[0]), n_terms=len(terms),
                n_conflicts_gc=len(edges))
        base = {}
        for kind in ("gc", "qwc"):
            for arm, fn in (("sorted_insertion", gp.sorted_insertion), ("qiskit_group_commuting", gp.qiskit_grouping)):
                g = fn(terms, kind)
                out.add(**grouping_row(name, arm, kind, terms, g))
                base[(arm, kind)] = g
        start = min(len(base[("sorted_insertion", "gc")]), len(base[("qiskit_group_commuting", "gc")])) - 1
        best = None if args.skip_4colors else coloring_sweep(out, name, terms, edges, start, args, client, lim)
        if not args.skip_compile:
            price(out, name, "sorted_insertion", terms, base[("sorted_insertion", "gc")], args, client, cache)
            if best is not None:
                price(out, name, "4colors-qubo", terms, best, args, client, cache)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
