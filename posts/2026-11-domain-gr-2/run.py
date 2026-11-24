"""MIS on King's-lattice unit-disk graphs, from the published 289-atom size to a million atoms, through the API.

    python run.py --out runs/gr2 --atoms 289 1000 --time-limit 1 --seeds 1
    python run.py --out runs/gr2 --atoms 289 10000 100000 1000000 --time-limit 1 10 --seeds 1 2 3
    python run.py --out runs/gr2 --atoms 289 1000 --arms sa-fill --time-limit 1 --seeds 1   # rival arm [OD-2]

Arms: `rows` (maximize the set, one exact row per edge), `penalty` (the textbook penalty QUBO, P = 2), and the
open-source rival `sa-fill` (dwave-samplers simulated annealing on the penalty QUBO, one anneal filling the budget).
Every answer is decoded and checked in plain Python: edges violated, the set size after the usual repair, the gap to
the 2 x 2 block bound (a certificate at every size) and, up to --exact-max atoms, to the exact optimum (HiGHS).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors
import fourcolors.qubo

import examplekit
import kings

ARMS = ["rows", "penalty", "sa-fill"]
KEEP_SET_UP_TO = 2000            # store the chosen atoms (for the lattice figure) on instances this small


def instance(atoms: int, filling: float, seed: int):
    L = kings.side_for_atoms(atoms, filling)
    pos, edges = kings.lattice_arrays(L, filling, seed)
    return L, pos, edges


def score(x, edges, bound, exact):
    """Plain-Python verdict on one 0/1 answer, independent of the solver."""
    import numpy as np

    xa = np.asarray(x, dtype=np.int8)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    viol = int(np.sum(xa[e[:, 0]] & xa[e[:, 1]])) if len(e) else 0
    fixed = kings.repair(xa.tolist(), e.tolist()) if viol else xa.tolist()
    size = int(sum(fixed))
    return {"set_size_raw": int(xa.sum()), "violations": viol, "independent": viol == 0, "set_size": size,
            "gap_to_bound": (bound - size) / bound if bound else None,
            "gap_to_exact": None if exact is None else (exact - size) / exact if exact else 0.0,
            "_fixed": fixed}


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="directory for results.jsonl")
    p.add_argument("--seeds", type=int, nargs="+", default=[1], help="solver seeds (default %(default)s)")
    p.add_argument("--time-limit", dest="budgets", type=float, nargs="+", default=[1.0],
                   help="seconds per job, one or more budgets (default 1)")
    p.add_argument("--atoms", type=int, nargs="+", default=[289, 1000, 10000, 100000, 1000000])
    p.add_argument("--filling", type=float, default=kings.FILLING)
    p.add_argument("--instance-seed", type=int, default=1)
    p.add_argument("--arms", nargs="+", default=["rows", "penalty"], choices=ARMS)
    p.add_argument("--penalty", type=int, default=2, help="P in the penalty form (any P > 1 keeps the optimum)")
    p.add_argument("--exact-max", type=int, default=2000, help="exact HiGHS reference up to this many atoms")
    p.add_argument("--no-license-check", action="store_true")
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()

    if not a.no_license_check and client is not None:
        limits = next((s.get("model_limits") for s in client.license().get("solvers", [])
                       if isinstance(s, dict) and s.get("id") == "optimizer.qubo"), None)
        out.add(kind="license", model_limits=limits)
        print(f"model_limits from your license: {limits}")

    for atoms in a.atoms:
        L, pos, edges = instance(atoms, a.filling, a.instance_seed)
        n, n_edges = len(pos), len(edges)
        bound = kings.block_bound(pos.tolist())
        exact = kings.exact_mis(n, edges) if n <= a.exact_max else None
        inst = dict(atoms_target=atoms, L=L, filling=a.filling, instance_seed=a.instance_seed, n=n, n_edges=n_edges)
        out.add(kind="instance", **inst, block_bound=bound, exact_mis=exact,
                service_memory_gb_est=examplekit.service_memory_gb(n), local_peak_rss_gb=round(examplekit.peak_rss_gb(), 3),
                positions=pos.tolist() if n <= KEEP_SET_UP_TO else None,
                edges=edges.tolist() if n <= KEEP_SET_UP_TO else None)
        print(f"atoms~{atoms}: L={L} n={n} edges={n_edges} bound={bound} exact={exact}")

        models = {}
        for arm in a.arms:
            if arm == "rows":
                models[arm] = kings.rows_model(n, edges.tolist())
            elif arm == "penalty":
                models[arm] = kings.penalty_model(n, edges, a.penalty)
        for T in a.budgets:
            for arm in a.arms:
                for seed in a.seeds:
                    common = dict(kind="solve", arm=arm, **inst, budget_s=T, seed=seed,
                                  penalty=a.penalty if arm != "rows" else None)
                    if arm == "sa-fill":
                        import rival
                        x, extra = rival.anneal(n, edges, T, seed, a.penalty)
                        sc = score(x, edges, bound, exact)
                        fixed = sc.pop("_fixed")
                        out.add(**common, **sc, **extra, timed=False, rival="dwave-samplers SA [OD-2]",
                                chosen=[i for i, v in enumerate(fixed) if v] if n <= KEEP_SET_UP_TO else None)
                        print(f"  {arm:8s} T={T:g} seed={seed}: size={sc['set_size']} viol={sc['violations']}")
                        continue
                    m = models[arm]
                    try:
                        r = fourcolors.qubo.solve(m, time_limit_s=T, seed=seed, client=client)
                    except fourcolors.errors.FourColorsError as exc:      # a refusal is recorded, not hidden
                        out.add(**common, error_code=getattr(exc, "code", None), error=str(exc))
                        print(f"  {arm:8s} T={T:g} seed={seed}: refused: {exc}")
                        continue
                    x = r.solution
                    sc = {} if x is None else score(x, edges, bound, exact)
                    fixed = sc.pop("_fixed", None)
                    out.add(**common, **sc, model_feasible=None if x is None or arm != "rows" else m.feasible(x),
                            service_feasible=(r.constraints or {}).get("feasible") if arm == "rows" else None,
                            chosen=[i for i, v in enumerate(fixed) if v] if fixed and n <= KEEP_SET_UP_TO else None,
                            **examplekit.result_fields(r))
                    print(f"  {arm:8s} T={T:g} seed={seed}: size={sc.get('set_size')} viol={sc.get('violations')} "
                          f"reason={examplekit.reason_code(r)}")
        del models


if __name__ == "__main__":
    main()
