"""Sparsify to the chip, or route the real problem: both sides of the trade-off, recorded.

    python run.py --out runs/native [--problems reg3_50 er_100 ...] [--time-limit 5] [--seeds 1,2,3]
                  [--budgets 30] [--layers 1,2,3] [--skip-qubo] [--skip-compile] [--skip-qiskit] [--skip-4colors]
                  [--calibration-blind]

Per problem graph (native.PROBLEMS):
  1. the native placement on FakeFez's working couplers (calibration-aware unless --calibration-blind), its kept
     edges (k of them), and the random and spectral sparsifiers at the same k ("variant" rows);
  2. QUBO: each of full / native / random / spectral solved as Max-Cut with the 4colors Optimizer, every answer
     scored on the ORIGINAL graph here ("solve" rows); for n <= 22 the exact maximum cut by enumeration;
  3. compiler: the full graph's QAOA circuit at each p, and the three sparsified circuits at p = 1, compiled by
     4colors for FakeFez with its published calibration; Qiskit O3 on the full graph's circuits, seeds until the
     same budget is spent, best kept ("result" rows, circuitkit). The native circuit is already on device qubits.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fourcolors.qubo

import examplekit
import circuitkit as ck
import native as nv

POST = "hardware-native-cost"


def pid(family: str, n: int) -> str:
    return f"{'reg3' if family == 'regular3' else 'er'}_{n}"


def variants(family, n, seed, n_qubits, couplers, aware):
    """{name: edges} for full / native / random / spectral, and the placement."""
    edges, w = nv.problem(family, n, seed)
    place = nv.native_placement(n, edges, w, n_qubits, couplers, seed=seed, aware=aware)
    keep = nv.native_edges(place, edges, couplers)
    k = len(keep)
    return edges, w, place, {"full": edges, "native": keep, "random": nv.random_edges(edges, k, seed),
                             "spectral": nv.spectral_edges(n, edges, w, k, seed)}


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="30", seeds="1,2,3")
    p.add_argument("--problems", nargs="+", default=None, help="problem ids, e.g. reg3_50 er_100 (default: all)")
    p.add_argument("--time-limit", type=float, default=5.0, help="QUBO search seconds (default 5)")
    p.add_argument("--layers", type=ck.ints, default=[1, 2, 3], help="QAOA layers for the full graph (default 1,2,3)")
    p.add_argument("--compile-seed", type=int, default=1)
    p.add_argument("--skip-qubo", action="store_true")
    p.add_argument("--skip-compile", action="store_true")
    p.add_argument("--calibration-blind", action="store_true", help="native placement ignores coupler errors")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    coupler_sets, failed = ck.heavy_hex_coupler_sets()
    full_t = coupler_sets["full"]
    n_qubits, couplers = nv.device_graph(full_t)
    client = None if args.skip_4colors else examplekit.client()
    aware = not args.calibration_blind

    for family, n, seed in nv.PROBLEMS:
        name = pid(family, n)
        if args.problems is not None and name not in args.problems:
            continue
        edges, w, place, var = variants(family, n, seed, n_qubits, couplers, aware)
        info = {"problem": name, "graph_family": family, "n": n, "n_edges": len(edges)}
        for v, es in var.items():
            out.add(kind="variant", post=POST, **info, variant=v, kept_edges=len(es),
                    kept_weight=sum(w[e] for e in es), total_weight=sum(w.values()),
                    placement=place if v == "native" else None, calibration_aware=aware if v == "native" else None,
                    edges=[list(e) for e in es])
        exact = examplekit.brute_force_maxcut(n, edges, {e: float(w[e]) for e in edges})[0] if n <= 22 else None

        if not args.skip_qubo and not args.skip_4colors:
            for v, es in var.items():
                m = nv.qubo_model(n, es, w)
                for s in args.seeds:
                    row = dict(kind="solve", post=POST, **info, variant=v, kept_edges=len(es), seed=s,
                               time_limit_s=args.time_limit, exact_full_cut=exact)
                    try:
                        r = fourcolors.qubo.solve(m, time_limit_s=args.time_limit, seed=s, client=client)
                        row.update(examplekit.result_fields(r))
                        if r.solution is not None:
                            row.update(cut_on_variant=nv.cut(es, w, r.solution),
                                       cut_on_full=nv.cut(edges, w, r.solution),
                                       energy_python=m.energy(r.solution))
                    except fourcolors.errors.FourColorsError as exc:
                        row.update(refusal_code=getattr(exc, "code", None), refusal_message=str(exc)[:400])
                    out.add(**row)
                    print(f"{name} {v} seed {s}: cut on full graph {row.get('cut_on_full')}", flush=True)

        if args.skip_compile:
            continue
        jobs = [("full", lay) for lay in args.layers] + [(v, 1) for v in ("native", "random", "spectral")]
        for v, lay in jobs:
            es = var[v]
            if v == "native":
                qc = nv.qaoa_circuit(n_qubits, es, w, p=lay, qubit_of=place, nodes=range(n))
                assert nv.on_couplers(qc, couplers)
            else:
                qc = nv.qaoa_circuit(n, es, w, p=lay)
            circ = {**ck.circuit_fields(qc, f"{name}_{v}_p{lay}", "qaoa_maxcut"), **info, "variant": v, "layers": lay,
                    "kept_edges": len(es)}
            if not args.skip_4colors:
                res, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                           coupler_set=ck.HEAVY_HEX_DEVICE, seed=args.compile_seed,
                                           budget_s=args.budgets[0], failed=failed, client=client,
                                           approximation_mode="exact")
                if res is not None and res.circuit is not None:
                    r["estimated_esp"] = ck.estimated_esp(res.to_qiskit(), full_t)
                out.add(**r)
                print(f"{circ['circuit_id']}: 4colors {r['two_qubit_gates']} two-qubit gates", flush=True)
            if not args.skip_qiskit and v == "full":
                for r in ck.qiskit_rows(qc, {"full": full_t}, args.budgets, args.compile_seed, post=POST,
                                        circuit=circ, failed=failed, max_trials=args.max_qiskit_trials,
                                        esp_target=full_t):
                    out.add(**r)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
