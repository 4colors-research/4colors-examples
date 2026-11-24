"""Solve paint shop instances with the 4colors Optimizer and judge every answer here.

    python solve.py --out runs/bpsp --kind binary --sizes 12 16 20 100 1000 10000 --time-limit 5 --seeds 1 2 3
    python solve.py --out runs/multi --kind multicar --sizes 16 200 2000 --classes 4 --time-limit 5 --seeds 1 2 3

Binary: one QUBO per instance (random word, each car twice); the answer's color changes are recounted from the
0/1 solution, beside the textbook greedy heuristic's count and, for n <= --brute-force-max, the exact optimum by
enumeration. Multi-car: one exact row per car class (how many bodies must be black); every constrained answer is
checked here against the quotas, beside the sequence-order heuristic and, for small N, the exact optimum.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors.qubo

import examplekit
import paintshop as ps

POST = "paint-shop-problem"


def solve(m, a, seed, client):
    try:
        return fourcolors.qubo.solve(m, time_limit_s=a.time_limit, seed=seed, client=client), None
    except fourcolors.errors.FourColorsError as exc:       # a refusal is a row, never dropped
        return None, {"refusal_code": getattr(exc, "code", type(exc).__name__), "refusal_message": str(exc)[:400]}


def binary_rows(a, out, client):
    for n in a.sizes:
        for inst in a.instances:
            word = ps.random_word(n, seed=inst)
            ref = {"greedy": ps.color_changes(word, ps.greedy(word))}
            if n <= a.brute_force_max:
                ref["optimum"] = ps.brute_force(word, a.brute_force_max)[0]
            m = ps.model(word)
            common = dict(post=POST, kind_model="binary", n=n, instance=inst, n_vars=m.n, n_quad=m.nnz,
                          time_limit_s=a.time_limit)
            out.add(kind="reference", **common, **ref, greedy_method="greedy (textbook heuristic)",
                    optimum_method="enumeration (exact)" if "optimum" in ref else None)
            for seed in a.seeds:
                r, err = solve(m, a, seed, client)
                row = dict(kind="solve", seed=seed, **common, **(err or {}))
                if r is not None:
                    row.update(examplekit.result_fields(r))
                    if r.solution is not None:
                        ch = ps.color_changes(word, r.solution)
                        row.update(color_changes=ch, energy_python=m.energy(r.solution),
                                   greedy=ref["greedy"], vs_greedy=ch - ref["greedy"],
                                   optimum=ref.get("optimum"),
                                   gap_to_optimum=None if "optimum" not in ref else ch - ref["optimum"])
                        if n <= 20:                 # small answers kept whole, for the hero figure
                            row["solution"] = [int(v) for v in r.solution]
                out.add(**row)
                print(f"binary n={n} inst={inst} seed={seed}: {row.get('color_changes')} changes "
                      f"(greedy {ref['greedy']}, optimum {ref.get('optimum')}) {row.get('refusal_code') or ''}",
                      flush=True)


def multicar_rows(a, out, client):
    for n in a.sizes:
        for inst in a.instances:
            classes, quotas = ps.random_multicar(n, a.classes, seed=inst)
            heur = ps.multicar_sequence_heuristic(classes, quotas)
            ref = {"heuristic": ps.count_changes(heur)}
            if n <= a.multicar_brute_force_max:
                ref["optimum"] = ps.multicar_brute_force(classes, quotas, a.multicar_brute_force_max)[0]
            m = ps.multicar_model(classes, quotas)
            common = dict(post=POST, kind_model="multicar", n=n, instance=inst, n_classes=a.classes,
                          quotas=quotas, n_vars=m.n, n_rows=m.m, time_limit_s=a.time_limit)
            out.add(kind="reference", **common, **ref, heuristic_method="sequence order (labeled heuristic)",
                    optimum_method="enumeration (exact)" if "optimum" in ref else None)
            for seed in a.seeds:
                r, err = solve(m, a, seed, client)
                row = dict(kind="solve", seed=seed, **common, **(err or {}))
                if r is not None:
                    row.update(examplekit.result_fields(r))
                    x = r.solution
                    if x is not None:
                        ok = ps.multicar_feasible(classes, quotas, x)
                        ch = ps.count_changes(x)
                        row.update(color_changes=ch, quotas_met=ok, model_feasible=m.feasible(x),
                                   energy_python=m.energy(x), heuristic=ref["heuristic"],
                                   vs_heuristic=ch - ref["heuristic"], optimum=ref.get("optimum"),
                                   gap_to_optimum=None if "optimum" not in ref else ch - ref["optimum"])
                out.add(**row)
                print(f"multicar N={n} inst={inst} seed={seed}: {row.get('color_changes')} changes, quotas met "
                      f"{row.get('quotas_met')} (heuristic {ref['heuristic']}, optimum {ref.get('optimum')})",
                      flush=True)


def main(argv=None):
    p = examplekit.base_parser(__doc__)
    p.add_argument("--kind", choices=["binary", "multicar"], default="binary")
    p.add_argument("--sizes", type=int, nargs="+", default=[12, 16, 20, 100, 1000, 10000],
                   help="cars (binary) or bodies (multi-car)")
    p.add_argument("--instances", type=int, nargs="+", default=[1, 2, 3], help="instance seeds")
    p.add_argument("--classes", type=int, default=4, help="car classes (multi-car)")
    p.add_argument("--brute-force-max", type=int, default=20, help="largest binary n to enumerate")
    p.add_argument("--multicar-brute-force-max", type=int, default=16, help="largest multi-car N to enumerate")
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    out.add(kind="meta", post=POST, argv=sys.argv[1:] if argv is None else list(argv), args=vars(a),
            client_version=examplekit.client_version())
    client = examplekit.client()
    (binary_rows if a.kind == "binary" else multicar_rows)(a, out, client)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
