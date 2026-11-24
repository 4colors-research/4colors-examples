"""Fold RNA sequences with the stem QUBO of Fox et al. (2022) and referee each answer against the exact optimum.

    python run.py --out runs/example --example --time-limit 1 --seeds 1
    python run.py --out runs/rand --lengths 30 45 60 --instances 1 2 3 --time-limit 1 --seeds 1 2 3
    python run.py --out runs/mine --fasta my.fasta --time-limit 1          # your own sequences

For each sequence: enumerate the candidate stems, build the QUBO with one exact "at most one stem per base" row
per shared base (--rows pairs: one "not both" row per overlapping pair), compute the exact optimum by branch and
bound where it finishes within --exact-max-nodes (labeled "exact"; otherwise the row says "gave_up" and no gap is
claimed), then solve through the public API once per seed. Every answer is re-scored here from its 0/1 values with
Eq 13 and checked against the rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors.qubo

import examplekit
import rna


def instances(a):
    out = []
    if a.example:
        out.append(("fox2022-example", rna.FOX_EXAMPLE))
    for L in a.lengths or []:
        for s in a.instances:
            out.append((f"random-L{L}-s{s}", rna.random_sequence(L, s)))
    for seq in a.sequence or []:
        out.append((f"given-{len(out) + 1}", rna.clean(seq)))
    if a.fasta:
        out += rna.read_fasta(a.fasta)
    if not out:
        raise SystemExit("no sequence: use --example, --lengths, --sequence or --fasta")
    return out


def main(argv=None):
    p = examplekit.base_parser(__doc__, time_limit=1.0, seeds=(1,))
    p.add_argument("--example", action="store_true", help="the 18-nt worked example of Fox et al. 2022")
    p.add_argument("--lengths", type=int, nargs="*", help="random sequences of these lengths")
    p.add_argument("--instances", type=int, nargs="+", default=[1], help="sequence seeds for --lengths")
    p.add_argument("--sequence", nargs="*", help="sequences given on the command line")
    p.add_argument("--fasta", help="a FASTA file of your own sequences")
    p.add_argument("--rows", choices=["base", "pairs"], default="base",
                   help="no-overlap rule: one row per shared base (default) or one row per overlapping pair")
    p.add_argument("--min-stem", type=int, default=3)
    p.add_argument("--min-loop", type=int, default=2)
    p.add_argument("--exact-max-nodes", type=int, default=2_000_000,
                   help="branch-and-bound node budget for the exact reference (0: skip it)")
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()

    for name, seq in instances(a):
        st = rna.stems(seq, a.min_stem, a.min_loop)
        common = {"instance": name, "length": len(seq), "n_stems": len(st), "rows_kind": a.rows,
                  "time_limit_s": a.time_limit, "params": {"c_B": rna.C_B, "c_L": rna.C_L, "c_P": str(rna.C_P),
                                                           "min_stem": a.min_stem, "min_loop": a.min_loop}}
        if not st:
            out.add(kind="instance", sequence=seq, note="no candidate stem", **common)
            print(f"{name}: no candidate stem")
            continue
        ref = rna.exact_reference(st, a.exact_max_nodes) if a.exact_max_nodes else None
        m = rna.build_model(seq, st, rows=a.rows)
        inst = dict(kind="instance", sequence=seq, mu=rna.longest(st), n_vars=m.n, n_quad=m.nnz, n_rows=m.m,
                    **common)
        if ref is not None:
            sel = rna.chosen(st, ref["x"])
            inst.update(reference_status=ref["status"], reference_energy=float(ref["energy"]),
                        reference_energy_exact=str(ref["energy"]), reference_nodes=ref["nodes"],
                        reference_dot_bracket=rna.dot_bracket(len(seq), sel),
                        reference_pairs=rna.base_pairs(st, ref["x"]),
                        reference_pseudoknot=rna.has_pseudoknot(sel))
        out.add(**inst)
        for seed in a.seeds:
            r = fourcolors.qubo.solve(m, time_limit_s=a.time_limit, seed=seed, client=client)
            x = r.solution
            row = dict(kind="solve", seed=seed, n_vars=m.n, n_quad=m.nnz, n_rows=m.m, **common)
            if x is not None:
                sel = rna.chosen(st, x)
                h = rna.energy(st, x)
                row.update(stems=[[s.i, s.j, s.k] for s in sel], pairs=rna.base_pairs(st, x),
                           dot_bracket=rna.dot_bracket(len(seq), sel) if rna.compatible(st, x) else None,
                           pseudoknot=rna.has_pseudoknot(sel), feasible=rna.compatible(st, x),
                           model_feasible=m.feasible(x), energy=float(h), energy_exact=str(h),
                           energy_python=float(h * rna.SCALE), energy_model=m.energy(x))
                if ref is not None and ref["status"] == "exact":
                    row.update(reference_energy=float(ref["energy"]), gap_abs=float(h - ref["energy"]),
                               matches_exact=h == ref["energy"])
            else:
                row.update(feasible=False)
            out.add(**row, **examplekit.result_fields(r))
            ref_txt = "no reference" if ref is None else f"{ref['status']} {float(ref['energy']):g}"
            print(f"{name} seed={seed} stems={len(st)} rows={m.m} H={row.get('energy')} ({ref_txt}) "
                  f"{row.get('dot_bracket')}")


if __name__ == "__main__":
    main()
