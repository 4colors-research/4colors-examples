"""Run the two-call pre-flight check on every instance (or a chosen few) and record both signals.

    python check.py --out runs/triage [--ids reg3_n16 sk_n10] [--time-limit 10] [--budget 10] [--seed 1]
                    [--layers 1] [--skip-qiskit] [--skip-4colors] [--max-qiskit-trials K]

Per instance:
  1. fourcolors.qubo.solve(model, time_limit_s=T): the reference value (cut recomputed here from the 0/1 answer)
     and the trajectory signal (triage.classical_signal: service search seconds, not wall time). For n <= 22 the
     exact maximum cut by enumeration is recorded beside it, so small rows say whether the answer is optimal.
  2. fourcolors.quantum.compiler.compile(qaoa, target=targets.from_qiskit(FakeFez()), time_budget_s=B): two-qubit
     gates, depth and the modeled ESP on the published calibration.
  Optional comparison arm: Qiskit O3 on FakeFez as shipped, seeds until B seconds are spent, best kept, with its
  modeled ESP; `crosses_line` says when the 4colors compile is on the shallow side of the line and Qiskit's is not.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fourcolors.qubo

import examplekit
import circuitkit as ck
import triage

POST = "worth-qpu-time"


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10", seeds="1")
    p.add_argument("--ids", nargs="+", default=None, help="instance ids (default: all 30)")
    p.add_argument("--time-limit", type=float, default=10.0, help="QUBO search seconds (default 10)")
    p.add_argument("--layers", type=int, default=1, help="QAOA layers p (default 1)")
    p.add_argument("--easy-s", type=float, default=triage.EASY_S)
    p.add_argument("--esp-line", type=float, default=triage.ESP_LINE)
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    insts = [i for i in triage.instances() if args.ids is None or i["id"] in args.ids]
    coupler_sets, failed = ck.heavy_hex_coupler_sets()
    full = coupler_sets["full"]
    client = None if args.skip_4colors else examplekit.client()
    seed = args.seeds[0]
    budget = args.budgets[0]

    for inst in insts:
        n, edges, w = inst["n"], inst["edges"], inst["weights"]
        info = {"instance": inst["id"], "family": inst["family"], "n": n, "n_edges": len(edges)}
        exact = examplekit.brute_force_maxcut(n, edges, {e: float(w[e]) for e in edges})[0] if n <= 22 else None
        qc = triage.qaoa_circuit(n, edges, w, p=args.layers)
        circ = ck.circuit_fields(qc, f"{inst['id']}_p{args.layers}", inst["family"])
        sig, esp4, espq = {}, None, None
        if not args.skip_4colors:
            m = triage.qubo_model(n, edges, w)
            try:
                r = fourcolors.qubo.solve(m, time_limit_s=args.time_limit, seed=seed, client=client)
                f = examplekit.result_fields(r)
                c = triage.cut(edges, w, r.solution) if r.solution is not None else None
                sig = triage.classical_signal(f["trajectory"], f["solver_objective"], budget_s=args.time_limit)
                out.add(kind="solve", post=POST, **info, seed=seed, time_limit_s=args.time_limit, cut=c,
                        energy_python=None if r.solution is None else m.energy(r.solution), exact_cut=exact,
                        is_optimal=None if exact is None or c is None else c == exact, **f, **sig)
            except fourcolors.errors.FourColorsError as exc:
                out.add(kind="solve", post=POST, **info, seed=seed, refusal_code=getattr(exc, "code", None),
                        refusal_message=str(exc)[:400])
            res, row = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                         coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed, budget_s=budget,
                                         failed=failed, client=client, approximation_mode="exact")
            if res is not None and res.circuit is not None:
                esp4 = ck.estimated_esp(res.to_qiskit(), full)
            out.add(**{**row, **info, "estimated_esp": esp4})
        if not args.skip_qiskit:
            for r in ck.qiskit_rows(qc, {"full": full}, [budget], seed, post=POST, circuit=circ, failed=failed,
                                    max_trials=args.max_qiskit_trials, esp_target=full):
                if r["kind"] == "result":
                    espq = r.get("estimated_esp")
                out.add(**{**r, **info})
        out.add(kind="verdict", post=POST, **info, t_final_s=sig.get("t_final_s"), esp_4colors=esp4,
                esp_qiskit=espq, easy_s=args.easy_s, esp_line=args.esp_line,
                verdict_4colors=triage.verdict(sig.get("t_final_s"), esp4, easy_s=args.easy_s,
                                               esp_line=args.esp_line),
                crosses_line=triage.crosses_line(esp4, espq, args.esp_line))
        print(f"{inst['id']}: t_final {sig.get('t_final_s')} s (service), ESP 4colors {esp4}, Qiskit {espq}",
              flush=True)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
