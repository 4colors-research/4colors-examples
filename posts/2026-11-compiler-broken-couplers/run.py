"""Broken couplers: what happens if the compiler is not told.

    python run.py --out results/broken [--budgets 30] [--seeds 1,2,3,4,5] [--circuits chip_native,qaoa_100,bv_100]

FakeFez (Qiskit's snapshot of a 156-qubit Heron device) lists 176 couplers, and its calibration marks 7 of them
with a two-qubit error of 1.0: they do not work. A compiler only avoids them if it is told. The arms:

  qiskit / working          transpile(level 3) on FakeFez without the failed couplers
  qiskit / full             transpile(level 3) on FakeFez as shipped (the errors are in the Target)
  4colors / full            compile(qc, target=<FakeFez's description dict>): the device map, failed couplers
                            included, and no calibration sent: "not told"
  4colors / full+calibration  compile(qc, target=targets.from_qiskit(FakeFez())): the same map with FakeFez's
                            published calibration (the client reads it from the backend and sends it): "told"

There are no named targets: a target is a description, and the calibration is the caller's own. A Qiskit backend
given as the target sends its calibration too, so "not told" is the description alone.

Every row counts two-qubit gates on failed couplers and an estimated success probability (ESP) from the FakeFez
error rates; any gate on a failed coupler makes the estimate 0.
"""
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck

POST = "compiler-broken-couplers"


def chip_native(couplers):
    """A circuit whose interaction graph is the whole FakeFez map: H everywhere, then one rzz(0.3) per
    coupler, failed ones included. On the full map it needs no routing at all, if the failed couplers are used."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(156, name="chip_native")
    qc.h(range(156))
    for a, b in couplers:
        qc.rzz(0.3, a, b)
    return qc


def circuits(names, full_couplers):
    out = {}
    for name in names:
        if name == "chip_native":
            qc = chip_native(full_couplers)
        elif name.startswith("qaoa_"):
            n = int(name.split("_")[1])
            qc = ck.qaoa_maxcut(n, examplekit.random_regular(n, 3, seed=n), p=1)
        elif name.startswith("bv_"):
            qc = ck.bernstein_vazirani("1" * int(name.split("_")[1]))
        else:
            raise SystemExit(f"unknown circuit {name}")
        out[name] = (qc, ck.circuit_fields(qc, name, name.split("_")[0]))
    return out


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="30")
    p.add_argument("--circuits", default="chip_native,qaoa_100,bv_100",
                   help="comma list: chip_native, qaoa_<n>, bv_<n>")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))

    coupler_sets, failed = ck.heavy_hex_coupler_sets()
    full = coupler_sets["full"]
    told = ck.heavy_hex_target()                      # description + FakeFez's published calibration
    out.add(kind="device", post=POST, name="FakeFez", n_qubits=full.num_qubits,
            couplers=ck.device_couplers(full), failed=sorted(failed),
            target_source="fourcolors.quantum.targets.from_qiskit(FakeFez())",
            calibration_couplers_at_error_1=sum(1 for c in told.calibration["couplers"]
                                                if (c.get("error") or 0) >= ck.FAILED_ERROR),
            target_notes=list(told.notes))
    client = None if args.skip_4colors else examplekit.client()
    fourcolors_arms = [("full", dict(told.description), {}),
                       ("full+calibration", told, {})]

    for name, (qc, circ) in circuits(args.circuits.split(","), ck.device_couplers(full)).items():
        for seed in args.seeds:
            if not args.skip_qiskit:
                for r in ck.qiskit_rows(qc, coupler_sets, args.budgets, seed, post=POST, circuit=circ,
                                        failed=failed, max_trials=args.max_qiskit_trials, with_counts=True,
                                        esp_target=full):
                    out.add(**r)
            if args.skip_4colors:
                continue
            for budget in args.budgets:
                for cset, target, opts in fourcolors_arms:
                    res, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=target, coupler_set=cset,
                                               seed=seed, budget_s=budget, failed=failed, with_counts=True,
                                               client=client, approximation_mode="exact", **opts)
                    if res is not None and res.circuit is not None:
                        r["estimated_esp"] = ck.estimated_esp(res.to_qiskit(), full)
                    out.add(**r)
                    print(f"{name} seed {seed} {cset}: {r['two_qubit_gates']} two-qubit gates, "
                          f"{r['gates_on_failed_couplers']} on failed couplers {r['refusal_code'] or ''}", flush=True)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
