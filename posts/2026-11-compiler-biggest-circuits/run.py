"""The biggest circuits: QAOA on 150 qubits with 1,000 to 20,000+ two-qubit gates.

    python run.py --out results/big [--sizes 150] [--configs 0.05:1,0.1:2,0.2:2,0.2:4,0.4:4,0.5:4]
                  [--budgets 10,60,300,600] [--seeds 1]

Each config density:p is QAOA with p layers on a G(n, density) random graph (seed n), so the input has about
p * density * n (n - 1) / 2 two-qubit gates (one rzz per edge per layer; 2 CZ each after translation).

The 4colors arm compiles each circuit at each budget (smallest first) for FakeFez (its description and published
calibration, `fourcolors.quantum.targets.from_qiskit`; there are no named targets). Circuits beyond the
service's size limits are refused: each refusal is a row with its code, message and limit, never dropped.
After a refusal about the circuit itself (input_too_large, does_not_fit_target) the larger budgets of that
circuit are written as "skip" rows instead of being sent again. Any other refusal is recorded and does not stop
the larger budgets (under the anytime contract a short budget is not a reason to refuse; such a row is a finding).

The first row after `meta` is `limits`: the key's live limits as `GET /v1/license` returns them (only the
`limits` object; no account fields). Budgets above `--plan-max-budget` (read it from that row; OD-9) are
written as "skip" rows ("above the plan limit") and never sent.

Qiskit runs level 3 seeds until the largest budget is spent, on --coupler-sets: `full` (FakeFez as published,
the primary arm) by default; `working` (our error >= 0.999 cut) only as a labeled extra arm.
"""
import os
import resource
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit
import circuitkit as ck

POST = "compiler-biggest-circuits"
CIRCUIT_REFUSALS = {"input_too_large", "does_not_fit_target", "invalid_input"}


def configs(text):
    out = []
    for item in text.split(","):
        d, p = item.split(":")
        out.append((float(d), int(p)))
    return out


def build(n, density, p):
    edges = examplekit.gnp(n, density, seed=n)
    qc = ck.qaoa_maxcut(n, edges, p=p)
    return qc, {**ck.circuit_fields(qc, f"qaoa_gnp{density:g}_{n}_p{p}", "qaoa_gnp"), "edges": len(edges),
                "density": density, "layers": p}


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux: kilobytes


def license_limits(client):
    """The `limits` object of `GET /v1/license` (nothing else: no account or usage fields), or a note."""
    if client is None:
        return {"limits": None, "note": "no client (4colors arm skipped or offline test)"}
    try:
        lic = client.license()
    except Exception as exc:                       # recorded, not fatal: the probe row says what happened
        return {"limits": None, "note": f"license call failed: {type(exc).__name__}"}
    inner = lic.get("license") if isinstance(lic.get("license"), dict) else {}
    return {"limits": lic.get("limits") or inner.get("limits"), "note": None}


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10,60,300,600", seeds="1", sizes="150")
    p.add_argument("--configs", type=configs, default=configs("0.05:1,0.1:2,0.2:2,0.2:4,0.4:4,0.5:4"),
                   help="density:layers pairs")
    p.add_argument("--coupler-sets", default="full",
                   help="Qiskit coupler sets: full (published, primary; default), working (labeled extra)")
    p.add_argument("--plan-max-budget", type=float, default=None,
                   help="largest budget the standard plan allows (from the probe's limits row; OD-9); larger "
                        "budgets are recorded as skip rows and not sent")
    args = p.parse_args(argv)
    budgets = sorted(args.budgets)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    all_sets, failed = ck.heavy_hex_coupler_sets()
    coupler_sets = {k: all_sets[k] for k in args.coupler_sets.split(",")}
    client = None if args.skip_4colors else examplekit.client()
    out.add(kind="limits", post=POST, plan_max_budget=args.plan_max_budget, **license_limits(client))

    for n in args.sizes:
        for density, layers in args.configs:
            qc, circ = build(n, density, layers)
            print(f"{circ['circuit_id']}: {circ['qubits']} qubits, {circ['input_2q']} two-qubit gates in", flush=True)
            for seed in args.seeds:
                if not args.skip_4colors:
                    refused_for = None
                    for budget in budgets:
                        if args.plan_max_budget is not None and budget > args.plan_max_budget:
                            out.add(**ck.row(kind="skip", post=POST, **circ, arm="4colors", coupler_set=ck.HEAVY_HEX_DEVICE,
                                             seed=seed, budget_s=budget),
                                    note=f"not sent: above the plan limit {args.plan_max_budget:g} s (OD-9)")
                            continue
                        if refused_for:
                            out.add(**ck.row(kind="skip", post=POST, **circ, arm="4colors", coupler_set=ck.HEAVY_HEX_DEVICE,
                                             seed=seed, budget_s=budget),
                                    note=f"not sent: refused at a smaller budget for {refused_for}")
                            continue
                        _, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=ck.heavy_hex_target(),
                                                 coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed, budget_s=budget, failed=failed,
                                                 client=client, approximation_mode="exact")
                        out.add(**r)
                        print(f"  4colors seed {seed} {budget:g}s: {r['two_qubit_gates']} {r['refusal_code'] or ''}",
                              flush=True)
                        if r["refusal_code"] in CIRCUIT_REFUSALS:
                            refused_for = r["refusal_code"]
                qbudgets = [b for b in budgets if args.plan_max_budget is None or b <= args.plan_max_budget]
                if not args.skip_qiskit and qbudgets:              # the same budgets as the 4colors arm
                    for r in ck.qiskit_rows(qc, coupler_sets, qbudgets, seed, post=POST, circuit=circ, failed=failed,
                                            max_trials=args.max_qiskit_trials):
                        out.add(**r, peak_rss_mb=peak_rss_mb())
                    print(f"  qiskit seed {seed}: done, peak RSS {peak_rss_mb():.0f} MB", flush=True)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
