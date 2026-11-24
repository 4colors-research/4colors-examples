"""Run the stated noise model on every compiled circuit of one or more run.py records.

    python simulate.py results/obs [more record dirs] --out results/sim [--shots 100000] [--sim-seeds 1]

For each circuit: one "ideal" row (exact statevector of the logical circuit) and, for each compiled circuit of each
arm, one "sim" row with the noisy observable, its error against the ideal (observables.error), the shot-noise
standard error, the compiled circuit's two-qubit gates and its estimated ESP (a model, carried over from run.py).
Simulation only: nothing here runs on hardware or calls the service.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit
import domain_circuits as circuits
import circuitkit as ck
import noise
import observables

POST = "compiler-noisy-observables"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="run.py output dirs (results.jsonl and circuits/)")
    p.add_argument("--out", required=True)
    p.add_argument("--shots", type=int, default=100_000)
    p.add_argument("--sv-shots", type=int, default=20_000,
                   help="shot cap when a circuit has more active qubits than the density-matrix limit")
    p.add_argument("--sim-seeds", type=ck.ints, default=[1])
    p.add_argument("--no-idle", action="store_true", help="leave idle-time relaxation out (a labeled extra)")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "sim.jsonl"))
    out.add(kind="meta", post=POST, argv=sys.argv[1:], args=vars(args), shots=args.shots,
            noise_model="qiskit_aer NoiseModel.from_backend(FakeFez()): gate depolarizing + thermal relaxation + "
                        "readout, ALAP idle delays" + ("" if not args.no_idle else " (idle left out)"))
    fez = ck.fez()
    sim = noise.simulator(fez)
    ideal_cache = {}
    for d in args.inputs:
        rows = examplekit.load_rows([os.path.join(d, "results.jsonl")])
        for r in ck.results(rows):
            if not r.get("circuit_file") or (r["arm"] == "qiskit" and r["coupler_set"] != "full"):
                continue
            cid = r["circuit_id"]
            if cid not in ideal_cache:
                qc, spec = circuits.build(cid)
                ideal = observables.measure(spec, circuits.ideal_probabilities(qc))
                ideal_cache[cid] = (spec, ideal)
                out.add(kind="ideal", post=POST, circuit_id=cid, family=spec["family"], spec=spec, **ideal)
            spec, ideal = ideal_cache[cid]
            compiled = circuits.load_qpy(os.path.join(d, "circuits", r["circuit_file"]))
            for s in args.sim_seeds:
                dist, meta = noise.noisy_distribution(compiled, sim, fez.target, shots=args.shots, seed=s,
                                                      idle=not args.no_idle, sv_shots=args.sv_shots)
                noisy = observables.measure(spec, dist)
                row = out.add(kind="sim", post=POST, circuit_id=cid, family=spec["family"], spec=spec, arm=r["arm"],
                        seed=r["seed"], budget_s=r["budget_s"], two_qubit_gates=r["two_qubit_gates"],
                        two_qubit_depth=r["two_qubit_depth"], estimated_esp=r.get("estimated_esp"),
                        job_id=r.get("job_id"), engine_version=r.get("engine_version"), **meta, **noisy,
                        error=observables.error(spec, noisy, ideal),
                        error_se=observables.standard_error(spec, noisy, meta["shots"]))
                print(f"{cid} {r['arm']} seed {r['seed']}: error {row['error']:.4f}", flush=True)
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
