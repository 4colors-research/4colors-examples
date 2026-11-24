"""Compile the side-by-side circuits with every arm and store every output in full.

    python snapshot.py --backend ibm:ibm_kingston --out snap        # once: the frozen published calibration
    python run.py --snapshot snap --out out --arms rivals           # Qiskit O2, O3, O3 best-of-seeds, TKET
    python check.py out/<device>                                    # equivalence and measured distances
    python run.py --snapshot snap --out out --arms ours             # 4colors, with the rivals' measured distance
    python inspect_cell.py out/<device>/<circuit> --out inspect     # counts, blocks, moves, figures

Layout written (one device dir per snapshot, one cell dir per circuit):

    out/<device>/target.json  cal.json  snapshot.json             what every arm was given
    out/<device>/rivals.jsonl ours.jsonl                          one row per compiled output
    out/<device>/<circuit>/in.qasm                                the input, byte for byte what every arm got
    out/<device>/<circuit>/<arm>-s<seed>.qasm (+ .layouts.json)   every output, with its initial layout

Arms (names as in the rows and file stems):
    qiskit_o2, qiskit_o3        transpile(optimization_level=2 or 3, seed_transpiler=s), s in --qiskit-seeds, on the
                                snapshot's Qiskit Target as published; each output's own wall time
    qiskit_o3best_T<T>          O3 with new seeds until T seconds are spent, in --streams streams, best kept
    tket_tk2                    pytket's IBMQBackend.pass_from_info(..., optimisation_level=2) on the same couplers
                                (skipped with a note if pytket is not installed; TKET's pass reads no error rates)
    ours_inst                   4colors with no time budget, seeds --ours-seeds
    ours_T<T>                   4colors with time_budget_s=T
Ours gets the same approximation as the rivals (rule D2): exact where every rival output of the cell measured
distance 0, else `max_approximation_error` = the smallest measured rival distance; a cell where any rival distance
is unmeasured waits (a "waiting_d2" row, nothing compiled). Refusals are rows, never dropped. When the rivals were
compiled in another record, `--rivals-from <that out/device dir> --checks-from <its checks.jsonl>` reads them there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit  # noqa: E402
import circuits  # noqa: E402
import circuitkit as ck  # noqa: E402
import snapshot  # noqa: E402

POST = "compiler-side-by-side"
RIVAL_ARMS = ("qiskit_o2", "qiskit_o3", "qiskit_o3best", "tket_tk2")
OUR_ARMS = ("ours_inst", "ours_T")
EXACT_TOL = 1e-12


def jl_add(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def jl(path):
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def qasm_of(qc) -> str:
    from qiskit import qasm3

    return qasm3.dumps(qc)


def initial_layout(qc_out, n_logical: int):
    lay = getattr(qc_out, "layout", None)
    if lay is None:
        return None
    try:
        return [int(p) for p in lay.initial_index_layout(filter_ancillas=True)[:n_logical]]
    except Exception:
        return None


def write_output(cell, stem, text, layout):
    with open(os.path.join(cell, stem + ".qasm"), "w") as f:
        f.write(text)
    with open(os.path.join(cell, stem + ".layouts.json"), "w") as f:
        json.dump({"initial": layout}, f)
    return sha_text(text)


def d2_options(distances):
    """The approximation ours may use, from the measured distances of the rival outputs in the cell: None (wait)
    if there is none or any is unmeasured; exact if every one is 0; else the smallest measured distance, so ours
    never gets a looser bound than any rival used."""
    if not distances or any(d is None for d in distances):
        return None
    d = min(float(x) for x in distances)
    if d <= EXACT_TOL:
        return {"approximation_mode": "exact"}
    return {"max_approximation_error": d}


def usable_couplers(desc: dict, cal: dict):
    """The description's couplers minus those the published calibration marks unavailable (never our cut)."""
    off = {tuple(sorted(c["qubits"])) for c in (cal or {}).get("couplers", []) if c.get("available") is False}
    return [tuple(c) for c in desc["couplers"] if tuple(sorted(c)) not in off]


def tket_compile(qc, n_device: int, couplers, level: int = 2):
    """pytket's default IBM pass offline on the given couplers; returns (physical Qiskit circuit, layout, wall)."""
    from pytket.architecture import Architecture
    from pytket.backends.backendinfo import BackendInfo
    from pytket.circuit import Node, OpType
    from pytket.extensions.qiskit import IBMQBackend, qiskit_to_tk, tk_to_qiskit
    from pytket.predicates import CompilationUnit
    from qiskit import QuantumCircuit

    arch = Architecture([(Node(a), Node(b)) for a, b in couplers] + [(Node(b), Node(a)) for a, b in couplers])
    info = BackendInfo("IBMQBackend", "snapshot", "offline", arch,
                       {OpType.CZ, OpType.Rz, OpType.SX, OpType.X, OpType.Measure, OpType.Barrier})
    t0 = time.perf_counter()
    cu = CompilationUnit(qiskit_to_tk(qc))
    IBMQBackend.pass_from_info(info, level).apply(cu)
    wall = time.perf_counter() - t0
    small = tk_to_qiskit(cu.circuit)
    full = QuantumCircuit(n_device, small.num_clbits)
    full.compose(small, qubits=[small.find_bit(q).registers[0][1] for q in small.qubits],
                 clbits=list(range(small.num_clbits)), inplace=True)
    init = {k.index[0]: v.index[0] for k, v in cu.initial_map.items() if k.reg_name == "q"}
    return full, [init[i] for i in range(qc.num_qubits)], wall


def rival_rows(qc, cid, cell, qt, desc, cal, args, rows_path):
    from qiskit import transpile

    n = qc.num_qubits
    base = {"post": POST, "circuit": cid, **ck.circuit_fields(qc, cid, circuits.CIRCUITS[cid]["family"])}
    for level in (2, 3):
        if f"qiskit_o{level}" not in args.arms_list:
            continue
        for s in args.qiskit_seeds:
            t0 = time.perf_counter()
            out = transpile(qc, target=qt, optimization_level=level, seed_transpiler=s)
            wall = time.perf_counter() - t0
            stem = f"qiskit_o{level}-s{s}"
            sha = write_output(cell, stem, qasm_of(out), initial_layout(out, n))
            jl_add(rows_path, {**base, "arm": f"qiskit_o{level}", "seed": s, "stem": stem, "wall_s": wall,
                               "qasm_sha256": sha, "engine_version": ck.qiskit_version(), **ck.circuit_metrics(out)})
    if "qiskit_o3best" in args.arms_list:
        for T in args.budgets:
            trials = []
            for k in range(1, args.streams + 1):           # one stream per core ours uses (equal cores)
                trials += ck.qiskit_trials(qc, qt, T, seed=k, max_trials=args.max_qiskit_trials, keep_circuits=True)
            pick = ck.best_within(trials, T)
            stem = f"qiskit_o3best_T{T:g}-s1"
            sha = write_output(cell, stem, qasm_of(pick["circuit"]), initial_layout(pick["circuit"], n))
            jl_add(rows_path, {**base, "arm": f"qiskit_o3best_T{T:g}", "seed": 1, "stem": stem, "budget_s": T,
                               "streams": args.streams, "qiskit_seed": pick["qiskit_seed"],
                               "trials_in_budget": pick["trials_in_budget"], "over_budget": pick["over_budget"],
                               "wall_s": pick["cumulative_s"], "qasm_sha256": sha,
                               "engine_version": ck.qiskit_version(), **ck.circuit_metrics(pick["circuit"])})
    if "tket_tk2" in args.arms_list:
        try:
            import pytket  # noqa: F401
            import pytket.extensions.qiskit  # noqa: F401
        except ImportError:
            jl_add(rows_path, {**base, "arm": "tket_tk2", "status": "skipped", "note": "pytket not installed"})
            return
        out, lay, wall = tket_compile(qc, desc["n_qubits"], usable_couplers(desc, cal))
        stem = "tket_tk2-s0"
        sha = write_output(cell, stem, qasm_of(out), lay)
        jl_add(rows_path, {**base, "arm": "tket_tk2", "seed": 0, "stem": stem, "wall_s": wall, "qasm_sha256": sha,
                           "engine_version": f"pytket {pytket.__version__}", **ck.circuit_metrics(out)})


def our_rows(qc, cid, cell, dev_target, dev_dir, args, client):
    n = qc.num_qubits
    base = {"post": POST, "circuit": cid, **ck.circuit_fields(qc, cid, circuits.CIRCUITS[cid]["family"])}
    rdir = args.rivals_from or dev_dir
    cfiles = args.checks_from or [os.path.join(rdir, "checks.jsonl")]
    checks = {r["stem"]: r for f in cfiles for r in jl(f) if r["circuit"] == cid}
    rivals = [r for r in jl(os.path.join(rdir, "rivals.jsonl")) if r["circuit"] == cid and r.get("stem")]
    opts = d2_options([(checks.get(r["stem"]) or {}).get("distance") for r in rivals])
    budgets = ([None] if "ours_inst" in args.arms_list else []) + \
        (list(args.budgets) if "ours_T" in args.arms_list else [])
    for T in budgets:
        arm = "ours_inst" if T is None else f"ours_T{T:g}"
        for s in args.ours_seeds:
            if opts is None:
                jl_add(os.path.join(dev_dir, "ours.jsonl"), {**base, "arm": arm, "seed": s, "status": "waiting_d2",
                                                             "note": "a rival distance is unmeasured"})
                continue
            res, f = ck.compile_4colors(qc, target=dev_target, time_budget_s=T, seed=s, client=client,
                                        verify="background", **opts)
            row = {**base, "arm": arm, "seed": s, "budget_s": T, "options": opts, **f}
            if res is not None and res.circuit is not None:
                out = res.to_qiskit()
                text = res.circuit if (getattr(res, "format", "qasm3") == "qasm3" and
                                       str(res.circuit).lstrip().startswith("OPENQASM 3")) else qasm_of(out)
                row["stem"] = f"{arm}-s{s}"
                row["qasm_sha256"] = write_output(cell, row["stem"], text, initial_layout(out, n))
                row["timings"] = dict(getattr(res, "timings", None) or {})
            jl_add(os.path.join(dev_dir, "ours.jsonl"), row)
            print(f"{cid} {arm} s{s}: {row.get('two_qubit_gates')} two-qubit gates {row.get('refusal_code') or ''}",
                  flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", required=True, help="a dir written by snapshot.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--circuits", default=",".join(circuits.CIRCUITS))
    ap.add_argument("--arms", default="rivals", help="rivals, ours, or a comma list of arm names")
    ap.add_argument("--qiskit-seeds", type=ck.ints, default=ck.ints("1,2,3,4,5"))
    ap.add_argument("--ours-seeds", type=ck.ints, default=ck.ints("1,2,3"))
    ap.add_argument("--budgets", type=ck.floats, default=ck.floats("1,10"))
    ap.add_argument("--streams", type=int, default=1, help="Qiskit O3 seed streams per budget (equal cores)")
    ap.add_argument("--max-qiskit-trials", type=int, default=None)
    ap.add_argument("--rivals-from", help="ours: the device dir whose rivals.jsonl gives the D2 bound (default: --out's)")
    ap.add_argument("--checks-from", nargs="+", help="ours: checks.jsonl files with the rivals' measured distances")
    a = ap.parse_args(argv)
    a.arms_list = {"rivals": RIVAL_ARMS, "ours": OUR_ARMS}.get(a.arms, tuple(a.arms.split(",")))
    qt, dev_target, meta = snapshot.load(a.snapshot)
    desc, cal = dev_target.description, dev_target.calibration
    dev_dir = os.path.join(a.out, meta["device"])
    os.makedirs(dev_dir, exist_ok=True)
    for name, obj in (("target.json", desc), ("cal.json", cal), ("snapshot.json", meta)):
        with open(os.path.join(dev_dir, name), "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
    client = examplekit.client() if any(x.startswith("ours") for x in a.arms_list) else None
    for cid in a.circuits.split(","):
        qc = circuits.build(cid)
        cell = os.path.join(dev_dir, cid)
        os.makedirs(cell, exist_ok=True)
        text = qasm_of(qc)
        path = os.path.join(cell, "in.qasm")
        if os.path.exists(path) and open(path).read() != text:
            raise SystemExit(f"{path} differs from the builder's output: a different generator commit")
        with open(path, "w") as f:
            f.write(text)
        if any(x in RIVAL_ARMS for x in a.arms_list):
            rival_rows(qc, cid, cell, qt, desc, cal, a, os.path.join(dev_dir, "rivals.jsonl"))
        if any(x in OUR_ARMS for x in a.arms_list):
            our_rows(qc, cid, cell, dev_target, dev_dir, a, client)
    print(f"wrote {dev_dir}")


if __name__ == "__main__":
    main()
