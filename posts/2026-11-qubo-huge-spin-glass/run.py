"""Generate a huge 3D spin glass, upload it once, solve it by id at several budgets.

    python run.py --out results/L100 --L 100 --seed 1 --budgets 30 60 300
    python run.py --out results/L182 --L 182 --scratch /big/disk --budgets 30 60 300 --solver-seeds 0 1 2
    python run.py --out results/x --L 100 --plan-only         # sizes and estimates, nothing built or sent

Each solve writes one JSON line to <out>/results.jsonl: budget, seed, job id, status, the service's
objective, our own re-evaluation of the returned answer, energy per spin, gap to -1.7863 per spin,
wall time from submit to answer, and the job's billed seconds. Generation and upload get rows too.
The key is read in-process from FOURCOLORS_KEY_FILE (or FOURCOLORS_API_KEY).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

import examplekit as bk
import fourcolors
import fourcolors.qubo
import spinglass as sg


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output folder (results.jsonl, solutions/)")
    p.add_argument("--L", type=int, default=100, help="lattice side; n = L^3 spins")
    p.add_argument("--seed", type=int, default=1, help="instance seed")
    p.add_argument("--budgets", type=float, nargs="+", default=[30, 60, 300], help="time_limit_s values")
    p.add_argument("--solver-seeds", type=int, nargs="+", default=[0], help="seed= for each solve")
    p.add_argument("--scratch", default=None, help="where the 4CQB file goes (default: <out>/scratch)")
    p.add_argument("--keep-file", action="store_true", help="keep the 4CQB file after the upload")
    p.add_argument("--input-id", default=None, help="skip generation and upload; solve this uploaded model")
    p.add_argument("--zstd-threads", type=int, default=0, help="zstd worker threads (still one frame)")
    p.add_argument("--plan-only", action="store_true", help="print the size report and stop")
    p.add_argument("--no-license-check", action="store_true", help="do not read model_limits first")
    p.add_argument("--save-solutions", action="store_true", help="save each answer as packed bits (n/8 bytes)")
    p.add_argument("--upload-timeout", type=float, default=None,
                   help="seconds to wait for the API to check an upload (client default: 600)")
    return p.parse_args(argv)


def main(argv=None, client=None):
    a = parse(argv)
    out = Path(a.out)
    res = bk.Results(out / "results.jsonl")
    rep = sg.size_report(a.L)
    sg.print_size_report(rep)
    if a.plan_only:
        return rep
    if client is None and a.upload_timeout is None:
        client = bk.client()
    elif client is None:                              # the same key, with a longer wait for the upload check
        path = os.environ.get("FOURCOLORS_KEY_FILE")
        key = Path(path).expanduser().read_text().strip() if path else None
        client = fourcolors.Client(api_key=key, upload_timeout=a.upload_timeout)
    common = dict(L=a.L, n=rep["n"], nnz=rep["nnz"], instance_seed=a.seed)

    if not a.no_license_check:
        # A null size limit means no fixed limit; the job's memory check decides when it starts.
        limits = next((s.get("model_limits") for s in client.license().get("solvers", [])
                       if isinstance(s, dict) and s.get("id") == "optimizer.qubo"), None)
        print(f"  model_limits from your license: {limits}")

    input_id = a.input_id
    if input_id is None:
        scratch = Path(a.scratch or out / "scratch")
        scratch.mkdir(parents=True, exist_ok=True)
        path = scratch / f"ea3d-L{a.L}-s{a.seed}.4cqb.zst"
        info = sg.write_instance(path, a.L, a.seed, threads=a.zstd_threads)
        res.add(event="generate", **common, file_bytes=info["bytes_on_disk"], raw_bytes=info["bytes_raw"],
                write_s=round(info["write_s"], 3), generator_peak_rss_gb=round(bk.peak_rss_gb(), 3),
                input_id_local=info["input_id"])
        print(f"  wrote {path} ({info['bytes_on_disk'] / 1e9:.2f} GB) in {info['write_s']:.0f} s")
        t0 = time.perf_counter()
        input_id = fourcolors.qubo.upload(str(path), client=client)       # streamed from disk
        up = time.perf_counter() - t0
        res.add(event="upload", **common, input_id=input_id, upload_wall_s=round(up, 3),
                matches_local_id=(info["input_id"] in (None, input_id)))
        print(f"  uploaded as {input_id} in {up:.0f} s")
        if not a.keep_file:
            os.remove(path)                                               # disk is the scarce thing here

    for T in a.budgets:
        for seed in a.solver_seeds:
            t0 = time.perf_counter()
            r = fourcolors.qubo.solve(input_id=input_id, time_limit_s=T, seed=seed, client=client)
            wall = time.perf_counter() - t0
            row = dict(event="solve", **common, input_id=input_id, budget_s=T, solver_seed=seed,
                       wall_s=round(wall, 3), status=r.status, reason=r.reason, objective_reported=r.objective,
                       engine_version=(getattr(r, "engine", None) or {}).get("version"), client_version=bk.client_version())
            job = getattr(r, "job", None)
            for k in ("id", "billed_s", "run_s", "wait_s"):
                row["job_" + k] = getattr(job, k, None) if job is not None else None
            row["first_answer_s"] = r.trajectory[0].get("t_search_s") if r.trajectory else None
            row["improvements"] = len(r.trajectory)
            if r.solution is not None:
                s = np.asarray(r.solution, dtype=np.int8)
                sc = sg.score(a.L, a.seed, s)                 # our own evaluation, never the service's number
                row.update(sc, agrees=(r.objective == sc["energy"]))
                if a.save_solutions:
                    (out / "solutions").mkdir(parents=True, exist_ok=True)
                    np.save(out / "solutions" / f"L{a.L}-T{T:g}-s{seed}.npy", np.packbits(s > 0))
            res.add(**row)
            gap = row.get("gap_pal1996")
            print(f"  T={T:g}s seed={seed}: {r.status}, e/spin={row.get('energy_per_spin')}, "
                  f"gap={gap if gap is None else f'{100 * gap:.3f}%'}, wall {wall:.1f}s")
    return res.rows()


if __name__ == "__main__":
    main()
