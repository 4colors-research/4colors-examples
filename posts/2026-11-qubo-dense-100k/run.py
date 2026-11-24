"""The 4colors arm: generate each dense instance, upload it once, solve it by id at every budget and seed, and
re-score every answer locally with dense.energy (never the service's number alone).

    python run.py --out runs/dense --family reg --degrees 4 16 64 256 --instance-seeds 101 102 103 \
        --budgets 1 3 10 60 300 --solver-seeds 1 2 3
    python run.py --out runs/dense --family full --sizes 3000 5000 7000 --instance-seeds 201 202 203
    python run.py --out runs/probe --family reg --degrees 256 --instance-seeds 101 --probe   # admission probe
    python run.py --out runs/x --family reg --degrees 256 --plan-only                          # sizes, no client

--probe sends the one largest model once, at 1 s, seed 1, and records whether it was admitted (a refusal is a row
with its error code, never dropped). Run it, recorded, before any sweep.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import examplekit as bk
import dense as D


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--family", choices=["reg", "full"], required=True)
    p.add_argument("--n", type=int, default=D.POST_REG["n"], help="nodes of the reg family")
    p.add_argument("--degrees", type=int, nargs="+", default=D.POST_REG["degrees"])
    p.add_argument("--sizes", type=int, nargs="+", default=D.POST_FULL["sizes"], help="n of the full family")
    p.add_argument("--instance-seeds", type=int, nargs="+", default=[101, 102, 103])
    p.add_argument("--budgets", type=float, nargs="+", default=[1, 3, 10, 60, 300])
    p.add_argument("--solver-seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--probe", action="store_true", help="admission probe: largest model, 1 s, seed 1, once")
    p.add_argument("--plan-only", action="store_true", help="print sizes and exit (no client)")
    p.add_argument("--scratch", default=None, help="where the upload file is written (deleted after upload)")
    p.add_argument("--save-solutions", action="store_true")
    return p.parse_args(argv)


def specs(a):
    if a.family == "reg":
        return [("reg", a.n, d) for d in a.degrees]
    return [("full", n, None) for n in a.sizes]


def license_limits(client):
    lic = client.license() if hasattr(client, "license") else {}
    return next((s.get("model_limits") for s in lic.get("solvers", [])
                 if isinstance(s, dict) and s.get("id") == "optimizer.qubo"), None)


def refusal(exc) -> dict:
    return {"error_code": getattr(exc, "code", None), "error": str(exc)[:500]}


def main(argv=None, client=None):
    import fourcolors
    import fourcolors.qubo

    a = parse(argv)
    plan = [D.size_report(f, n, d) for f, n, d in specs(a)]
    for rep in plan:
        print(f"  {rep['family']} n={rep['n']} degree={rep['degree']}: {rep['couplings']:,} couplings "
              f"({rep['matrix_nonzeros']:,} matrix entries), {rep['file_bytes_raw'] / 1e9:.2f} GB raw")
    if a.plan_only:
        return plan
    if a.probe:                                        # the largest model only, once
        biggest = max(specs(a), key=lambda s: D.expected_terms(*s))
        todo, budgets, seeds, iseeds = [biggest], [1.0], [1], a.instance_seeds[:1]
    else:
        todo, budgets, seeds, iseeds = specs(a), a.budgets, a.solver_seeds, a.instance_seeds
    out = Path(a.out)
    res = bk.Results(out / "results.jsonl")
    client = client if client is not None else bk.client()
    res.add(event="license", model_limits=license_limits(client), client_version=bk.client_version(),
            probe=a.probe)

    for family, n, d in todo:
        for iseed in iseeds:
            t0 = time.perf_counter()
            inst = D.build(family, n, iseed, d)
            common = dict(instance=inst["name"], family=family, n=n, degree=inst["degree"], instance_seed=iseed,
                          couplings=len(inst["i"]), instance_sha256=D.sha256(inst), probe=a.probe)
            scratch = Path(a.scratch or out / "scratch")
            scratch.mkdir(parents=True, exist_ok=True)
            path = scratch / f"{inst['name']}.4cqb.zst"
            info = D.write_file(inst, path)
            res.add(event="generate", **common, file_bytes=info["bytes_on_disk"], raw_bytes=info["bytes_raw"],
                    generate_s=round(time.perf_counter() - t0, 3), client_peak_rss_gb=round(bk.peak_rss_gb(), 3))
            try:
                t0 = time.perf_counter()
                input_id = fourcolors.qubo.upload(str(path), client=client)
                res.add(event="upload", **common, input_id=input_id, upload_wall_s=round(time.perf_counter() - t0, 3))
            except fourcolors.errors.FourColorsError as exc:
                res.add(event="refused", stage="upload", **common, **refusal(exc))
                print(f"  {inst['name']}: upload refused ({getattr(exc, 'code', None)})")
                continue
            finally:
                if path.exists():
                    os.remove(path)                    # disk is scarce; the id is what the jobs use
            for T in budgets:
                for seed in seeds:
                    t0 = time.perf_counter()
                    try:
                        r = fourcolors.qubo.solve(input_id=input_id, time_limit_s=T, seed=seed, client=client)
                    except fourcolors.errors.FourColorsError as exc:
                        res.add(event="refused", stage="solve", **common, budget_s=T, solver_seed=seed,
                                **refusal(exc))
                        print(f"  {inst['name']} T={T:g}s seed={seed}: refused ({getattr(exc, 'code', None)})")
                        continue
                    wall = time.perf_counter() - t0
                    row = dict(event="solve", arm="4colors", **common, input_id=input_id, budget_s=T,
                               solver_seed=seed, wall_s=round(wall, 3), status=r.status,
                               reason=bk.reason_code(r), objective_reported=r.objective,
                               engine_version=(getattr(r, "engine", None) or {}).get("version"),
                               client_version=bk.client_version())
                    job = getattr(r, "job", None)
                    for k in ("id", "billed_s", "run_s", "wait_s"):
                        row["job_" + k] = getattr(job, k, None) if job is not None else None
                    traj = list(getattr(r, "trajectory", None) or [])
                    row["first_answer_s"] = traj[0].get("t_search_s") if traj else None
                    row["trajectory"] = [[p.get("t_search_s"), p.get("objective")] for p in traj]
                    if r.solution is not None:
                        x = np.asarray(r.solution, dtype=np.int64)
                        e = D.energy(inst, x)                      # our own evaluation
                        row.update(energy=e, agrees=(r.objective == e))
                        if family == "reg":
                            row["cut"] = D.cut(inst, x)
                        if a.save_solutions:
                            (out / "solutions").mkdir(parents=True, exist_ok=True)
                            np.save(out / "solutions" / f"{inst['name']}-T{T:g}-s{seed}.npy",
                                    np.packbits(x > 0))
                    else:
                        row.update(energy=None, agrees=None)
                    res.add(**row)
                    print(f"  {inst['name']} T={T:g}s seed={seed}: {r.status}, energy={row['energy']}, "
                          f"agrees={row['agrees']}, wall {wall:.1f}s")
    return res.rows()


if __name__ == "__main__":
    main()
