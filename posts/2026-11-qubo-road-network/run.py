"""Max-Cut or balanced partition of a road network: ours through the API, and a local-search baseline.

    python run.py --out results/utrecht --graphml utrecht.graphml --problem maxcut --budgets 10 60 300
    python run.py --out results/utrecht-split --graphml utrecht.graphml --problem bisection --imbalance 0.01
    python run.py --out results/x --place "Utrecht, Netherlands" --allow-download --save-graphml utrecht.graphml

Rows in <out>/results.jsonl:
  event=graph     nodes, edges, weight kind, where the graph came from
  event=solve     ours: cut re-evaluated here, balance (bisection), wall and billed seconds, job id
  event=baseline  the local search (Max-Cut: from a random split; bisection: from the median split), with
                  its own time limit
<out>/solutions/*.npz hold node coordinates, edges and each best answer, for draw_map.py.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import examplekit as bk
import fourcolors.qubo
import roads

# About 1 GB of service memory per million variables: 6 million intersections leave headroom under the standard
# plan's 8 GiB job cap. A bigger graph is not sent (a recorded "skipped" row), never split or shrunk silently.
MAX_NODES = 6_000_000


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--graphml", help="a graph saved by osmnx")
    src.add_argument("--osm", help="an OSM XML file (convert .osm.pbf with osmium first)")
    src.add_argument("--place", help="a place name for osmnx (downloads; needs --allow-download)")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--save-graphml", default=None, help="save the loaded graph, to reuse it offline")
    p.add_argument("--network-type", default="drive")
    p.add_argument("--weight", choices=["unit", "length"], default="unit")
    p.add_argument("--problem", choices=["maxcut", "bisection"], default="maxcut")
    p.add_argument("--imbalance", type=float, default=0.01, help="bisection: allowed side imbalance, share of n")
    p.add_argument("--budgets", type=float, nargs="+", default=[10, 60, 300])
    p.add_argument("--solver-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--baseline-time", type=float, nargs="+", default=None,
                   help="local-search time limits (default: the budgets)")
    p.add_argument("--baseline-seed", type=int, default=0)
    p.add_argument("--name", default=None, help="instance name in the rows (default: from the source)")
    p.add_argument("--arms", nargs="+", choices=["4colors", "local-search"], default=["4colors", "local-search"],
                   help="which arms to run (the API arm and the timed local baseline can run as separate records)")
    p.add_argument("--max-nodes", type=int, default=MAX_NODES,
                   help="refuse to upload a graph with more intersections than this (standard plan: %(default)s)")
    return p.parse_args(argv)


def main(argv=None, client=None, graph=None):
    a = parse(argv)
    out = Path(a.out)
    res = bk.Results(out / "results.jsonl")
    t0 = time.perf_counter()
    G = graph if graph is not None else roads.load_graph(graphml=a.graphml, osm=a.osm, place=a.place,
                                                         allow_download=a.allow_download,
                                                         network_type=a.network_type)
    if a.save_graphml:
        import osmnx as ox
        ox.save_graphml(G, a.save_graphml)
    g = roads.to_simple(G, a.weight)
    del G
    name = a.name or Path(a.graphml or a.osm or a.place.replace(",", "").replace(" ", "_")).stem
    inst = f"{name}-{a.problem}-{a.weight}"
    base = dict(instance=inst, problem=a.problem, weight=a.weight, n=g["n"], m=int(len(g["u"])))
    res.add(event="graph", **base, source=a.graphml or a.osm or a.place, load_s=round(time.perf_counter() - t0, 3),
            total_weight=int(g["w"].sum()), credit=roads.OSM_CREDIT)
    print(f"{inst}: {g['n']:,} nodes, {len(g['u']):,} edges; service memory about "
          f"{bk.service_memory_gb(g['n']):.1f} GB")
    sol_dir = out / "solutions"; sol_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sol_dir / f"{inst}-graph.npz", x=g["x"], y=g["y"], u=g["u"], v=g["v"], w=g["w"])

    if g["n"] > a.max_nodes:
        res.add(event="skipped", **base, reason=f"{g['n']} intersections > --max-nodes {a.max_nodes} "
                "(standard plan, 8 GiB job cap): pick a smaller region")
        print(f"  skipped: {g['n']:,} > {a.max_nodes:,} intersections; pick a smaller region")
        return res.rows()

    maximize = a.problem == "maxcut"
    if maximize:
        model, bounds = roads.maxcut_model(g), None
    else:
        model, bounds = roads.bisection_model(g, a.imbalance)
        base.update(lo=bounds[0], hi=bounds[1])

    def score(x):
        row = {"cut": roads.cut_value(g, x)}
        if bounds:
            ones = int(np.sum(x))
            row.update(ones=ones, balanced=bool(bounds[0] <= ones <= bounds[1]))
        return row

    if "4colors" in a.arms:
        _api_arm(a, res, base, g, model, score, sol_dir, inst, client)
    if "local-search" in a.arms:
        _local_arm(a, res, base, g, bounds, maximize, score, sol_dir, inst)
    return res.rows()


def _api_arm(a, res, base, g, model, score, sol_dir, inst, client):
    """Upload once, then one job per budget and seed. Seconds here are service-side (billed, search)."""
    client = client if client is not None else bk.client()
    t0 = time.perf_counter()
    input_id = fourcolors.qubo.upload(model, client=client)
    res.add(event="upload", **base, input_id=input_id, upload_wall_s=round(time.perf_counter() - t0, 3))
    for T in a.budgets:
        for seed in a.solver_seeds:
            t0 = time.perf_counter()
            r = fourcolors.qubo.solve(input_id=input_id, time_limit_s=T, seed=seed, client=client)
            wall = time.perf_counter() - t0
            row = dict(event="solve", solver="4colors", **base, budget_s=T, solver_seed=seed, wall_s=round(wall, 3),
                       status=r.status, reason=bk.reason_code(r), objective_reported=r.objective,
                       engine_version=(getattr(r, "engine", None) or {}).get("version"), client_version=bk.client_version())
            job = getattr(r, "job", None)
            for k in ("id", "billed_s", "run_s", "wait_s"):
                row["job_" + k] = getattr(job, k, None) if job is not None else None
            if r.solution is not None:
                x = np.asarray(r.solution, dtype=np.int64)
                row.update(score(x), agrees=(roads.cut_value(g, x) == r.objective))
                np.save(sol_dir / f"{inst}-4colors-T{T:g}-s{seed}.npy", np.packbits(x.astype(bool)))
            res.add(**row)
            print(f"  4colors T={T:g}s: cut={row.get('cut')} {row.get('balanced', '')}")


def _local_arm(a, res, base, g, bounds, maximize, score, sol_dir, inst):
    """Our own baseline on one local core: its wall time is only quotable from a timed record."""
    for T in a.baseline_time or a.budgets:
        rng = np.random.default_rng(a.baseline_seed)
        if maximize:
            x0 = rng.integers(0, 2, g["n"])
        else:
            x0 = roads.geometric_split(g, *bounds)
        start = score(x0)
        x, rounds, secs = roads.local_search(g, x0, maximize=maximize, time_limit_s=T, seed=a.baseline_seed,
                                             balanced=not maximize)
        res.add(event="baseline", solver="local-search", **base, budget_s=T, wall_s=round(secs, 3), rounds=rounds,
                start_cut=start["cut"], **score(x))
        np.save(sol_dir / f"{inst}-local-T{T:g}.npy", np.packbits(x.astype(bool)))
        print(f"  local search T={T:g}s: cut={roads.cut_value(g, x)} after {rounds} rounds")


if __name__ == "__main__":
    main()
