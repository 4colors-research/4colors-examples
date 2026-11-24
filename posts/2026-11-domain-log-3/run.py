"""Route assignment for thousands of cars on a real street map, in the 2017 paper's formulation, through the API.

    python run.py --out runs/traffic --graph grid:30x30 --cars 418 --time-limit 1 --seeds 1        # no download
    python run.py --out runs/traffic --graph osm:city.osm --cars 418 4180 10000 --time-limit 1 10 --seeds 1 2 3
    python run.py --out runs/traffic --graph osm:city.osm --cars 418 4180 10000 --count-only       # sizes only

Each car gets --routes candidates (shortest path plus penalty-method alternatives), one exact "exactly one route"
row, and the objective is the paper's congestion, the sum over street segments of the squared number of cars.
Before any job the script counts the model's terms and skips (with a recorded row) any size above --max-terms.
Baselines in plain Python: everyone on the shortest path, random routes (the paper's baseline) and best-response
dynamics. Every answer is decoded and its congestion recomputed in plain Python. The rival arm `sa-fill` [OD-2]
runs dwave-samplers simulated annealing on the penalty form and is only quotable from a timed run.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors
import fourcolors.qubo

import examplekit
import traffic

KEEP_MAP_UP_TO = 30000            # store coordinates and segment loads (for the map figure) up to this many nodes
MAX_TERMS = 13_000_000            # default skip threshold for quadratic terms [CHECK against the live limits]


def load_graph(spec: str):
    kind, _, arg = spec.partition(":")
    if kind == "grid":
        w, h = (int(v) for v in arg.lower().split("x"))
        G, xy = traffic.grid_city(w, h)
        return G, xy, {"source": f"grid {w}x{h}", "sha256": None}
    if kind == "osm":
        G, xy = traffic.osm_graph(arg)
        sha = hashlib.sha256(Path(arg).read_bytes()).hexdigest()
        return G, xy, {"source": "OpenStreetMap extract (ODbL)", "file": Path(arg).name, "sha256": sha}
    raise SystemExit(f"--graph must be grid:WxH or osm:PATH, not {spec!r}")


def seg_loads(routes, choice):
    return [[u, v, L] for (u, v), L in sorted(traffic.loads(routes, choice).items())]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="directory for results.jsonl")
    p.add_argument("--graph", default="grid:30x30", help="grid:WxH or osm:PATH (default %(default)s)")
    p.add_argument("--cars", type=int, nargs="+", default=[418, 4180, 10000])
    p.add_argument("--routes", type=int, default=3, help="candidate routes per car (the paper used 3)")
    p.add_argument("--instance-seed", type=int, default=1)
    p.add_argument("--time-limit", dest="budgets", type=float, nargs="+", default=[1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--arms", nargs="+", default=["4colors"], choices=["4colors", "sa-fill"])
    p.add_argument("--max-terms", type=int, default=MAX_TERMS)
    p.add_argument("--timed", action="store_true", help="mark rival rows timed (only inside a booked, pinned run)")
    p.add_argument("--count-only", action="store_true", help="build and count the models; send nothing")
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")

    G, xy, meta = load_graph(a.graph)
    n_seg = sum(len(nb) for nb in G.values()) // 2
    small = len(G) <= KEEP_MAP_UP_TO
    out.add(kind="graph", **meta, nodes=len(G), segments=n_seg,
            xy={str(k): v for k, v in xy.items()} if small else None,
            segment_list=[[u, v] for u in G for v in G[u] if u < v] if small else None)
    print(f"graph: {meta['source']}, {len(G)} nodes, {n_seg} segments")
    client = None if a.count_only else examplekit.client()

    for n_cars in a.cars:
        cars = traffic.random_cars(G, n_cars, a.instance_seed)
        routes = [traffic.candidate_routes(G, s, t, a.routes) for s, t in cars]
        size = traffic.count_terms(routes)
        inst = dict(cars=n_cars, routes_per_car=a.routes, instance_seed=a.instance_seed, **size)
        skip = size["n_quad"] > a.max_terms
        out.add(kind="instance", **inst, skipped_over_max_terms=skip, max_terms=a.max_terms,
                mean_candidates=round(size["n_vars"] / n_cars, 3), local_peak_rss_gb=round(examplekit.peak_rss_gb(), 3))
        print(f"{n_cars} cars: {size['n_vars']} vars, {size['n_quad']} quadratic terms, {size['n_rows']} rows"
              + ("  SKIPPED (over --max-terms)" if skip else ""))
        if skip or a.count_only:
            continue

        short = traffic.all_shortest(routes)
        bases = {"shortest": (short, None), "random": (traffic.random_choice(routes, a.instance_seed), None)}
        br, rounds = traffic.best_response(routes, short)
        bases["best-response"] = (br, rounds)
        for name, (choice, rnd) in bases.items():
            out.add(kind="baseline", method=name, **inst, **traffic.summary(G, routes, choice), rounds=rnd,
                    segment_loads=seg_loads(routes, choice) if small and name == "shortest" else None)

        m = traffic.build_model(routes) if "4colors" in a.arms else None
        for T in a.budgets:
            for seed in a.seeds:
                common = dict(**inst, budget_s=T, seed=seed)
                if "4colors" in a.arms:
                    try:
                        r = fourcolors.qubo.solve(m, time_limit_s=T, seed=seed, client=client)
                    except fourcolors.errors.FourColorsError as exc:      # a refusal is recorded, not hidden
                        out.add(kind="solve", arm="4colors", **common, error_code=getattr(exc, "code", None),
                                error=str(exc))
                        print(f"  4colors T={T:g} seed={seed}: refused: {exc}")
                    else:
                        x = r.solution
                        choice = None if x is None else traffic.decode(routes, x)
                        summ = traffic.summary(G, routes, choice) if choice is not None else {}
                        out.add(kind="solve", arm="4colors", **common, valid=choice is not None, **summ,
                                energy_python=None if x is None else m.energy(x),
                                model_feasible=None if x is None else m.feasible(x),
                                service_feasible=(r.constraints or {}).get("feasible"),
                                segment_loads=seg_loads(routes, choice) if small and choice is not None else None,
                                **examplekit.result_fields(r))
                        print(f"  4colors T={T:g} seed={seed}: congestion={summ.get('congestion')} "
                              f"reason={examplekit.reason_code(r)}")
                if "sa-fill" in a.arms:
                    import rival
                    x, extra = rival.anneal(routes, T, seed)
                    choice = traffic.decode(routes, x)
                    summ = traffic.summary(G, routes, choice) if choice is not None else {}
                    out.add(kind="solve", arm="sa-fill", **common, valid=choice is not None, **summ, **extra,
                            timed=a.timed, rival="dwave-samplers SA, penalty form [OD-2]")
                    print(f"  sa-fill T={T:g} seed={seed}: congestion={summ.get('congestion')}")


if __name__ == "__main__":
    main()
