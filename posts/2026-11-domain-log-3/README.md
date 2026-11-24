# Traffic flow, eight years on: route assignment for 10,000 cars from OpenStreetMap

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=domain-log-3&utm_content=readme-top)

## What you will learn

- How to turn an OpenStreetMap extract into a drivable street graph in plain Python (no routing engine).
- How to give each car a few candidate routes: the shortest path plus alternatives by the penalty method.
- The paper's congestion objective as a QUBO, with "exactly one route per car" as an exact row instead of a penalty
  weight, and why pairs of routes of the same car can be left out.
- How to count a model's terms before you send it, so you know whether it fits.
- How to check each returned answer: decode one route per car and recompute the sum of squared loads yourself.

## The model

For car i and candidate route j, a 0/1 variable q[i, j]. One exact row per car: `sum_j q[i, j] == 1`. The objective
is the paper's:

    H = sum over street segments s of ( number of chosen routes that use s )^2

Expanded, each route gets a linear term (its number of segments, because q^2 = q), and each pair of routes of
**different** cars gets 2 x (number of segments they share). Pairs of the same car's routes never both equal 1
under the row, so they are left out: fewer terms, same answers. With the rows in force, the model's energy equals
the recomputed congestion exactly; the unit tests check it on every assignment of a small case.

The paper wrote the one-route rule as a penalty `lambda * (1 - sum_j q[i, j])^2` and had to choose lambda. The row
needs no weight. (The penalty form is still built, with a lambda large enough to be safe, for the rival arm in
"How we measured".)

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=domain-log-3&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=domain-log-3&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

3. Download a street extract of your city as an .osm XML file, for example a bounding box from the Overpass API
   (c) OpenStreetMap contributors, available under the Open Database License.

## Step 1: graph, cars, routes

```python
import traffic

G, xy = traffic.osm_graph("city.osm")                 # drivable ways, largest connected piece, lengths in meters
cars = traffic.random_cars(G, 418, seed=1)            # origin-destination pairs
routes = [traffic.candidate_routes(G, s, t, k=3) for s, t in cars]
print(traffic.count_terms(routes))                    # variables, quadratic terms, rows, busiest segment
```

`candidate_routes` returns the shortest path first, then reruns Dijkstra with every already used segment made 1.6
times longer, which yields distinct alternatives that avoid the first route where they can.

## Step 2: build and solve

```python
import fourcolors.qubo

m = traffic.build_model(routes)                       # congestion objective + one exact row per car
r = fourcolors.qubo.solve(m, time_limit_s=1, seed=1)
choice = traffic.decode(routes, r.solution)           # one route index per car (None if a row were broken)
print(traffic.congestion(routes, choice), r.constraints["feasible"])
```

Every constrained answer is checked by the service against its rows; `decode` and `congestion` check it again in
plain Python.

## Step 3: the sizes, the baselines and the pictures

```bash
python run.py --out runs/traffic --graph osm:city.osm --cars 418 4180 10000 --count-only     # sizes only
python run.py --out runs/traffic --graph osm:city.osm --cars 418 4180 10000 --time-limit 1 10 --seeds 1 2 3
python plot.py runs/traffic --out figures
```

The script first counts every model's terms and skips any size above `--max-terms` with a recorded row, so nothing
oversized is sent. For each size it also records three plain-Python baselines: everyone on the shortest path,
random routes (the 2017 paper's baseline), and best-response dynamics (each car in turn switches to its least
congested candidate until no car can improve), the classical first thing a transport engineer would try.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that the model's energy equals the recomputed congestion on every assignment of a small case, that
the rows reject zero or two routes, that same-car pairs are left out, that candidate routes are distinct paths with
the shortest first, that best-response never makes things worse and stops at an equilibrium, that the term count
matches the built model, and that the OSM parser keeps only drivable, accessible ways in the largest connected piece
with correct lengths; and they run the scripts' plumbing (rows, skips, the chart's selection rule, plots) on fixed
hand-made answers.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
