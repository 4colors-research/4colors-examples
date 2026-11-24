# Cutting a road network in two

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-road-network&utm_content=readme-top)

## What you will learn

- How to turn an OpenStreetMap extract into a simple weighted graph: one variable per intersection, one edge per
  pair of joined intersections.
- How Max-Cut and the balanced partition are the same QUBO matrix with opposite senses, and how the balance goes in
  as one exact row instead of a dense penalty.
- How to upload a large model once and solve it at several time budgets without sending it again.
- How to check each answer (cut re-counted, balance re-checked) and draw it on the map.

## The model

Nodes are intersections and dead ends (osmnx's simplified graph); roads between the same two nodes, in either
direction, become one edge. Weights are 1 (`--weight unit`) or the road length in whole meters (`--weight length`),
so every coefficient is an integer and every energy is exact.

With x_i in {0, 1} the side of node i, the number of cut roads is

    cut(x) = sum over roads (i, j) of w_ij (x_i + x_j - 2 x_i x_j) = x^T Q x,

with Q_ii = sum_j w_ij on the diagonal and Q_ij = -2 w_ij in the upper triangle (i < j, each road once). Max-Cut
maximizes it. The balanced partition minimizes it subject to one row:

    n/2 - k <= sum_i x_i <= n/2 + k,     k = imbalance * n (1% here)

| Problem | Variables | Matrix nonzeros | Rows |
|---|---|---|---|
| Max-Cut | n (one per intersection) | n + m (m roads) | none |
| Balanced partition | n | n + m | 1 row with n nonzeros |
| Same partition as a penalty QUBO (for comparison, not run) | n | about n^2 / 2 | none |

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-road-network&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-road-network&utm_content=readme-text).
2. Install the client and the map tools, and point the client at the key:

```bash
pip install "fourcolors[optimizer]" osmnx numpy scipy matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

For a whole province or country you also need `osmium` (osmium-tool) to cut the roads out of a Geofabrik extract.

## Step 1: get the map

The scripts never download anything unless you ask. Start with a city (osmnx fetches it from the Overpass API):

```bash
python run.py --out results/utrecht --place "Utrecht, Netherlands" --allow-download \
    --save-graphml utrecht.graphml --problem bisection --budgets 10
```

Save it once with `--save-graphml`, then use `--graphml utrecht.graphml` from then on.

For a province or a country, `prepare_region.sh` checks the free disk first, downloads one Geofabrik extract, keeps
the drivable roads with osmium, builds the graph once, saves it as GraphML and prints its size:

```bash
bash prepare_region.sh ~/maps netherlands       # prints nodes=..., edges=..., fits_standard_plan=...
```

How big can the region be? About 1 GB of service memory per million variables is a fair rule of thumb, so the
standard plan's 8 GiB job cap holds about 6 million intersections with headroom. A province or a small country
fits; a large country does not. `run.py` refuses to upload a graph above `--max-nodes` (6,000,000) and records why.
Building the graph on your own machine takes more memory than solving it (osmnx needs roughly 1 KB per node and per
edge while it builds), so a country wants a machine with tens of GB, once.

**License.** OpenStreetMap data is (c) OpenStreetMap contributors, under the Open Database License (ODbL). Credit
"(c) OpenStreetMap contributors" on every map, chart or table made from it (the scripts add the line), and share
any database you derive from it under the ODbL.

## Step 2: the QUBO in a few lines

`roads.py` builds the matrix from the edge list with scipy, then:

```python
import fourcolors.qubo
import roads

g = roads.to_simple(G, "unit")                                   # x, y, edge arrays u, v, w
m, (lo, hi) = roads.bisection_model(g, imbalance=0.01)           # Q (min cut) + one row lo <= sum x <= hi
model_id = fourcolors.qubo.upload(m)                             # sent once
for T in (10, 60, 300):
    r = fourcolors.qubo.solve(input_id=model_id, time_limit_s=T, seed=0)
    x = r.solution
    print(T, roads.cut_value(g, x), lo <= sum(x) <= hi)          # re-count the cut, re-check the balance
```

Max-Cut is `roads.maxcut_model(g)`: the same matrix with `sense="max"` and no row.

## Step 3: solve and compare

```bash
python run.py --out results/region-split --graphml ~/maps/netherlands.graphml --name region \
    --problem bisection --imbalance 0.01 --budgets 10 60 300
python run.py --out results/region-maxcut --graphml ~/maps/netherlands.graphml --name region \
    --problem maxcut --budgets 10 60 300
```

Each `solve` row records the cut as we re-count it, whether it agrees with the service's objective, the balance,
the billed seconds and the release that answered. As a yardstick, `run.py` also runs a simple local search of our
own on one core at the same time limits: greedy one-flip moves for Max-Cut from a random split, and balanced swaps
for the partition starting from the median split along the map's longer axis (a good geometric start, and what most
people would try first).

## Step 4: pictures

```bash
python plot.py results/region-*/results.jsonl --out figures
python draw_map.py results/region-split/solutions/region-bisection-unit-graph.npz \
    results/region-split/solutions/region-bisection-unit-4colors-T300-s0.npy --out figures/region-split.png
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests build a small toy road graph and check that directions, parallel roads and loops merge correctly, that the
QUBO equals the cut on every assignment (brute force), that the balance row admits exactly the balanced answers,
that the local search improves its start and keeps the balance, that a graph above `--max-nodes` is never uploaded,
that the chart shows only the budgets the selection rule allows, and run the scripts' plumbing (rows, plot, map)
on a fixed canned answer.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
