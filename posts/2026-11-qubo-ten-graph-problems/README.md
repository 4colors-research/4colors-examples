# Ten graph problems as QUBOs, ten lines each

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-ten-graph-problems&utm_content=readme-top)

## The process, step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-ten-graph-problems&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" networkx
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

Every graph comes from a networkx generator with a fixed seed (`graphs.instances`), so you build the same instances
we did. No dataset is downloaded.

### The pattern

Each builder in `graphs.py` does three things:

1. **Variables.** One 0/1 variable per node (cut, set, cover, clique, dominating set), per (node, color) (coloring),
   per edge (matchings) or per (city, step) (tour).
2. **Objective and rows.** A linear or quadratic objective, `sense="max"` where you maximize, and one exact row per
   hard rule.
3. **Decode and check.** `decode(x)` turns the 0/1 answer into a node set, a coloring, an edge list or a tour.
   `check(answer)` verifies it in plain Python, without the solver, and scores it.

Solving any of them is the same call:

```python
import fourcolors.qubo, networkx as nx
import graphs

prob = graphs.max_independent_set(nx.gnp_random_graph(60, 0.08, seed=7))
r = fourcolors.qubo.solve(prob.model, time_limit_s=1, seed=1)
answer = prob.decode(r.solution)
print(prob.check(answer), r.constraints)      # {"valid": True, "score": ...}, the service's own row check
```

The search runs to its time limit, so 1 s is enough to try each one. [OD-10 CHECK: billing is per search second at
the time limit; confirm the wording with the platform team before publication.]

### 1. Max-Cut: no rows at all

Split the nodes into two sides so that as many edges as possible cross. An edge (u, v) is cut when exactly one end is
on side 1: x_u + x_v - 2 x_u x_v.

```python
Q, lin = {}, [0] * n
for u, v in G.edges():
    lin[u] += 1; lin[v] += 1
    Q[(u, v)] = Q.get((u, v), 0) - 2
m = fourcolors.qubo.to_model(Q, lin, n=n, sense="max")
```

### 2. Maximum independent set: one row per edge

As many nodes as possible, no two adjacent.

```python
m = fourcolors.qubo.to_model({}, [1] * n, n=n, sense="max")
for u, v in G.edges():
    m.add_constraint({u: 1, v: 1}, hi=1)
```

The Lucas form is `min -sum x + P * sum_edges x_u x_v` with P > 1 (`graphs.penalty_form_independent_set`). On this
unweighted problem any P > 1 works. Add weights, or a second kind of rule, and P becomes a parameter you sweep. With
the row there is nothing to sweep.

### 3. Minimum vertex cover: the same rows, the other direction

As few nodes as possible such that every edge has an end chosen: `m.add_constraint({u: 1, v: 1}, lo=1)` per edge,
minimize the sum. (The complement of an independent set is a cover, and the tests check that the two optima add up
to n.)

### 4. Maximum clique: rows on the non-edges

A clique in G is an independent set in G's complement, so the rows go on the pairs that are *not* edges:

```python
for u, v in itertools.combinations(G.nodes(), 2):
    if not G.has_edge(u, v):
        m.add_constraint({u: 1, v: 1}, hi=1)
```

### 5. Graph coloring: one-hot rows plus a conflict objective

One variable per (node, color), exactly one color per node as a row, and the number of edges whose two ends share a
color as the objective. Score 0 means a proper coloring. This is the same one-hot pattern as our Sudoku and map
coloring post.

```python
for v in G.nodes():
    m.add_constraint({v * k + c: 1 for c in range(k)}, lo=1, hi=1)
# objective: +1 for x[u,c] * x[v,c] on every edge (u, v) and color c
```

### 6. Balanced partitioning (bisection): Max-Cut turned around, plus one row

Minimize the cut edges, with exactly half the nodes on each side: `m.add_constraint({i: 1 for i in range(n)},
lo=n // 2, hi=n // 2)`. In the penalty form this one row becomes a dense n x n term `P * (sum x - n/2)^2`, which adds a
quadratic entry for every pair of nodes. As a row it stays one row.

### 7. Minimum dominating set: one row per node

Every node is chosen or has a chosen neighbor: `m.add_constraint({u: 1 for u in [v, *G.neighbors(v)]}, lo=1)`.

### 8. Maximum-weight matching: variables on edges

One variable per edge, maximize the chosen weight, at most one chosen edge at each node. (This one is solvable in
polynomial time by Edmonds' blossom algorithm, which networkx ships as `max_weight_matching`. It is here as a check:
the tests compare the model's optimum with it.)

```python
edges = list(G.edges())
m = fourcolors.qubo.to_model({}, [G[u][v]["weight"] for u, v in edges], n=len(edges), sense="max")
for v in G.nodes():
    m.add_constraint({k: 1 for k, e in enumerate(edges) if v in e}, hi=1)
```

### 9. Minimal maximal matching: a Lucas classic

The smallest matching that cannot be extended. It takes the matching rows from step 8, plus one "maximal" row per
edge: the edge or some edge touching one of its ends is chosen. In Lucas's paper this needs three penalty terms with
carefully ordered weights. Here it is two kinds of row and the objective `sum y`.

### 10. Traveling salesperson: small, and honest about it

x[c, t] = 1 when city c is visited at step t. Exactly one city per step and one step per city (2n rows), and the
objective adds the distance between consecutive steps. It works, and the tests check it against a brute force over
every tour. But n cities need n^2 variables and a dense quadratic objective. QUBO is the wrong tool for real tours:
dedicated routing heuristics handle thousands of cities directly. We keep it here at 7 cities because it is the
formulation everyone asks about. [OD-11: for real routing problems, point to the 4colors Optimizer, our routing solver; CTA-only mention
pending the owner's decision.]

### Run all ten

```bash
python run.py --out runs/ten --time-limit 1 --seeds 1
```

Each line reports the plain-Python verdict and the score:

`results.jsonl` has one row per problem and seed: the instance size (`n_nodes`, `n_edges`, `n_vars`, `n_rows`), the
decoded `answer`, `valid` and `score` from plain Python, the client's exact row check (`model_feasible`), the
service's (`service_feasible`), and the service's fields (`engine_version`, `billed_s`, the trajectory).

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

For every problem the tests enumerate all assignments of a tiny instance and check that the model's best feasible
answer, decoded and checked in plain Python, equals an independent brute force on the graph (or
`networkx.max_weight_matching` for matching). They also check that the validators reject broken answers, that the
penalty form of the independent set has the same optimum, and that `run.py` writes and checks its rows on fixed
answers. They need networkx; without it the tests are skipped.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-ten-graph-problems&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
