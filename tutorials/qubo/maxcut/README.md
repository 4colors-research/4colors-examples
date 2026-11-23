# Max-Cut on a large graph

**Docs:** [this tutorial in the 4colors docs](https://docs.4colors-research.com/start/tutorials/qubo-maxcut/?utm_source=github&utm_medium=examples&utm_campaign=qubo-maxcut&utm_content=readme-top) · Product: 4colors Optimizer for QUBO · Level: beginner

Max-Cut splits the nodes of a graph into two groups so that as many edges as possible cross between the groups. It
is a natural QUBO: one binary variable per node says which group the node is in. The whole script is
[`maxcut.py`](maxcut.py).

1. Start a free trial at [TRIAL SIGNUP URL] and copy your key.
2. Run the tutorial:

```bash
pip install "fourcolors[optimizer]"
export FOURCOLORS_API_KEY=...                 # the key from step 1, or FOURCOLORS_KEY_FILE=/path/to/file
python maxcut.py                              # 2,000 nodes, 6,000 edges, 30 s
```

## Write the problem as a QUBO

Edge (i, j) is cut when x<sub>i</sub> + x<sub>j</sub> − 2 x<sub>i</sub> x<sub>j</sub> = 1. So minimizing

E(x) = Σ<sub>edges</sub> (2 x<sub>i</sub> x<sub>j</sub> − x<sub>i</sub> − x<sub>j</sub>)

maximizes the cut, and the cut size is −E.

## Build the model

A random graph with 2,000 nodes and 6,000 edges:

```python
import random

n, m = 2000, 6000
rnd = random.Random(7)
edges = set()
while len(edges) < m:
    i, j = rnd.sample(range(n), 2)
    edges.add((min(i, j), max(i, j)))

Q = {e: 2 for e in edges}  # quadratic terms: 2 x_i x_j per edge
linear = [0] * n           # linear terms: -1 per edge end
for i, j in edges:
    linear[i] -= 1
    linear[j] -= 1
```

## Solve it and watch the cut grow

```python
import fourcolors

job = fourcolors.qubo.submit(Q, time_limit_s=30, linear=linear)  # returns at once

for best in job.incumbents():  # each better solution as it is found
    print(f"cut {-best.objective} of {m} edges")

result = job.result()
print(result.status, -result.objective)
```

The job queues, starts, and streams each improvement. The last item `incumbents()` yields is the final result.

Press **Ctrl-C** at any time: the job stops on the service and you get the best cut found so far, with
`status == "stopped"`. You are billed only for the time it ran.

## Check the answer yourself

The result's solution is yours to check. Count the cut edges directly:

```python
x = result.solution  # one 0/1 value per variable
cut = sum(1 for i, j in edges if x[i] != x[j])
print(f"{cut} edges cross between groups of {x.count(0)} and {x.count(1)} nodes")
```

For a plain model like this one, the service returns the search's own objective without computing it a second
time, so all of your time limit goes to the search; answers to models with constraints get one exact check. See
[How answers are checked](https://docs.4colors-research.com/qubo/verification/?utm_source=github&utm_medium=examples&utm_campaign=qubo-maxcut&utm_content=readme).

## See how the search went

`result.trajectory` lists every improvement the search made, with the search time it was found at:

```python
for step in result.trajectory[-5:]:
    print(f"t={step['t_search_s']:.2f} s  objective {step['objective']}")
```

If the last improvement came early in the time limit, a shorter limit would likely give the same answer; if
improvements kept coming until the end, try a longer one.

## Tests

```bash
python -m pytest -q tests
```

They check that the QUBO energy is minus the cut on every assignment of a small graph, and that the script
prints, checks and returns an answer correctly. For that they feed it fixed, hand-written answers
(`tests/canned.py`); nothing in the tests solves the model, so the tests say nothing about answer quality.

## Next

- [Large QUBO models: upload once, reuse](https://docs.4colors-research.com/start/tutorials/qubo-large-models/?utm_source=github&utm_medium=examples&utm_campaign=qubo-maxcut&utm_content=readme-next).
- [Results](https://docs.4colors-research.com/qubo/results/?utm_source=github&utm_medium=examples&utm_campaign=qubo-maxcut&utm_content=readme-next) for every field of a result.
- The same tutorial as a notebook: [`notebooks/qubo-maxcut.ipynb`](../../../notebooks/qubo-maxcut.ipynb).

---
Adapted from the 4colors documentation tutorial "Max-Cut on a large graph".
