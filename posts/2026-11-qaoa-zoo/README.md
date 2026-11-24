# The QAOA zoo, priced: what each variant costs in two-qubit gates

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=qaoa-zoo&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qaoa-zoo&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit,optimizer]" qiskit-aer qiskit-qasm3-import networkx scipy matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

The instances, circuits and noisy scoring come from `../../qaoa-blog/qaoakit/core.py`; run `run.py` and `plot.py`
from this directory.

### Step 1: the instances and their exact optima

21 graphs: 15 Max-Cut (unweighted 3-regular and weighted G(n, 0.3), 12 to 20 nodes) and 6 max-3-cut graphs
(one-hot, 12 to 18 qubits). Every optimum is found by enumerating all 2^n cuts or 3^n colorings, so every ratio is
against a certificate.

```python
from qaoakit import core
for inst in core.instances():
    cost = core.cost_vector(inst)            # the objective of every assignment
    print(inst["id"], cost.max())            # the exact optimum
```

### Step 2: the variants

| variant | circuit | angles |
|---|---|---|
| std-p1, p2, p3 | H, then per layer rzz(-gamma w) on every edge and rx(2 beta) on every node | optimized noiselessly (COBYLA; INTERP start for p > 1) |
| fixed-p1, p2, p3 | the same circuits | tree fixed angles of Wurtz and Lykov 2021 (3-regular graphs only; no optimizer loop) |
| ma-p1 | one gamma per edge, one beta per node | optimized from the std-p1 angles |
| warm-local-p1 | warm start (Egger et al. 2021, eps 0.25) from a 1-flip local optimum | optimized |
| warm-qubo-p1 | the same, from the QUBO optimizer's answer at 10 ms | optimized |
| rqaoa-p1 | recursive QAOA (Bravyi et al. 2020): p = 1, fix the strongest correlation, repeat down to 8 nodes, then enumerate | optimized per step |
| xy-p1, p2 | max-3-cut, one-hot, W-state start, XY mixer on each node's three qubits | optimized |

### Step 3: compile once, bind many

Each parametric circuit is compiled once per arm and seed, and every variant that shares it binds its own angles:

```python
import fourcolors
from qaoakit import apikit
heavy_hex = apikit.device_target()        # FakeFez's 169 working couplers, as a description (never a name)
res = fourcolors.quantum.compiler.compile(qc, target=heavy_hex, time_budget_s=5, seed=seed,
                                          approximation_mode="exact")
out = res.to_qiskit()
bound = out.assign_parameters(dict(zip(qc.parameters, angles)))
```

The Qiskit arm is `transpile(..., optimization_level=3)` on the same couplers, new seeds until the same 5 seconds are
spent, fewest two-qubit gates kept.

### Step 4: price RQAOA over all its steps

RQAOA is not one circuit. Each recursion step removes one node, so a 20-node graph needs 12 different circuits, each
sampled enough to estimate every edge correlation. Pricing it at its first circuit would understate it by an order
of magnitude. The zoo reports it two ways and never mixes them:

- **rqaoa-p1, recursion simulated classically.** The correlations come from the exact p = 1 state, which is
  classically computable at p = 1. No circuit is compiled, so this row has a ratio but no gate price.
- **rqaoa-p1 (all steps).** Every step's circuit is compiled by the arm, simulated under the noise model, and its
  correlations are estimated from 1000 shots. The price counts the two-qubit gates of every step's circuit, summed,
  and the row records the total shots.

```bash
python run.py --out runs/zoo --rqaoa-noisy-seeds 1 --rqaoa-shots 1000
```

### Step 5: score, price and draw

```bash
python run.py --out runs/zoo                    # all 21 instances, all variants, 5 seeds
python plot.py --out runs/zoo
```

The price of a row is `two-qubit gates / (100 x (noisy ratio - random-guess ratio))`: the gates one percentage point
above a zero-gate random guess costs. The QUBO bar on the same graph:

```python
r = fourcolors.qubo.solve(Q, time_limit_s=0.01, seed=1)     # Q: x^T Q x = -cut(x)
ratio = -r.objective / max_cut
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check the uniform-state expectation, the tree fixed angles against their published cut fractions, that
the XY mixer keeps the one-hot subspace, that RQAOA recovers the optimum on tiny graphs, the parametric warm start,
best-of-shots, the price and every no-price reason, that no-price rows are counted and not dropped, that the
selection shows only strict wins, that RQAOA is priced over every step (gates and shots summed, never the first
step's circuit), the sampled correlations, and the plumbing of `run.py` and `plot.py` on canned results.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qaoa-zoo&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
