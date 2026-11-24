# The referee: re-scoring a published QPU benchmark with a real optimum

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=referee-qpu-benchmark&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key: start a free trial at https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=referee-qpu-benchmark&utm_content=readme-text, then create a key at account.4colors-research.com
   (docs: docs.4colors-research.com).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime networkx matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

Run the scripts from this directory (they import `../../shared/examplekit.py` and `../../shared/circuitkit.py`).

### Step 1: one Q-score instance and its reference

```python
import fourcolors.qubo
import referee as R

n = 100
edges = R.er_half(n, seed=1)                          # G(100, 1/2), about 2,475 edges
m = R.maxcut_model(n, edges)                          # maximize the cut; integer coefficients, no rows
r = fourcolors.qubo.solve(m, time_limit_s=10, seed=1)
cut = R.cut_value(edges, r.solution)                  # recomputed in plain Python
print(cut, R.expected_optimum(n))                     # best found vs Q-score's asymptotic reference
print(R.beta_needed(n, cut))                          # what a solver that always found this cut would score
```

### Step 2: a certificate where one is possible

```python
best, x = R.exact_maxcut(24, R.er_half(24, seed=1))   # exhaustive: 2^23 assignments, in chunks
```

For n <= 24 the exhaustive search proves the optimum, and the post checks that the optimizer reached it.

### Step 3: the benchmark's circuits, compiled

```python
import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

e, w = R.fc_weighted(40, seed=1)                      # fully connected, integer weights 1..10
qc = R.lr_qaoa(40, e, w, p=3)                         # 3 linear-ramp layers, 780 rzz per layer
res = fourcolors.quantum.compiler.compile(qc, target=targets.from_qiskit(FakeFez()), time_budget_s=60, seed=1,
                                          approximation_mode="exact", verify="wait")
print(res.metrics["two_qubit_gates"] / 3)             # two-qubit gates per layer after routing
```

A cost layer's gates all commute, so their order is the compiler's choice; on a fully connected graph that choice
is most of the circuit. Qiskit gets `transpile(optimization_level=3)` with new seeds for the same 60 s on FakeFez
with its published calibration as-is.

### Step 4: the whole run

```bash
python run.py --out results/ref                 # 51 instances x 3 time limits, then 10 circuits x 3 seeds x 60 s
python plot.py results/ref --figs figs
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

They check that the exhaustive Max-Cut equals brute force (weighted and unweighted, across chunks), that the QUBO
model's value is the cut on every assignment, the Q-score arithmetic, the G(n, 1/2) density, the LR-QAOA ramp and
gate count, the tables and the compile selection rule, and the plumbing of `run.py` and `plot.py` on fixed canned
results.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=referee-qpu-benchmark&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
