# Aircraft loading, quantum and classical

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=aircraft-loading&utm_content=readme-top)

## The model: one decision per container and position, five kinds of row

`x[i, j] = 1` when container i goes to position j (only pairs where the container fits under the position's
structural limit are variables). Then, exactly:

| Rule | Row |
|---|---|
| a container flies at most once | sum_j x[i, j] <= 1 for each container |
| a position holds at most one container | sum_i x[i, j] <= 1 for each position |
| payload | sum w_i x[i, j] <= W |
| center of gravity at or behind the forward limit | sum w_i (d_j - lo) x[i, j] >= 0 |
| center of gravity at or ahead of the aft limit | sum w_i (d_j - hi) x[i, j] <= 0 |

and maximize the loaded weight. The two balance rows are the envelope lo <= (sum w d x) / (sum w x) <= hi,
multiplied out so they are linear. With exact rows there are no penalty weights to tune and no slack variables.

```python
import loading

h = loading.paper_scale(28, seed=1)           # a hold whose model has 28 variables, as in the largest study size
m = loading.model(h)                           # objective + exact rows
print(h.n, m.m, len(h.weights), len(h.arms))   # variables, rows, containers, positions
```

## Step by step

### Setup

1. Get an API key: start a free trial at https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=aircraft-loading&utm_content=readme-text, then create a key at account.4colors-research.com
   (docs: docs.4colors-research.com).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime scipy networkx matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

Run the scripts from this directory (they import `../../shared/examplekit.py` and `../../shared/circuitkit.py`).

### Step 1: solve it classically, and check the plan

```python
import fourcolors.qubo

r = fourcolors.qubo.solve(m, time_limit_s=1, seed=1)
print(loading.check(h, r.solution))   # {"valid": True, "load_kg": ..., "cg": ..., "containers_loaded": ...}
print(loading.milp_reference(h)["reference_value"])   # HiGHS on the same rows: the proven optimum
```

`check` knows nothing about the model: it confirms that no container is placed twice, no position holds two,
every structural limit holds, the payload limit holds and the center of gravity is inside the envelope. Every
constrained answer is also checked by the service against the rows.

### Step 2: the quantum formulation, compiled

A QAOA circuit needs the problem without rows: a penalty QUBO. Squaring the weight and balance rows (the slack-free
shape) makes nearly every pair of variables interact, so the circuit's cost layer is close to all-to-all:

```python
qc = loading.qaoa_circuit(h, p=1)            # H, rzz on every interacting pair, rz, rx mixer; fixed angles
print(qc.num_qubits, len(loading.penalty_qubo(h)[0]))

import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

res = fourcolors.quantum.compiler.compile(qc, target=targets.from_qiskit(FakeFez()), time_budget_s=10, seed=1,
                                          approximation_mode="exact", verify="wait")
print(res.metrics["two_qubit_gates"], res.verification.label)
```

On a trapped-ion machine, where any two qubits can interact, a dense cost layer needs no routing. On a heavy-hex
chip, where each qubit has at most three neighbors, most of the compiled circuit is the routing of that dense
layer. The two-qubit count after compilation is the price of the quantum formulation on that chip.

### Step 3: the whole run

```bash
python run.py --out results/aero              # both halves
python plot.py results/aero --figs figs
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

They check the studied sizes, that the rows admit exactly the valid plans (every assignment of a small hold), that
the checker catches each rule, that the HiGHS reference equals the brute-force optimum, that the QAOA circuit has one
two-qubit gate per interacting pair per layer, the selection rules, and the plumbing of `run.py` and `plot.py` on
fixed canned results.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=aircraft-loading&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
