# One problem, three encodings: what a constraint costs in gates

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qaoa-encodings&utm_content=readme-top), [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=qaoa-encodings&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qaoa-encodings&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-aer qiskit-ibm-runtime qiskit-qasm3-import networkx scipy matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # the key is read from the file, in the process
```

`run.py` and `plot.py` import `../../shared/examplekit.py` and the shared QAOA code in `../../qaoa-blog/qaoakit/` (graphs,
the angle tuner, both compilers at a matched budget, the noisy simulator); run them from this directory.

### Step 1: build the three circuits

```python
import run

inst = dict(id="demo", kind="max3cut", nodes=5, edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 0, 1.0)])
for enc in run.ENCODINGS:                       # onehot-penalty, onehot-xy, binary
    qc, theta = run.circuit(inst, enc, p=1)     # parametric: 2 angles per layer
    print(enc, qc.num_qubits)                   # 15, 15, 10
```

The binary encoding's phase layer is built term by term: each edge's cut indicator over its 4 bits is written as a
sum of Z strings (a Walsh-Hadamard transform), and each Z string is a CNOT ladder around one rz. That is where its
extra two-qubit gates come from: it saves a third of the qubits and pays in gates.

### Step 2: tune the angles, noiselessly

```python
from qaoakit import apikit as ak, core

opt = ak.max3cut_optimum(inst["nodes"], inst["edges"])    # best cut, by trying every coloring
cost, feasible = run.scores(inst, "binary")                # score of every basis state (small graphs only)
angles, value, evals = core.tune_circuit(qc, theta, cost, restarts=4, seed=7)
print(value / opt, run.start_ratio("binary", cost, feasible, opt))   # after QAOA, and the start state's own ratio
```

`max3cut_optimum` tries all 3^n colorings, which is fine up to about 13 nodes. The p = 2 angles
start from the p = 1 answer, interpolated.

### Step 3: compile with a time budget

```python
from fourcolors.quantum import compiler, targets
from qiskit_ibm_runtime.fake_provider import FakeFez
from qaoakit import ansatz as az

circ = az.to_input(qc)
circ.measure_all()
res = compiler.compile(circ, target=targets.from_qiskit(FakeFez()), time_budget_s=5, seed=1,
                       approximation_mode="exact")
out = res.to_qiskit()
bound = out.assign_parameters(dict(zip(theta, angles)))
```

A target is a description of the device (couplers, native gates, calibration), never a name. `time_budget_s` is
the one knob. The post's runs pass the device's 169 working couplers as the description. The Qiskit arm is
`transpile(..., optimization_level=3)` on FakeFez's published target, repeated with new seeds until the same 5
seconds are spent, best kept.

Before scoring, `run.py` checks each compiled circuit: bound to the tuned angles and simulated without noise, it
must give the tuned circuit's expected score (`check_ok`). Both compilers' outputs get the same check.

### Step 4: score under noise

```python
st, probs = core.noisy_ratio(bound, qc.num_qubits, cost, opt, e2=3e-3, e_ro=0.01, trajectories=100, batches=5)
```

The noise model is a model: depolarizing error e2 on every CZ, e2/10 on SX and X, 1% readout error, no idle error
and no qubit-to-qubit spread. It rewards fewer two-qubit gates and little else. It is not a device.

### Step 5: a larger graph, gate counts only

Small graphs keep the simulation exact, but they flatter the binary encoding's 4-body terms and the XY mixer's
W-state preparation alike. `run.py` also compiles all three encodings on random 3-regular graphs with 12 and 24
nodes (up to 72 qubits) for gate counts only: no angles and no simulation, just what each encoding costs the chip
as the graph grows.

### Step 6: the same problem for the optimizer

```python
import fourcolors.qubo

m = ak.max3cut_model(inst["nodes"], inst["edges"])        # one-hot variables, the cut as the objective
# inside: m.add_constraint({3*v: 1, 3*v + 1: 1, 3*v + 2: 1}, lo=1, hi=1) for every node
r = fourcolors.qubo.solve(m, time_limit_s=1, seed=1)
print(r.objective, r.constraints)     # the service's exact check of every row
```

The optimizer gets the one-hot model two ways: with the rows as a penalty in the objective, and with exact rows.
The binary encoding is not a QUBO without extra variables (its cost has 3- and 4-variable terms), so it is recorded
as such. Besides the small graphs, the optimizer gets a 200-node 3-regular graph (600 variables, 200 rows). Its best
cut needs no search to certify: a connected graph where no node has more than three neighbors, other than the
complete graph on four nodes, can always be colored with three colors (Brooks' theorem), so the optimum cuts every
edge.

### Run it

```bash
python run.py --out runs/enc                     # small graphs, both compilers, larger graphs, the optimizer
python plot.py --out runs/enc --figs figs        # encodings.png, compiler.png, table.md, selection.md
```

`results.jsonl` rows: `qaoa` (encoding, p, arm, seed: qubits, two-qubit gates, start-state, ideal and noisy ratio,
`check_ok`, `noisy_feasible`, job id, billed seconds, verification label, refusal code), `count` (the larger graphs:
encoding, qubits, compiled two-qubit gates) and `optimizer` (model, variables, rows, ratio, feasibility, job id,
billed seconds). A refused compile or solve is a row with its code, never dropped.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that the penalty and binary phase layers are exactly the intended diagonal unitaries, that the
penalty QUBO, the model with rows and enumeration agree on the optimum, that the XY mixer keeps every node one-hot
and the X mixer does not, that the binary code gives color 2 half the time and each start-state ratio is right,
that the Brooks certificate matches enumeration and refuses K4, that the selection rule shows only cells ahead on
gates and not behind on noise, and check the plumbing of `run.py` and `plot.py` on fixed circuits.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qaoa-encodings&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
