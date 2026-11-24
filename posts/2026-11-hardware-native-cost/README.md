# What "hardware-native" costs you

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=hardware-native-cost&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=hardware-native-cost&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime networkx numpy matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

`run.py` and `plot.py` import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory.

### Step 1: reshape the problem to the chip

```python
import circuitkit as ck
import native as nv

fez = ck.heavy_hex_coupler_sets()[0]["full"]          # FakeFez as shipped, published calibration
n_qubits, couplers = nv.device_graph(fez)             # working couplers and their errors
edges, w = nv.problem("regular3", 150, seed=150)
place = nv.native_placement(150, edges, w, n_qubits, couplers, seed=150)
keep = nv.native_edges(place, edges, couplers)
print(len(keep), "of", len(edges), "edges land on couplers")
```

### Step 2: solve every version, score it on the original graph

```python
import fourcolors.qubo

for name, es in {"full": edges, "native": keep,
                 "random": nv.random_edges(edges, len(keep), 150),
                 "spectral": nv.spectral_edges(150, edges, w, len(keep), 150)}.items():
    r = fourcolors.qubo.solve(nv.qubo_model(150, es, w), time_limit_s=5, seed=1)
    print(name, nv.cut(edges, w, r.solution))           # the cut on the ORIGINAL graph
```

The share retained is that cut divided by the best cut the optimizer found on the full graph (over the seeds). On
the 20-node graph the full answer is also checked against the exact maximum cut by enumeration.

### Step 3: compile the circuits

```python
import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

target = targets.from_qiskit(FakeFez())               # description + published calibration
full_qc = nv.qaoa_circuit(150, edges, w, p=1)
native_qc = nv.qaoa_circuit(156, keep, w, p=1, qubit_of=place, nodes=range(150))  # already on device qubits
for qc in (full_qc, native_qc):
    res = fourcolors.quantum.compiler.compile(qc, target=target, time_budget_s=30, seed=1,
                                              approximation_mode="exact", verify="skip")
    print(ck.circuit_metrics(res.to_qiskit())["two_qubit_gates"])
```

The native circuit's rzz gates already sit on couplers (`nv.on_couplers` checks it), so its compiled two-qubit
count is close to two per kept edge (each rzz is two CZ on a CZ chip). The full circuit has to be routed.

### Step 4: the whole set

```bash
python run.py --out runs/native            # 7 graphs: solves (4 variants x 3 seeds) and compiles
python plot.py runs/native --figs figs
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check, on hand-made graphs and a hand-made ring "device", that the QUBO energy is minus the cut, that a
ring problem embeds fully on a ring, that a triangle keeps two edges and avoids a noisy coupler when the placement
is calibration-aware, the placement score, that random and spectral sparsifiers keep k distinct edges, effective
resistances on a path and a triangle, that the native circuit sits on couplers, that FakeFez's failed couplers are
dropped from the device graph, the selection rule, and the plumbing of `run.py` and `plot.py` with canned answers.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=hardware-native-cost&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
