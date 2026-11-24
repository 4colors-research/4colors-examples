# Is this instance worth QPU time? A two-call pre-flight check

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=worth-qpu-time&utm_content=readme-top), [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=worth-qpu-time&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=worth-qpu-time&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime networkx matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

The scripts import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory.

### Step 1: call 1, the optimizer

```python
import fourcolors.qubo
import examplekit, triage

inst = next(i for i in triage.instances() if i["id"] == "reg3_n96")
m = triage.qubo_model(inst["n"], inst["edges"], inst["weights"])    # Max-Cut, E(x) = -cut(x)
r = fourcolors.qubo.solve(m, time_limit_s=10, seed=1)
f = examplekit.result_fields(r)
sig = triage.classical_signal(f["trajectory"], f["solver_objective"], budget_s=10)
print(triage.cut(inst["edges"], inst["weights"], r.solution), sig["t_final_s"])
```

### Step 2: call 2, the compiler

```python
import circuitkit as ck
import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

qc = triage.qaoa_circuit(inst["n"], inst["edges"], inst["weights"], p=1)
fez = targets.from_qiskit(FakeFez())      # the device's description and its published calibration
res = fourcolors.quantum.compiler.compile(qc, target=fez, time_budget_s=10, seed=1,
                                          approximation_mode="exact", verify="skip")
esp = ck.estimated_esp(res.to_qiskit(), FakeFez().target)    # a model from the published calibration
print(ck.circuit_metrics(res.to_qiskit())["two_qubit_gates"], esp)
```

The circuit is QAOA at p = 1 with fixed angles; the check compiles it, it does not run it or optimize angles. A
deeper p multiplies the two-qubit gates and divides the ESP accordingly.

### Step 3: the verdict, and the whole set

```python
print(triage.verdict(sig["t_final_s"], esp))     # one of the four boxes
```

```bash
python check.py --out runs/triage                # 30 instances: 30 solves x 10 s + 30 compiles x 10 s
python plot.py runs/triage --figs figs
```

`check.py` also runs Qiskit O3 on the same circuit for the same seconds (seeds until the budget is spent, best
kept, the published calibration as-is), so you can see how much of the ESP is the compiler's doing.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that the 30 instances are seeded, unique and fit one device, that the family shapes are right, that
the QUBO energy is minus the cut on every assignment, that the QAOA circuit has one rzz per edge per layer, the
trajectory signal on hand-made trajectories, the four boxes and the line crossing, that the selection draws Qiskit
only where 4colors is strictly higher, and the plumbing of `check.py` and `plot.py` with canned answers.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=worth-qpu-time&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
