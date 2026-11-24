# Parametric circuits at 150 qubits: compile once, and check it at full size

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=parametric-large&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=parametric-large&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime qiskit-qasm3-import networkx matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # the key is read from the file, in the process
```

`run.py` and `plot.py` import `../../shared/examplekit.py`, the shared QAOA code in `../../qaoa-blog/qaoakit/` (ansatze, both
compilers at a matched budget) and this post's `clifford_check.py`; run them from this directory.

### Step 1: the ansatze

| Spec | Qubits | Parameters | What it is |
|---|---|---|---|
| `ma-100`, `ma-150` | 100, 150 | 250, 375 | multi-angle QAOA, p = 1, on a random 3-regular graph: one angle per edge and per node |
| `hea-ring-100x2`, `hea-ring-150x2` | 100, 150 | 400, 600 | hardware-efficient, 2 layers of ry and rz, CZ on a ring |
| `qaoa-150-p2` | 150 | 4 | standard QAOA, p = 2: few parameters, many gates |

```python
from qaoakit import ansatz as az

ansatz, graph = az.build("ma-150")       # 150 qubits, 375 Parameter objects, no measurements
```

The QAOA graphs are not native to the chip, so they need routing. The ring is routing-hard on purpose: a 150-qubit
ring does not sit on heavy-hex without moving qubits.

### Step 2: compile once, with a time budget

```python
from fourcolors.quantum import compiler, targets
from qiskit_ibm_runtime.fake_provider import FakeFez

fez = targets.from_qiskit(FakeFez())     # the device as a description: couplers, native gates, calibration
res = compiler.compile(ansatz, target=fez, time_budget_s=60, seed=1, approximation_mode="exact")
out = res.to_qiskit()

assert list(out.parameters) == list(ansatz.parameters)   # your own Parameter objects, all of them
print(len(res.parameters), res.billed_s, res.job.id)     # the names to bind, billed seconds, job id
```

A target is a description of the device, never a name. `time_budget_s` is the one knob: the service returns the
best circuit it found by then. `approximation_mode="exact"` asks for an exact circuit, the same thing Qiskit
returns, so the two are compared like for like. (The post's runs pass the device's 169 working couplers as the
description, the same map the compile-once post uses; Qiskit gets FakeFez's published target as shipped.)

The Qiskit arm is `transpile(ansatz, target=FakeFez().target, optimization_level=3, seed_transpiler=s)`, repeated
with new seeds until the same number of seconds is spent, keeping the fewest two-qubit gates. That is generous to
Qiskit: it gets the best of many tries, not one.

### Step 3: check it at full size

```python
import clifford_check as cc

print(cc.check_many(ansatz, out, draws=3, seed=1))
# {'clifford_ok': True, 'clifford_method': 'stabilizer', 'clifford_draws': 3, 'clifford_mismatches': 0, ...}
```

Under the hood, for each draw:

```python
values = cc.clifford_values(len(ansatz.parameters), seed)          # each a multiple of pi/2
cc.stabilizer_match(ansatz, out, values)                           # {'ok': True, 'method': 'stabilizer', ...}
```

The service also runs its own check after each compile and labels the result. For circuits this large the label
can come back `unverified`, meaning the service's exact check does not cover that size. The full-size Clifford check
is this post's evidence; the label is reported beside it as returned.

### Step 4: bind

```python
bound = out.assign_parameters(dict(zip(ansatz.parameters, values)))     # by your own Parameter objects
bound = out.assign_parameters(values)                                   # or by position, in ansatz order
```

Binding changes angles only, so every binding has the same two-qubit gate count. `run.py` binds each compiled
circuit 200 times by object and once by position, and checks that the two binds give the same circuit, gate by gate.

### Step 5: run the comparison and draw it

```bash
python run.py --out runs/large                  # 5 ansatze, budgets 10 and 60 s, seeds 1-5, both compilers
python plot.py runs/large --figs figs           # gates.png, bind.png, table.md, selection.md
```

(ansatz, budget, seed, arm, two-qubit gates, refusal code, parameters kept, full-size check). `results.jsonl` has
one `compile` row per ansatz, budget, seed and arm: two-qubit gates, two-qubit depth, depth, the parameter checks,
the Clifford check (`clifford_ok`, `clifford_method`, `clifford_mismatches`), binding times (local, untimed), job id,
billed seconds, verification label and refusal code. A refused compile is a row with its code, never dropped.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that every default ansatz binds to a Clifford circuit at full size, that the check follows routing
(a circuit Qiskit routed onto heavy-hex passes) and catches a wrong binding and a dropped gate, that Qiskit O3's
re-synthesized blocks pass through the block merge, that a non-Clifford circuit gets no claim, that the selection
rule shows only strict wins where both arms passed the check (ties, losses, refusals and unchecked cells left out),
and check the plumbing of `run.py` and `plot.py` on fixed circuits.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=parametric-large&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
