# The biggest circuits: where the service says no, and why

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-biggest-circuits&utm_content=readme-top)

## The process, step by step

Setup:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-biggest-circuits&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime networkx matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

### Step 1: read your plan's limits

The limits that matter here (time budget, job memory, result size) belong to your key's plan. Read them instead of
guessing:

```python
import fourcolors
client = fourcolors.Client()             # key from FOURCOLORS_API_KEY
print(client.license()["limits"])         # GET /v1/license: the live limits of this key
```

The run script writes this object as its first row, and the post quotes the plan's limits from that row [OD-9: the
standard plan's time ceiling and floor; quote the row, not a range].

### Step 2: build a big input

```python
import examplekit, circuitkit as ck
edges = examplekit.gnp(150, 0.2, seed=150)             # about 2,200 edges
qc = ck.qaoa_maxcut(150, edges, p=2)                # one rzz per edge per layer: about 4,500
```

Six circuits on 150 qubits, from sparse and shallow to dense and deep: (density, layers) = (0.05, 1), (0.1, 2),
(0.2, 2), (0.2, 4), (0.4, 4), (0.5, 4), so about 560 to 22,000 input two-qubit gates. Gates on three or more qubits
would count as their decomposition; QAOA has none.

### Step 3: compile it, and read a refusal

A target is always a description of the device; there are no named targets. Here it is Qiskit's FakeFez snapshot
(156 qubits, heavy-hex) with its published calibration:

```python
from fourcolors.errors import InvalidRequest, JobFailed
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

fez = targets.from_qiskit(FakeFez())          # description + published calibration
try:
    res = fourcolors.quantum.compiler.compile(qc, target=fez, time_budget_s=60, seed=1, verify="skip")
    print(ck.circuit_metrics(res.to_qiskit()), res.billed_s)
except (JobFailed, InvalidRequest) as err:
    limit = (getattr(getattr(err, "job", None), "result", None) or {}).get("limit")
    print("refused:", err.code, err.field, err.message, limit)
```

`InvalidRequest` is raised at submit; `JobFailed` if the job is accepted and then refused. Both carry `code`,
`field` and `message`; a refused job's result also has `limit`. The script records every refusal as a row and does
not resend the larger budgets of a circuit refused for its size.

### Step 4: run the probe, then the sweep

The probe (six jobs at 10 s, refusals free) says which circuits the service takes. The sweep compiles each accepted
circuit at each budget the plan allows. Qiskit runs `transpile(optimization_level=3)` on FakeFez as published,
retrying seeds until the same budget is spent and keeping the best circuit.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that the six configs span about 500 to over 20,000 input gates, that a size refusal stops the larger
budgets of that circuit while other refusals do not, that budgets above the plan limit are never sent, that the
limits row carries only the `limits` object, and that the selection rule keeps only cells where ours is lower
(ties, losses, refusals and Qiskit over-budget cells are left out with a reason).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-biggest-circuits&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
