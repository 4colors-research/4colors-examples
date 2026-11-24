# Give it N seconds: a compiler you can watch improve

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-give-it-seconds&utm_content=readme-top)

## What you will learn

- `fourcolors.quantum.compiler.submit(...)` returns a running job; `for res in job.incumbents():` yields each better
  circuit as it is found (status `running`), then the final result. `job.best()` reads the best so far, `job.stop()`
  ends the job and keeps it.
- How the two-qubit gate count moves with `time_budget_s` from 1 to 60 seconds.
- A fair use of time on the Qiskit side: level 3 with new random seeds until the same seconds are spent, keeping the
  best.

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-give-it-seconds&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime networkx matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

`run.py` and `plot.py` import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory. Step 2 bills
up to 60 compile-seconds; the full sweep in Step 3 bills about 39 compile-minutes.

### Step 1: build the circuit

QAOA with 3 layers on a Barabasi-Albert graph, in the style of the Benchpress QAOA benchmark [CHECK: the
Barabasi-Albert parameter must be checked against the Benchpress source before the post names it "Benchpress"]:

```python
import circuitkit as ck

qc = ck.qaoa_maxcut(27, ck.barabasi_albert(27, 3, seed=27), p=3)   # 27 qubits, one rzz per edge per layer
print(ck.circuit_metrics(qc)["two_qubit_gates"])
```

Barabasi-Albert graphs have a few highly connected hubs, so the interaction graph is far from the chip's
three-neighbor lattice: there is a lot of routing to do, and a lot to gain from doing it well.

### Step 2: watch one compile improve

```python
import time
import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

fez = targets.from_qiskit(FakeFez())     # the device's description and its published calibration
t0 = time.monotonic()
job = fourcolors.quantum.compiler.submit(qc, target=fez, time_budget_s=60, seed=1, verify="skip",
                                         approximation_mode="exact")
for res in job.incumbents():
    print(f"{time.monotonic() - t0:5.1f}s {res.status:8s} {ck.circuit_metrics(res.to_qiskit())['two_qubit_gates']}")
```

The time printed is when the circuit reached your machine (it includes queueing and network time). Every line is a
complete circuit you could run. If the count stops moving, `job.stop()` keeps the best so far and ends the billing
there.

### Step 3: sweep the budget and plot

```bash
python run.py --out results/seconds        # sizes 15, 21, 27, 50; budgets 1, 3, 10, 30, 60 s; seeds 1-5
python plot.py results/seconds --figs figs
```

For every size, seed and budget, `run.py` compiles once with that budget, and runs Qiskit O3 on the same device
with new seeds (`seed_transpiler` = 1000 x seed + k) until the largest budget is spent, recording when each seed
finished; Qiskit's best within each budget is read from that one series. One stream per size at 60 s records the
incumbents. `plot.py` prints the medians, draws the figures and writes `figs/selection.md`.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check the Barabasi-Albert graph and the QAOA circuit (one rzz per edge per layer, and the state it
prepares against a direct numpy construction), that Qiskit's best within a budget is read correctly with over-budget
cells flagged, that the selection rule shows strict wins only and computes the margin, that Qiskit runs only on the
published map, and the plumbing of `run.py` and `plot.py` on fixed canned circuits.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-give-it-seconds&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
