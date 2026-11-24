# Depth or gates? Choosing what the compiler minimizes

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-depth-or-gates&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-depth-or-gates&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime networkx matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

`run.py` and `plot.py` import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory. The full run
below bills 30 compiles at 30 s (15 compile-minutes); Step 1 bills 2 s.

### Step 1: compile one circuit both ways

```python
import fourcolors
import circuitkit as ck
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

qc = ck.draper_adder(16)                  # a 16-bit adder: 32 qubits
fez = targets.from_qiskit(FakeFez())      # the device's description and its published calibration
comp = fourcolors.quantum.compiler
gates = comp.compile(qc, target=fez, time_budget_s=1, seed=1, verify="skip", objective="two_qubit")
depth = comp.compile(qc, target=fez, time_budget_s=1, seed=1, verify="skip", objective="depth")
for r in (gates, depth):
    print(ck.circuit_metrics(r.to_qiskit()))   # two_qubit_gates, two_qubit_depth, depth, gates_on_failed_couplers
```

A target is a description of the device (couplers, native gates, calibration), never a name. One second is enough to
see the two answers differ; the post's runs use 30 s.

### Step 2: watch the trade-off as it is found

```python
job = comp.submit(qc, target=fez, time_budget_s=30, seed=1, verify="skip", objective="depth")
for res in job.incumbents():               # each better circuit as it is found, then the final result
    m = ck.circuit_metrics(res.to_qiskit())
    print(res.status, m["two_qubit_gates"], m["two_qubit_depth"])
```

Every incumbent is a complete, valid circuit for the chip. `job.best()` reads the best so far and `job.stop()` ends
the job and keeps it.

### Step 3: give Qiskit the same seconds, and pick fairly

The Qiskit arm is `transpile(qc, target=FakeFez().target, optimization_level=3, seed_transpiler=s)` on the device as
published (its calibration as-is, failed couplers included), repeated with new seeds until the same seconds are
spent. From that one series of seeds we keep two picks, one per objective:

| Qiskit's pick from its seeds | is set against | judged on |
|---|---|---|
| the fewest two-qubit gates | ours with `objective="two_qubit"` | two-qubit gates |
| the smallest two-qubit depth | ours with `objective="depth"` | two-qubit depth |

That is generous to Qiskit twice over: best of many seeds, and a separate best for each metric. It also avoids the
unfair comparison of one compiler's depth pick against the other's gate pick.

### Step 4: run everything and draw it

```bash
python run.py --out results/depth          # 3 circuits, seeds 1-5, 30 s, both objectives
python plot.py results/depth --figs figs
```

`plot.py` prints the table of medians, draws the figures, and writes `figs/selection.md`: which pairs the chart
shows, which it leaves out and why, and the line that goes under the chart.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check the circuit builders, that the gate and depth counter agrees with Qiskit's, that Qiskit's depth pick
is read from the published map's seeds with over-budget cells flagged, that each Qiskit pick is paired with our
matching objective, that the selection rule shows strict wins only, and the plumbing of `run.py` and `plot.py` on
fixed canned circuits.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-depth-or-gates&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
