# The coupler that wasn't there: what happens when your compiler isn't told

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-broken-couplers&utm_content=readme-top)

## The process, step by step

Setup:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-broken-couplers&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime networkx matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

### Step 1: find the dead couplers

```python
import circuitkit as ck
sets, failed = ck.heavy_hex_coupler_sets()   # {"working": Target, "full": Target}, set of pairs
print(sorted(failed))                          # the couplers FakeFez marks with error 1.0
```

A coupler counts as failed when its two-qubit error is at least 0.999 in either direction.

### Step 2: describe the device to the compiler

A 4colors target is always a description of the device. There are no named targets: you send the map you have, and
optionally the calibration you trust.

```python
from fourcolors.quantum import targets
fez = targets.from_qiskit(ck.fez())   # DeviceTarget: .description (the map) and .calibration
# the dead couplers are in the description, and at error 1.0 in the calibration
```

Passing a Qiskit backend as `target` does the same: the client reads the map and the calibration from it. To leave the
calibration out, send the description alone, `dict(fez.description)`. That is the "not told" arm below.

### Step 3: compile twice, with the same time budget

The compiler is time-first: you give it N seconds and get the best circuit it found by then.

```python
import fourcolors
comp = fourcolors.quantum.compiler
not_told = comp.compile(qc, target=dict(fez.description), time_budget_s=30, seed=1, verify="skip")
told = comp.compile(qc, target=fez, time_budget_s=30, seed=1, verify="skip")
for r in (not_told, told):
    print(ck.circuit_metrics(r.to_qiskit(), failed), ck.estimated_esp(r.to_qiskit(), sets["full"]))
```

When the compiler is told, the dead couplers are left out of that compile and listed in the result's `notes`, and
`calibration_used.sha256` ties the result to the snapshot it used: one line for the lab notebook.

### Step 4: run the three circuits and plot

```bash
python run.py --out results/broken
python plot.py results/broken --figs figs
```

The circuits:

- `chip_native`: one `rzz` on every FakeFez coupler, dead ones included. On the full map it needs no routing at all,
  but only if the dead couplers are used. It is the adversarial case.
- `qaoa_100`: QAOA (p = 1) on a random 3-regular graph with 100 nodes.
- `bv_100`: Bernstein-Vazirani with a 100-bit secret.

Each is compiled with seeds 1 to 5 at 30 s. Qiskit runs `transpile(optimization_level=3)` on the full published
FakeFez target, retrying seeds until the same 30 s are spent and keeping the best circuit.

### Step 5: check your own circuits

You do not need a 4colors account for this part. `check_circuit.py` counts two-qubit gates on dead couplers in any
circuit that is already laid out on a device:

```python
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeFez
from check_circuit import gates_on_dead_couplers

backend = FakeFez()                  # or your real backend: backend.target holds today's calibration
tqc = transpile(my_circuit, backend, optimization_level=3)
report = gates_on_dead_couplers(tqc, backend.target)
print(report["gates_on_dead"], "of", report["two_qubit_gates"], "two-qubit gates sit on dead couplers")
print(report["by_coupler"])          # {"a-b": count} for each dead coupler in use
```

Or from a QASM file: `python check_circuit.py my_circuit.qasm` (exit status 1 if any gate sits on a dead coupler).
Run it against the calibration of the day you submit, not the day you compiled.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that FakeFez's dead couplers are found and marked in the calibration the client sends, that
`chip_native` uses every coupler, that the estimate is 0 on a dead coupler, that the selection rule keeps only circuits
where told is ahead (ties and the not-told arm never count), that the check-your-circuits snippet counts correctly,
and check the scripts' plumbing on fixed circuits.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-broken-couplers&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
