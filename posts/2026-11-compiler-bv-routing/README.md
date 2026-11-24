# Trivial on paper, a routing nightmare on heavy-hex: Bernstein-Vazirani at 150 qubits

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-bv-routing&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-bv-routing&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime qiskit-aer networkx matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

`run.py` and `plot.py` import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory.

### Step 1: build the circuit and look at it

```python
import circuitkit as ck

secret = ck.random_secret(150, seed=150)   # the post's secret: random, fixed by n (about 75 ones)
qc = ck.bernstein_vazirani(secret)         # 151 qubits; one CNOT into qubit 150 per 1 bit
print(len(ck.interaction_edges(qc)))
```

The main pass uses a random secret, the usual BV benchmark. `--secret ones` runs the all-ones secret, the full star
and the worst case for routing; it is an optional second pass at the same cost.

### Step 2: compile it for the chip, with a time budget

```python
import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

fez = targets.from_qiskit(FakeFez())     # the device's description and its published calibration
res = fourcolors.quantum.compiler.compile(qc, target=fez, time_budget_s=10, seed=1,
                                          approximation_mode="exact", verify="skip")
out = res.to_qiskit()
print(ck.circuit_metrics(out))           # two_qubit_gates, two_qubit_depth, depth, gates_on_failed_couplers
print(res.engine["version"], res.billed_s, res.job.id)
```

A target is a description of the device (couplers, native gates, calibration), never a name. `time_budget_s` is
the one knob: the service returns the best circuit it found by then. The calibration marks the device's failed
couplers (error 1.0), so the compile avoids them.

The Qiskit arm is `transpile(qc, target=FakeFez().target, optimization_level=3, seed_transpiler=s)`, repeated with
new seeds until the same number of seconds is spent, keeping the best. That is generous to Qiskit: it gets
best-of-seeds, not one try.

### Step 3: check the answer

```python
import bv_check, blocks

print(bv_check.returns_secret(out, secret))   # {"ok": True, "method": "clifford_tableau" or "mps_64_shots", ...}
print(blocks.block_counts(out)["blocks"])     # {"cx": ..., "cx_move": ..., "swap": ..., ...}
```

`returns_secret` runs the compiled circuit from |0...0>, reads each measured qubit through the circuit's own measure
instructions (so routing and the final layout are covered), and checks that the bits are certain and spell the
secret. It uses an exact Clifford tableau when every single-qubit run is a Clifford. Some compilers (Qiskit at
level 3) re-synthesize two-qubit blocks into pieces that are Clifford only as a whole; those outputs go through
Aer's matrix-product-state simulator instead (no truncation, 64 shots; BV's ideal state is a product state, so this
stays cheap at 151 qubits). Each row records which method ran.

### Step 4: run the whole comparison and draw it

```bash
python run.py --out results/bv                  # n = 50..150, budgets 10 and 60 s, seeds 1..5, random secret
python plot.py results/bv --figs figs
```

`run.py` prints one line per 4colors compile, for example:

`plot.py` prints the table of medians over seeds, writes the figures, and writes `figs/selection.md`: which cells
the charts show, the line that goes under each chart, and the block table.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that BV returns its secret on a simulator, that the interaction graph is a star, that the gate
counter agrees with Qiskit's counts, that the 4colors target is the device with its published calibration, that the
Clifford check passes correct outputs (both methods) and fails a wrong secret or a non-Clifford circuit, that the
block classes are right on a CNOT, a SWAP and a CNOT-then-SWAP, that the selection rule shows only strict wins, and
check the plumbing of `run.py` and `plot.py` on fixed circuits.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-bv-routing&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
