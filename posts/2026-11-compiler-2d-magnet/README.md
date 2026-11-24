# A 2D magnet on a heavy-hex chip: fewer two-qubit gates per Trotter step

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-2d-magnet&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key: start a free trial at https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-2d-magnet&utm_content=readme-text, then create a key at account.4colors-research.com
   (docs: docs.4colors-research.com).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime networkx matplotlib pyyaml pytest scipy
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

Run the scripts from this directory (they import `../../shared/examplekit.py` and `../../shared/circuitkit.py`).

### Step 1: build one Trotter step and look at its structure

```python
import tfim

qc = tfim.circuit(7, 8, steps=1)               # open 7 x 8 grid: 56 qubits, 97 bonds, measured
print(len(tfim.bonds(7, 8)), tfim.logical_floor(7, 8, 1))        # 97 bonds, 194 two-qubit gates with no routing
print({k: len(v) for k, v in tfim.matchings(7, 8).items()})       # the 4 layers
print(max(tfim.degrees(7, 8).values()))                            # 4: more than any heavy-hex qubit has
```

The boundaries are **open**: the lattice ends at its edges. A periodic lattice (wrapping around, as many physics
papers use) has more bonds and long-range ones on a planar chip, and is harder; this post does not claim anything
about it.

### Step 2: compile it with a time budget

```python
import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

fez = targets.from_qiskit(FakeFez())          # the device's description and its published calibration
res = fourcolors.quantum.compiler.compile(qc, target=fez, time_budget_s=10, seed=1,
                                          approximation_mode="exact", verify="wait")
out = res.to_qiskit()
print(res.metrics["two_qubit_gates"], res.verification.label)
```

### Step 3: the same seconds for Qiskit, and the whole comparison

```bash
python run.py --out results/magnet      # grids 4x4, 6x6, 7x8, 8x8; 1 and 2 steps; seeds 1-3; 10 s
python plot.py results/magnet --figs figs
```

Qiskit gets `transpile(optimization_level=3)` with new seeds until the same 10 s are spent (on as many streams as
the plan has compile threads), on FakeFez with its published calibration as-is. The bar 4colors must clear is the
lower of Qiskit's best trial and its first (one default call). Both compiles are exact; Qiskit's default
`approximation_degree` 1.0 is exact on this circuit.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

They check the open grid's bonds, that the four matchings partition them with no qubit used twice in one layer,
the degree, that one step equals the Trotter product exactly (operator check on a 2 x 2 grid), that the circuit is
the report card's `tfim2d` gate for gate, the selection rule, and the plumbing of `run.py` and `plot.py` on a fixed
canned result.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-2d-magnet&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
