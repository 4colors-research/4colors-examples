# Which circuits gain most: a compiler report card by family

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-report-card&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key: start a free trial at https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-report-card&utm_content=readme-text, then create a key at account.4colors-research.com
   (docs: docs.4colors-research.com).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime networkx matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

The scripts import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory.

### Step 1: generate a family and pin it

```python
import families as F

qc, meta = F.make("tfim2d", 6, 101)      # open 6 x 6 grid, 2 Trotter steps, 36 qubits, measured
print(qc.num_qubits, meta["boundary"])   # 36 open
print(F.instance_sha256(qc))             # the sha256 of the exact OpenQASM 3 text you will compile
```

the generator version, the commit of `families.py` and the sha256 of every circuit, so a later change to a generator
can never silently change what a number means. Two families were fixed before this run: UCCSD now starts from the
Hartree-Fock reference, and the Hubbard models from a half-filled product state, so that neither acts trivially on
|0...0>.

### Step 2: compile it for the chip with a time budget

```python
import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

fez = targets.from_qiskit(FakeFez())      # the device's description and its published calibration
res = fourcolors.quantum.compiler.compile(qc, target=fez, time_budget_s=10, seed=1,
                                          approximation_mode="exact", verify="wait")
print(res.metrics["two_qubit_gates"], res.verification.label, res.verification.reason)
```

`time_budget_s` is the one knob: the service returns the best circuit it found in that time. `verify="wait"` holds
the result until the independent equivalence check has given its label. Above 10 active qubits most compiles come
back `unverified` (not proved, not wrong); the post reports the label of every quoted circuit.

### Step 3: give Qiskit the same seconds

```python
from qiskit import transpile

import time

target = FakeFez().target                                   # published calibration as-is
best, t0, k = None, time.perf_counter(), 0
while time.perf_counter() - t0 < 10:                        # new seeds until the 10 s are spent
    out = transpile(qc, target=target, optimization_level=3, seed_transpiler=1000 + k)
    n2 = sum(1 for i in out.data if i.operation.num_qubits == 2 and i.operation.name != "barrier")
    best, k = min(best or n2, n2), k + 1
```

`run.py` does this on as many parallel streams as the plan has compile threads (`--qiskit-streams`), keeps the best
trial within the budget (arm `qiskit-o3-best`), and also keeps the very first trial, one default call (arm
`qiskit-o3-once`). The bar 4colors must clear is the lower of the two.

### Step 4: check the approximation is the same on both sides

Qiskit's `transpile` is exact by default; `dq.py` measures it anyway, on random product input states, wherever the
compiled circuit touches at most 20 qubits. If Qiskit approximated, 4colors gets the same allowance
(`max_approximation_error=d_q`); otherwise both are exact. Circuits too wide for the check stay exact on our side,
which can only cost us gates.

### Step 5: run the card and draw it

```bash
python run.py --out results/card                 # 29 families x 3 sizes x 2 instances x 3 seeds, 10 s each
python plot.py results/card --figs figs           # figs/card.md, figs/selection.md and three figures
```

To grade your own circuits, add a builder to `families.py` (or pass your circuits to the same two arms) and rerun
only that family: `python run.py --out results/mine --families myfamily`.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

They check that all 29 families build at all sizes and fit the chip, that the generator is deterministic and the
sha256 pins the text, that UCCSD and the Hubbard models now act on |0...0>, that the grids are open, that the D2
pre-pass returns 0 on Qiskit's exact output and catches a wrong circuit, that circuits with resets stay unmeasured,
that the selection rule shows only strict wins against the lower Qiskit arm, and the plumbing of `run.py` and
`plot.py` on a fixed canned result.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-report-card&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
