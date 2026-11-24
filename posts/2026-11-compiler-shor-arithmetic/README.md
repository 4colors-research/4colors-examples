# The arithmetic inside Shor's algorithm, compiled: 100-qubit adders on a heavy-hex chip

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=compiler-shor-arithmetic&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key: start a free trial at https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-shor-arithmetic&utm_content=readme-text, then create a key at account.4colors-research.com
   (docs: docs.4colors-research.com).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime qiskit-aer networkx matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

Run the scripts from this directory (they import `../../shared/examplekit.py` and `../../shared/circuitkit.py`).

### Step 1: build a 100-qubit adder

```python
import adders

qc = adders.circuit("cuccaro", 49)        # cin + a (49) + b (49) + cout = 100 qubits, Toffolis kept as ccx
print(qc.num_qubits, sum(1 for i in qc.data if i.operation.name == "ccx"))
```

The block has no measurements: it is the arithmetic a larger algorithm would call.

### Step 2: compile it with a time budget

```python
import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

fez = targets.from_qiskit(FakeFez())     # the device's description and its published calibration
res = fourcolors.quantum.compiler.compile(qc, target=fez, time_budget_s=10, seed=1,
                                          approximation_mode="exact", verify="wait")
out = res.to_qiskit()                    # its layout says where every input bit starts and ends
print(res.metrics["two_qubit_gates"], res.verification.label)
```

### Step 3: check the adder adds

```python
print(adders.check(out, "cuccaro", 49, trials=8))   # {"adder_check": "returns_sum", ...}
```

`check` draws random a, b and carry-in, sets them with X gates on the physical qubits the layout gives, runs the
compiled circuit on Aer's matrix-product-state simulator and compares every shot with a + b. A basis input stays
close to a basis state throughout, so this is cheap at 100 qubits. It is evidence on the inputs tried, not a proof
for every input; the service's own label is reported beside it (above 10 active qubits most compiles are
`unverified`: not proved, not wrong).

### Step 4: the whole comparison

```bash
python run.py --out results/adders       # Cuccaro and VBE, 3 sizes each up to 100 qubits, seeds 1-3, 10 s
                                         # (and 60 s for the 100-qubit adders)
python plot.py results/adders --figs figs
```

Qiskit gets `transpile(optimization_level=3)` on FakeFez with its published calibration as-is, new seeds until the
same budget is spent (on as many streams as the plan has compile threads); the bar is the lower of its best trial
and its first. Qiskit's picks are rebuilt from their seeds and checked the same way.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

They check that both adders add every 3-bit input exactly (statevector), that the largest sizes are 100 qubits,
that the sum check passes Qiskit's compiled adder and catches a wrong one, the selection rule (a wrong sum is never
shown), and the plumbing of `run.py` and `plot.py` on a fixed canned result.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-shor-arithmetic&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
