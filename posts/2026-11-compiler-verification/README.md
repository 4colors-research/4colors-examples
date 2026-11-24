# Verified, approximate, unverified: reading a compiler's receipt

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=compiler-verification&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-verification&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime networkx matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

`run.py` and `plot.py` import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory. Each step below
bills its budget (10 s); the check itself is not charged, but `verify="wait"` returns only once it has finished.

### Step 1: a circuit small enough to prove

A 6-qubit quantum Fourier transform (QFT) on a 2 x 3 grid chip, with the result returned only when its label is
decided:

```python
import fourcolors
import circuitkit as ck

chip = ck.custom_target(6, [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]])
# {"format_version": 2, "n_qubits": 6, "two_qubit_gate": "cz", "directed": False, "couplers": [[0, 1], ...]}
qc = ck.qft(6)
res = fourcolors.quantum.compiler.compile(qc, target=chip, time_budget_s=10, seed=1, verify="wait",
                                          approximation_mode="exact")
v = res.verification
print(v.label, v.method, v.distance, v.reason)
```

A custom chip is a description: qubits, couplers and the native two-qubit gate.

### Step 2: allow a small approximation, and read the distance

The QFT is full of controlled phases with tiny angles, which is where an approximation can save gates. Allow each
two-qubit block a small error, on a 156-qubit heavy-hex device with its published calibration:

```python
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

fez = targets.from_qiskit(FakeFez())     # the device's description and its published calibration
res = fourcolors.quantum.compiler.compile(ck.qft(20), target=fez, time_budget_s=10, seed=1, verify="wait",
                                          approximation_mode="per_block")
print(res.verification.label, res.verification.distance,
      res.to_qiskit().metadata["fourcolors"]["approximation_spent"])
```

`per_block` is approximate on purpose: it may drop gates that are close to the identity. If the label comes back
`approximate`, the distance is what the check measured, within the bound; if the exact answer won, it can come back
`verified`.

### Step 3: a circuit past what the check can decide

A 100-qubit QAOA circuit (p = 1 on a random 3-regular graph), exact as input, on the same device:

```python
import examplekit
big = ck.qaoa_maxcut(100, examplekit.random_regular(100, 3, seed=100), p=1)
res = fourcolors.quantum.compiler.compile(big, target=fez, time_budget_s=10, seed=1, verify="wait",
                                          approximation_mode="exact")
print(res.verification.label, res.verification.reason)
```

If the check cannot decide in its time, the label is `unverified` and `reason` says why. That is a statement about
the check, not about the circuit. A checker for larger circuits is in development.

### Step 4: the default flow, in the background

`verify` defaults to `"background"`: the result returns at once with the label `pending`, and you read the decided
label later.

```python
res = fourcolors.quantum.compiler.compile(ck.qft(20), target=fez, time_budget_s=10, seed=1)
print(res.verification.label)                       # pending
v = res.wait_verification(timeout=900)              # or res.refresh() whenever you look again
print(v.label, v.distance, v.reason)
```

### Step 5: check a small one yourself

For a few qubits you need no one's word: compare the compiled circuit with the input in Qiskit, on the input's qubits
with every other chip qubit starting in |0>:

```python
import run
small = fourcolors.quantum.compiler.compile(qc, target=chip, time_budget_s=10, seed=1, verify="skip")
print(run.local_check(small.to_qiskit(), qc))       # (equivalent up to a global phase?, 1 - process fidelity)
```

`Operator.from_circuit` follows the compiled circuit's layout, so routing is covered. The matrix has 4^qubits
entries, so this is for small circuits only.

### Step 6: run everything and print the table

```bash
python run.py --out results/verify        # QFT 6 and 20, QAOA 100; exact and per_block; seeds 1-3; 10 s
python plot.py results/verify --figs figs
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that the QFT builder matches Qiskit's QFT, that the grid chip is connected, that the local check
passes a correct compiled circuit and fails a perturbed one, that no QFT above 32 qubits is built, that Qiskit's
counts are kept only where its distance was measured, and the plumbing of `run.py` and `plot.py` on fixed canned
circuits.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-verification&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
