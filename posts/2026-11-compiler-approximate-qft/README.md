# Approximate QFT, measured like for like

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=compiler-approximate-qft&utm_content=readme-top)

## The process, step by step

Setup:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-approximate-qft&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-aer qiskit-ibm-runtime networkx matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

### Step 1: three kinds of QFT input

```python
import circuitkit as ck, check
qft = ck.qft(24)                 # exact: h and every controlled phase, no final swaps
aqft6 = check.aqft(24, 6)        # textbook AQFT: keep phases of angle >= pi/2**6
aqft10 = check.aqft(24, 10)      # keep phases >= pi/2**10
```

Sizes 8 to 32 qubits [CHECK owner read: does the "no QFT above 32 qubits" rule bind an approximation post? This post
stays at 32 or fewer either way].

### Step 2: compile, exact and approximate, with the same time budget

The target is a description of the device, never a name: Qiskit's FakeFez snapshot (156 qubits, heavy-hex) with its
published calibration, for both compilers.

```python
import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

fez = targets.from_qiskit(FakeFez())
comp = fourcolors.quantum.compiler
exact = comp.compile(qft, target=fez, time_budget_s=10, seed=1, approximation_mode="exact", verify="wait")
approx = comp.compile(qft, target=fez, time_budget_s=10, seed=1, approximation_mode="per_block", verify="wait")
print(approx.verification.label, approx.verification.distance,
      approx.to_qiskit().metadata["fourcolors"].get("approximation_spent"))
```

`per_block` never forces an approximation: when the exact answer is better it comes back, and its label can then be
`verified`. Read the label and the distance on every result.

Qiskit runs `transpile(optimization_level=3)` on FakeFez as published, retrying seeds until the same 10 s are spent,
at `approximation_degree` 1.0 (no approximation) and at 0.999999 and 0.9999 (its own approximate synthesis). Both
compilers also get the textbook AQFT inputs and compile them exactly.

### Step 3: one scorer for both

`check.infidelity` measures how far a compiled circuit is from the exact QFT. It prepares random product inputs, each
qubit in one of the six Pauli eigenstates, runs the compiled circuit, then the exact inverse QFT on the qubits where
each input ended up, and reads how often everything comes back to |0>:

```python
r = check.infidelity(approx.to_qiskit(), ck.qft(24), n_inputs=8)
print(r["infidelity"], r["worst_infidelity"], r["active_qubits"])
```

Two details matter. Inputs like |+> are needed: with only 0/1 inputs a stray Z gate at the start is invisible.
And a matrix-product-state simulation keeps this cheap at 32 qubits, because the QFT creates little entanglement.
It is a sampled estimate, not a proof of equivalence; the service's own verification record is shown beside it.

### Step 4: run, then match at the same distance

```bash
python run.py --out results/aqft
python plot.py results/aqft --figs figs
```

For each Qiskit configuration, the match is our configuration of the same size with the fewest two-qubit gates whose
measured infidelity is the same or smaller. For ours, the worst seed's infidelity counts; for Qiskit, its best seed's.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that an exact circuit scores zero and a dropped phase scores above zero, that a Z error at the input
is invisible to 0/1 inputs and caught by the |+>-type ones, that a routed circuit with a shuffled layout and an
ancilla still scores as exact, that the MPS and statevector simulations agree, that the AQFT inputs drop the right
phases and sizes above 32 are refused, that the matching rule shows only pairs where ours has fewer gates at the same
or smaller distance (worst seed for us, best for Qiskit), and run the scripts' plumbing on fixed canned results.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-approximate-qft&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
