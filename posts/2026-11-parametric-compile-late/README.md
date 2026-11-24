# Compile early or compile late?

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=parametric-compile-late&utm_content=readme-top)

## The process, step by step

Setup:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=parametric-compile-late&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-aer networkx scipy matplotlib pyyaml
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

### Step 1: an optimizer trajectory

A noiseless COBYLA run (at most `--maxiter` evaluations) on two kinds of ansatz:

- QAOA, p = 2 and 3, on random 3-regular graphs (12 and 16 nodes), maximizing the expected cut;
- a UCCSD-style ansatz (every single and double excitation, Jordan-Wigner) on 6 and 8 spin orbitals with 2
  occupied, minimizing a toy number-conserving Hamiltonian, starting from the Hartree-Fock point (all angles zero),
  as chemistry runs usually do.

`--points` angle vectors are compiled, evenly spaced from the first evaluation after COBYLA's initial simplex to the
last. The start point is never compiled. At the Hartree-Fock point every excitation block is an identity, so a bound
compile there removes almost everything: true, and useless as evidence. The first n + 1 evaluations each move one
angle from the start, so they are left out too.

### Step 2: compile three ways

The target is the 156-qubit heavy-hex map as a description (its 169 working couplers, no calibration; there are no
named targets). Qiskit gets FakeFez as published: all 176 couplers with its calibration as-is (the rival gets the
pure calibration, the 7 couplers at error 1.0 included).

```python
import fourcolors
from qaoakit import apikit                # ../../qaoa-blog
heavy_hex = apikit.device_target()         # a description dict: n_qubits, cz, couplers
comp = fourcolors.quantum.compiler

# early: once, parametric (the same gates at every point)
early = comp.compile(ansatz, target=heavy_hex, time_budget_s=10, seed=1, approximation_mode="exact")

# late: bound, exact and per-block
bound = ansatz.assign_parameters(values)
late = comp.compile(bound, target=heavy_hex, time_budget_s=10, seed=1, approximation_mode="exact")
approx = comp.compile(bound, target=heavy_hex, time_budget_s=10, seed=1, approximation_mode="per_block")
approx.wait_verification(timeout=300)
print(approx.verification.label, approx.verification.distance)
```

`approximation_mode="exact"` is sent explicitly, so the record says what was asked. `per_block` never forces an
approximation: when the exact answer is better it is returned, and its label can then be `verified`, not
`approximate`. Read the label; do not assume it.

Qiskit runs `transpile(optimization_level=3)` on FakeFez's published target, retrying seeds until the same 10 s are spent, both on
the parametric circuit and on each bound one (`approximation_degree` 1.0, the default: no approximation).

### Step 3: check every circuit yourself

Every bound compile, from both compilers, is checked here on a noiseless statevector: the compiled circuit, cut down
to the qubits it touches, must prepare the bound input's state (routing permutes qubits; the layout says where each
ended). That fidelity, `fidelity_here`, is one scorer for both compilers.

```bash
python run.py --out runs/p3
python plot.py --out runs/p3
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that the UCCSD-style ansatz keeps the particle number and starts in the Hartree-Fock state, that the
toy Hamiltonian conserves the number, that the start point and the initial simplex are never compiled, that the
selection rule keeps only ansatze where ours is lower and both are exact (ties, an approximate Qiskit circuit and
unmeasured rows are left out), and run the scripts' plumbing on fixed canned results.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=parametric-compile-late&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
