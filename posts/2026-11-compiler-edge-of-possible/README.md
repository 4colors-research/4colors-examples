# The edge of the possible: where a compiled circuit stops working on a real chip

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-edge-of-possible&utm_content=readme-top)

## The process

### 1. Take one snapshot of the chip's calibration

```python
from qiskit_ibm_runtime import QiskitRuntimeService
backend = QiskitRuntimeService().backend("ibm_kingston")     # your IBM account
target = backend.target                                    # the published calibration, as-is
```

Both compilers get this same snapshot. Qiskit gets IBM's published numbers exactly, with no filtering of ours.

### 2. Compile with Qiskit at its best

```python
from qiskit import transpile
out = transpile(qc, target=target, optimization_level=3, seed_transpiler=s)
```

We repeat this with new seeds for 60 seconds and keep the seed with the highest estimated success probability on
the same snapshot. That is Qiskit's strongest preset plus a noise-aware pick that it doesn't make by default.

### 3. Compile with 4colors: give it the same seconds

```python
import fourcolors
from fourcolors.quantum import targets

res = fourcolors.quantum.compiler.compile(
    qc,
    target=targets.from_qiskit(backend),   # the same device and calibration
    time_budget_s=60,                      # the wall time Qiskit just used
    seed=1,
)
ours = res.to_qiskit()
print(res.metrics["two_qubit_gates"], res.billed_s, res.engine["version"])
```

You pay per compile-second, and the budget is the only setting. The compile is exact: the circuit it returns
computes the same thing as the one you sent.

### 4. Check both circuits before spending QPU time

BV compiles to a Clifford circuit, so a classical stabilizer computation checks the compiled circuit at full width
in milliseconds: run without noise, it must return the secret with probability 1. Both arms pass this check
before anything is submitted.

### 5. Run both arms in one job

```python
from qiskit_ibm_runtime import SamplerV2
sampler = SamplerV2(mode=backend)
sampler.options.dynamical_decoupling.enable = True      # the same for both arms
job = sampler.run(pubs, shots=2000)                      # every arm twice, order shuffled with a fixed seed
```

Both arms go in one job, so calibration drift during the day affects both alike. Each compiled circuit runs twice,
and the two repetitions have to agree (an A/A check) before a cell counts.

### 6. Score

```bash
python run.py compile --out runs/kingston --backend ibm_kingston
# ... submit pubs.json with your account (step 5), save the counts to runs/kingston/counts.json ...
python run.py score --out runs/kingston --counts runs/kingston/counts.json
```

`score` writes, for every size and arm: the exact-answer rate, the runner-up, the margin, whether the run is
usable, and the A/A check. It then writes the edge for each compiler.

## Try it on your own circuit

Bring a circuit that has to be routed onto a real chip: an adder, an oracle, or QAOA on a problem graph that isn't
the chip's own. Then:

1. compile it with `fourcolors.quantum.compiler.compile(qc, target=targets.from_qiskit(backend), time_budget_s=60)`,
2. compile it with Qiskit at level 3 for the same 60 seconds,
3. run both in one job and score them the way this post does.

If your circuit stops working on the chip at some size, find out whether it's the chip or the compiler. Get an API
key at [LINK], or ask us about a Quantum PoC on your own workload.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-edge-of-possible&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
