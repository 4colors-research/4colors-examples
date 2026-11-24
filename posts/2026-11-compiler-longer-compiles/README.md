# Give it a second, and when longer pays: what more compile time buys, for 4colors and for Qiskit

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-longer-compiles&utm_content=readme-top)

## The process, step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-longer-compiles&utm_content=readme-text).

`run.py` and `plot.py` import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory.

### Step 1: build the circuits

```python
import circuits

qc, info = circuits.build("qaoa_ba3_50_p3")     # QAOA Max-Cut, 3 layers, Barabasi-Albert graph, 50 qubits
print(info["qubits"], info["input_2q"])
```

`circuits.py` holds the twelve circuits, fixed before any run: QAOA Max-Cut on Barabasi-Albert graphs (27, 50 and
100 qubits) and on a 3-regular graph (50), a 2D transverse-field Ising Trotter circuit (36), a UCCSD-type
chemistry ansatz (16), a hardware-efficient ansatz with full entanglement (16), a ZZ feature map (20), an XY-mixer
portfolio QAOA (20), the QFT (32), Grover search (8) and amplitude estimation (9). `python circuits.py --out
circuits/` writes each one once as OpenQASM 3 and QPY with its sha256; both compilers read the same file.

### Step 2: give the 4colors compiler a second, and watch the stream

```python
import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

fez = targets.from_qiskit(FakeFez())          # the device as Qiskit ships it, published calibration included
res = fourcolors.quantum.compiler.compile(qc, target=fez, time_budget_s=1, seed=1)
print(res.metrics["two_qubit_gates"], res.billed_s)

import time

t0 = time.monotonic()
job = fourcolors.quantum.compiler.submit(qc, target=fez, time_budget_s=60, seed=1)
points = []
for r in job.incumbents():                     # each better circuit as it is found, then the final one
    points.append((time.monotonic() - t0, r.metrics.get("two_qubit_gates")))
```

### Step 3: decide whether longer pays

```python
import plateau

print(plateau.last_improvement_s(points))      # when the last better circuit arrived
print(plateau.time_to_within(points, 0.01))    # when it was first within 1 % of the final count
print(plateau.advice(points, budget_s=60))     # "plateau" or "longer may pay"
```

The rule of thumb in `plateau.advice` is simple: if the second half of the budget still bought more than 1 %, try
a longer budget; if it bought nothing, the answer you have is the answer. Run it once per circuit family you
care about, not once per circuit.

### Step 4: give Qiskit the same time, properly

Qiskit's `optimization_level=3` is a fixed recipe. VF2Layout looks for a perfect placement; if there is none,
SabreLayout runs 20 layout trials with 4 forward-backward passes, then SabreSwap routes with its "decay"
heuristic, and an optimization loop runs until five rounds bring no gain. Reseeding repeats that recipe. The
post's Qiskit arm, **Qiskit-long** (`qiskit_long.py`), changes the recipe itself:

```bash
python qiskit_long.py --qasm circuits/qaoa_ba3_50_p3.qasm --id qaoa50 --budget 60 --workers 4 --out out/qaoa50
```

N worker processes, one per core, every thread pool at one thread, share these configurations. Each member doubles
its effort whenever the bigger run still fits in a third of the time left, and no trial starts that is not
predicted to finish before the deadline. The clock starts after every worker has imported Qiskit and warmed up.
When time is up, the best circuit wins, by the metric the chart shows. The driver is about 700 lines of plain
Qiskit, and you can use it on your own circuits.

### Step 5: run the comparison and draw it

```bash
python run.py --out runs/qiskit-long   --arm qiskit-long   --workers 4 --cpus 4,5,6,7
python run.py --out runs/qiskit-reseed --arm qiskit-reseed --workers 4 --cpus 4,5,6,7
python run.py --out runs/4colors       --arm 4colors --qiskit runs/qiskit-long --stream
python plot.py runs/4colors runs/qiskit-long runs/qiskit-reseed --figs figs
```

The Qiskit arms run first: the 4colors arm reads, for each cell, how long Qiskit-long actually ran and how much it
approximated, and gets no more of either. `plot.py` writes the figures, the table and `figs/selection.md` (which
cells are shown, which are left out and why, and the line under each chart).

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that every circuit is gate-level, deterministic and the size the design says, that the UCCSD and
portfolio circuits keep their particle number and budget, that Grover amplifies its marked string and amplitude
estimation peaks at the right amplitude, that every Qiskit-long member keeps the operator and stays on the device's
couplers, that effort changes the passes, that the scheduler never starts a trial that cannot finish, that the
scorer and the driver agree on FakeFez, that the operator check is zero for an exact compile and sees a wrong one,
the plateau reader, the selection rule (tie band, best single member, over-budget, plan floor, verification,
strict wins), and the plumbing of `run.py` and `plot.py` on canned results.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-longer-compiles&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
