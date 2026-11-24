# The QAOA landscape, before and after noise

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=parametric-landscape&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=parametric-landscape&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-aer qiskit-qasm3-import networkx scipy matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

`run.py` and `plot.py` import `../../shared/examplekit.py` and the shared QAOA helpers in `../../qaoa-blog`; run them from this
directory. The compiles bill 10 s each (11 in the full run, under 2 compile-minutes); the simulation runs on your
machine, and that is where the time goes (see "How long it takes").

### Step 1: one parametric circuit

```python
from qaoakit import ansatz as az, core

inst = az.reg3_instance(16, 1)                   # a random 3-regular graph, 16 nodes, networkx seed 1
qc, (gamma, beta) = core.build(inst, "std-p1")   # H on every qubit, rzz(-gamma) per edge, rx(2 beta) per qubit
qc = az.to_input(qc)
qc.measure_all()
```

### Step 2: compile it once per compiler

```python
from fourcolors.quantum import compiler
from qaoakit import apikit

heavy_hex = apikit.device_target()      # the chip's couplers as a description (never a name); no calibration
res = compiler.compile(qc, target=heavy_hex, time_budget_s=10, seed=1, approximation_mode="exact")
out = res.to_qiskit()                   # still has gamma and beta: bind it like the input
```

Qiskit O3 gets the device as published: FakeFez's full 176-coupler map with its calibration as-is
(`run.compile_qiskit`, `transpile(qc, target=FakeFez().target, optimization_level=3)`). It spends the same 10
seconds on new seeds and keeps its fewest-gates circuit. The noise model below is uniform, so neither score depends
on the calibration.

### Step 3: bind on the grid and simulate

Every grid point binds the compiled circuit (`out.assign_parameters({gamma: g, beta: b})`; the script hands Aer a
whole row of the grid as parameter binds) and simulates it with depolarizing error e2 on every CZ, e2/10 on every sx
and x, none on rz (it is virtual), and symmetric readout flips. A few points are also simulated without noise and
must equal the ideal value (`check_ok`), which checks the compile and the qubit mapping.

**Half the grid is enough.** For a graph where every vertex has odd degree, as in a 3-regular graph, the p = 1
landscape is mirror-symmetric: f(pi - gamma, beta) = f(gamma, beta). Time reversal gives f(gamma, beta) =
f(-gamma, -beta), and because every vertex has odd degree, exp(-i pi C) is a Z on every qubit, which gives
f(gamma + pi, beta) = f(gamma, -beta); together they give the mirror. So `run.py` simulates the rows gamma <= pi/2
and mirrors the rest. That is exact for the ideal landscape. For a noisy compiled circuit it is not exact gate by gate
(the mirror turns each sx into its inverse), so on the first seed `run.py` also simulates two mirrored rows directly
and records the largest difference (`mirror_check_max_abs`), next to the scatter between two simulator seeds on the
same row (`repeat_check_max_abs`). `--no-mirror` simulates every row.

### Step 4: run it and draw it

```bash
python run.py --out runs/landscape --n 12,16 --grid 32 --seeds 1,2,3,4,5
python plot.py --out runs/landscape
```

`plot.py` writes the landscapes, the measured-against-predicted figure, and `metrics.md`: per compiled circuit its
gate counts, the contrast kept, the first-order prediction, how far the best point moved, and the two symmetry
checks; then which graphs the figure shows and the line that goes under it.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check the circuit's sign convention against a direct numpy simulator, the mirror symmetry and the gamma +
pi shift of the ideal landscape, that a mirrored grid equals the full grid, the first-order prediction against a
hand-computed value and against an exactly global-depolarizing toy, that the best-point distance treats the mirror
image as no move, and the whole pipeline on a tiny graph with fixed canned circuits.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=parametric-landscape&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
