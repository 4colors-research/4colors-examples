# Solve it, then compile it: an honest reference for QAOA

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qaoa-solve-then-compile&utm_content=readme-top), [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=qaoa-solve-then-compile&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qaoa-solve-then-compile&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit,optimizer]" qiskit-aer qiskit-qasm3-import networkx scipy matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

`run.py` and `plot.py` import `../../shared/examplekit.py` and the QAOA helpers in `../../qaoa-blog/qaoakit/`; run them from
this directory.

### Step 1: solve it

Max-Cut is a QUBO: with `Q[i,i] = -deg(i)` and `Q[i,j] = 2 w_ij`, `x^T Q x = -cut(x)`, so the minimum is minus the
maximum cut.

```python
import fourcolors.qubo
from qaoakit import ansatz, apikit, core

inst = ansatz.reg3_instance(16, 1)                       # a random 3-regular graph, 16 nodes
Q = apikit.maxcut_matrix(inst["nodes"], inst["edges"])   # x^T Q x = -cut(x)
r = fourcolors.qubo.solve(Q, time_limit_s=1, seed=1)
cut = -r.objective
print(cut, core.cost_vector(inst).max())                 # the optimizer's cut, and the enumerated optimum
print(r.trajectory)                                      # every improvement with its search time
```

The enumeration (all 2^16 cuts) is the certificate; it is only possible because the graphs are small. "Reach the
optimizer's answer" below means: sample a cut at least as large as this one.

### Step 2: build the QAOA circuit and pick its angles

```python
qc, (gamma, beta) = core.build(inst, "std-p1")           # H, then rzz(-gamma w) per edge and rx(2 beta) per node
cost = core.cost_vector(inst)
angles, expected_cut, evals = core.tune_numpy(16, cost, 1, restarts=4, seed=7)   # noiseless, COBYLA
```

Two ways to choose angles: optimize them on a noiseless simulation of the circuit (the evaluations are the classical
overhead, recorded per row), or use the published fixed angles for 3-regular graphs (Wurtz and Lykov 2021, p <= 3),
which need no optimizer loop. Deeper p starts from the INTERP extrapolation of the previous depth (Zhou et al. 2020).

### Step 3: compile it for the chip, with a time budget

```python
import fourcolors
heavy_hex = apikit.device_target()        # FakeFez's 169 working couplers, as a description (never a name)
qc_in = ansatz.to_input(qc); qc_in.measure_all()
res = fourcolors.quantum.compiler.compile(qc_in, target=heavy_hex, time_budget_s=5, seed=1,
                                          approximation_mode="exact")
out = res.to_qiskit().assign_parameters({gamma: angles[0], beta: angles[1]})
print(apikit.gate_stats(out), res.billed_s, res.job.id)
```

The circuit is compiled once with its angles as parameters, so every angle choice binds the same compiled circuit.
The Qiskit arm is `transpile(..., optimization_level=3)` on FakeFez's published target (all 176 couplers, its
calibration as-is, the 7 couplers at error 1.0 included: the rival gets the pure calibration), repeated with new
seeds until the same 5 seconds are spent, keeping the fewest two-qubit gates. Both arms are scored by the same rules:
the stated noise model, and `apikit.device_score` (an estimated success probability from the published calibration,
0 for any gate on an error-1.0 coupler).

### Step 4: score it under a stated noise model

```python
stats, probs = core.noisy_ratio(out, 16, cost, cost.max(), e2=3e-3, e_ro=0.01, trajectories=100, batches=5)
print(stats["mean"], stats["se"])                        # E[cut] / max cut under noise, and its standard error
```

The model: depolarizing error e2 on every CZ, e2/10 on SX and X, symmetric 1% readout error, averaged over noisy
statevector trajectories. It is a simulation of the compiled circuit, not a run on hardware.

### Step 5: run the comparison and draw it

```bash
python run.py --out results/solve            # optimizer panel + QAOA panel (p = 1-3; p = 4, 5 on n = 12, 16)
python plot.py --out results/solve
```

`plot.py` writes the figures, `breakeven.md`, `cost.md` and `selection.md` (which cells each chart shows and why the
others are left out).

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check the QUBO sign convention against every assignment, the success probability and SWAP-network count,
the break-even interpolation (including "never reached"), that the selection rules show only strict wins (ties and
losses left out), that both shot counts are tabulated, and the plumbing of `run.py` and `plot.py` on canned results
(the later panels too).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qaoa-solve-then-compile&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
