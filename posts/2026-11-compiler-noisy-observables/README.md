# Fewer gates, better answers? Chemistry, finance and materials observables under a stated noise model

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-noisy-observables&utm_content=readme-top)

## The process

1. Build circuits that actually do something to |0...0> (this is the step most benchmark generators get wrong).
2. Compile each for the device with its published calibration: 4colors with a time budget, Qiskit at
   optimization level 3 with best-of-seeds in the same seconds, both exact.
3. Simulate each compiled circuit under one noise model built from the same calibration: gate errors, T1 and T2
   relaxation during gates and during idle time, readout error.
4. Read the domain observable from each output and compare it with the ideal value from an exact statevector.

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-noisy-observables&utm_content=readme-text).
2. Install the client and the simulator, and point the client at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime qiskit-aer matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

`run.py`, `simulate.py` and `plot.py` import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory.

### Step 1: circuits that are not trivial on |0...0>

```python
import circuitkit as ck, domain_circuits as dc

qc, spec = dc.build("uccsd_12")          # UCCSD on the LiH STO-3G register, Hartree-Fock reference
print(spec, dc.trivial_reason(qc))       # {'observable': 'occupations', ...} None

bare = ck.uccsd(6, 4, reference=False, measure=False)
print(dc.trivial_reason(bare))           # 'identity on |0>'
```

A UCCSD excitation operator moves an electron from an occupied orbital to an empty one. Started from the vacuum
|0...0>, there is no electron to move: every factor acts as the identity, and the whole circuit, thousands of
CNOTs, returns |0...0>. The same happens to a portfolio QAOA with an XY mixer started from |0...0> (the mixer
conserves the number of assets, and zero assets stay zero). Such a circuit makes an observable study meaningless:
both compilers' outputs give the ideal answer up to readout error, and a compiler allowed to approximate can delete
the whole circuit. `trivial_reason` flags it, and `build` refuses it. The post's circuits fix it at the source:

### Step 2: compile for the device, with a time budget

```python
import fourcolors
import circuitkit as ck

fez = ck.heavy_hex_target()              # targets.from_qiskit(FakeFez()): description + published calibration
res = fourcolors.quantum.compiler.compile(qc, target=fez, time_budget_s=10, seed=1,
                                          approximation_mode="exact", verify="skip")
ours = res.to_qiskit()
print(ck.circuit_metrics(ours)["two_qubit_gates"], res.billed_s)
```

The Qiskit arm is `transpile(qc, target=FakeFez().target, optimization_level=3, seed_transpiler=s)` with new seeds
until the same 10 seconds are spent, keeping the circuit with the fewest two-qubit gates. Both arms see the same
published calibration as-is (failed couplers included), and both are exact: `approximation_mode="exact"` for ours,
Qiskit's default approximation degree for theirs.

### Step 3: the noise model

```python
import noise, observables
from qiskit_ibm_runtime.fake_provider import FakeFez

sim = noise.simulator(FakeFez())                         # Aer's device noise model from the calibration
dist, meta = noise.noisy_distribution(ours, sim, FakeFez().target, shots=100_000, seed=1)
noisy = observables.measure(spec, dist)
ideal = observables.measure(spec, dc.ideal_probabilities(qc))
print(observables.error(spec, noisy, ideal), meta)
```

The noise model is stated in full in `noise.py`: a depolarizing error per gate at its calibrated error, thermal
relaxation with the calibrated T1 and T2 for each gate's duration, a readout error per qubit, and idle time
charged as relaxation (each circuit is scheduled as late as possible with the calibrated durations). It is built
from one calibration snapshot, the current published one, and every compiled circuit, ours and Qiskit's, goes
through the same model. Up to 13 active qubits the simulation is an exact density matrix; above that it samples
noisy trajectories.

### Step 4: run everything and draw it

```bash
python run.py --out results/obs                     # 20 circuits, 10 s, seeds 1..3, both arms
python simulate.py results/obs --out results/sim    # the noise model on every compiled circuit
python plot.py results/sim --figs figs
```

`plot.py` prints the table, writes the figures and writes `figs/selection.md` (which circuits are shown and the
line under each chart).

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that every post circuit is non-trivial on |0...0> and that the old generators (UCCSD without a
reference, portfolio QAOA from |0...0>) are flagged, that the Dicke state is uniform over k-asset portfolios, that
the cost layer is the mean-variance cost, that the XY mixer keeps ideal feasibility at 1, that UCCSD stays in its
particle-number sector, that count keys are read in the right bit order, that idle delays land only on used
qubits, that the noise model lowers feasibility on a small portfolio, that the selection rule needs fewer gates
and a lower error beyond shot noise, and the plumbing of `run.py`, `simulate.py` and `plot.py` on a fixed canned
result.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-noisy-observables&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
