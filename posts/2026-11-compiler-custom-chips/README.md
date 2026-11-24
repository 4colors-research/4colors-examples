# Which chip shape suits chemistry, QAOA or adders? Custom targets and the topology tax

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-custom-chips&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-custom-chips&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" networkx matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

`run.py` and `plot.py` import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory.

### Step 1: describe a chip

```python
import chips
from fourcolors.quantum import targets

chip = chips.heavy_hex(3, 7)      # 3 lines of 7 qubits, 4 bridge qubits: 25 qubits, max degree 3
print(chip["format_version"], chip["n_qubits"], len(chip["couplers"]), chip["couplers"][:3])

# what chips.heavy_hex builds, with no helper at all:
same = targets.build_description(25, "cz", ("rz", "sx", "x"), chip["couplers"])
```

A target is a description: the format version, the qubit count, the native two-qubit gate (and one-qubit gates)
and the couplers, or `"all_to_all"`. There is no device name anywhere. `chips.py` builds grids, rings, ladders,
heavy-hex patches and all-to-all chips this way.

For full-size device classes, the client ships design lattices, each with its own native gate set and no
calibration:

```python
print(targets.design_names())     # heavy_hex_156_cz, square_108_cz, square_54_prx, all_to_all_56_u1q, ...
hh156 = targets.design_lattice("heavy_hex_156_cz")
print(hh156.description["n_qubits"], hh156.calibration)    # 156 None
```

### Step 2: build the circuits

```python
import examplekit, circuitkit as ck

qaoa = ck.qaoa_maxcut(16, examplekit.random_regular(16, 3, seed=16), p=1)   # 16 qubits
bv = ck.bernstein_vazirani("1" * 15)                                     # 16 qubits, the full star
adder = ck.draper_adder(8)                                               # 16 qubits
chem = ck.uccsd(6, 4, seed=1)                                            # LiH STO-3G register: 12 qubits
print(ck.circuit_metrics(chem)["two_qubit_gates"])
```

`ck.uccsd` builds the UCCSD ansatz by hand (Jordan-Wigner, singles and doubles, Hartree-Fock reference), so the
post needs no chemistry package. The amplitudes are fixed and seeded, not optimized: the circuit has the
structure and gate cost of UCCSD on the LiH register, which is what the compiler sees. The Hartree-Fock reference
matters: without it every excitation acts on the vacuum as zero and the circuit is the identity on |0...0>
(the unit tests check both).

### Step 3: compile on each chip with a time budget

```python
import fourcolors

for name, build in chips.TAX_CHIPS.items():
    res = fourcolors.quantum.compiler.compile(chem, target=build(), time_budget_s=10, seed=1,
                                              approximation_mode="exact", verify="skip")
    print(name, ck.circuit_metrics(res.to_qiskit())["two_qubit_gates"], res.billed_s)

res = fourcolors.quantum.compiler.compile(chem, target=hh156, time_budget_s=10, seed=1,
                                          approximation_mode="exact", verify="skip")
```

`time_budget_s` is the one knob: the service returns the best circuit it found by then. The Qiskit arm gets the
same chip as a Qiskit `Target` (`chips.qiskit_target`: same couplers in both directions, same native gates, no
error rates) and runs `transpile(optimization_level=3)` with new seeds until the same number of seconds is spent,
keeping the best.

### Step 4: try a chip that is too big

```python
from fourcolors.errors import InvalidRequest, JobFailed
try:
    fourcolors.quantum.compiler.compile(bv, target=chips.grid(20, 20), time_budget_s=10)
except (InvalidRequest, JobFailed) as err:
    print(err.code, err.field, err.message)
```

### Step 5: run the whole comparison and draw it

```bash
python run.py --out results/chips       # parts tax, scale and design; 10 s; seeds 1..5
python plot.py results/chips --figs figs
```

`plot.py` prints the tables below, writes the figures and writes `figs/selection.md` (which Qiskit cells are shown,
and the line under each chart).

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that every chip is a valid, connected format version 2 description with the right shape (the
heavy-hex patch has 25 qubits and degree at most 3), that the Qiskit target has the same couplers and native gates
for each design lattice, that the adder adds, that the UCCSD builder has the 92 LiH excitations, keeps particle
number and spin, and is the identity on |0...0> without its reference state, that the selection rule shows Qiskit
only where ours strictly leads, and the plumbing of `run.py` and `plot.py` on fixed canned results.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-custom-chips&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
