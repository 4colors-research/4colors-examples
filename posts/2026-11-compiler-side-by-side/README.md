# Look inside the output: three compilers, one circuit, gate by gate

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler-side-by-side&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-side-by-side&utm_content=readme-text).
2. Install the client and the tools, and point the client at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime qiskit-aer qiskit-qasm3-import networkx pytest
pip install pytket pytket-qiskit
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

The scripts import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory.

### Step 1: freeze the device

```bash
python snapshot.py --backend fake:FakeKingston --out snap       # or ibm:<device> with your IBM account
```

`snapshot.py` writes the device's Qiskit Target as published (what Qiskit gets) and the 4colors description of
the same device with the same calibration (what the 4colors compiler gets), with the UTC time and the sha256 of both
files. Every later step reads this snapshot, so every arm sees the same numbers, however long the runs take.

### Step 2: compile with every arm

```bash
python run.py --snapshot snap --out out --arms rivals --circuits bv-n48
python check.py out/<device>
python run.py --snapshot snap --out out --arms ours --circuits bv-n48
```

The first line runs Qiskit at levels 2 and 3 (seeds 1 to 5, each with its own wall time), Qiskit level 3 with new
seeds until 1 s and 10 s are spent, and TKET's default pass if it is installed. `check.py` checks each output and
measures its distance from the input. The last line compiles with the 4colors compiler, once with no time budget
and once each with 1 s and 10 s, seeds 1 to 3, with the approximation bound read from the rivals' measured
distances:

```python
import fourcolors
import snapshot, circuits

qiskit_target, device, meta = snapshot.load("snap")
qc = circuits.build("bv-n48")
res = fourcolors.quantum.compiler.compile(qc, target=device, seed=1, approximation_mode="exact")
print(res.metrics["two_qubit_gates"], res.billed_s, res.job.id)
```

Every output is stored in full: the QASM, the initial layout, the time it used and the version.

### Step 3: open the outputs

```bash
python inspect_cell.py out/<device>/bv-n48 --out inspect
python select_cells.py inspect/<device> --out inspect/<device>/selection.md
```

`inspect_cell.py` counts every output with `sbs.py`, classifies its blocks, replays the moves from the initial
layout, and writes the heat maps and timelines below as SVG, on one shared scale per circuit. `select_cells.py`
applies the selection rule and writes which pairings are shown and why the others are left out.
`bash run_all.sh out/<device> inspect` does both for every circuit.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the counting and the pipeline; they do not call the service
```

The tests check the QASM reader, the counts against Qiskit's own on random tiny circuits, the block classes (CNOT,
SWAP, CNOT then SWAP, ZZ then SWAP), the move replay and the figures; that the Cuccaro adder adds, the lattice and
chain circuits have the stated edge counts, and the 32-qubit chain fits the chip; that the approximation bound is
never looser than any rival's measured distance and that a cell with an unmeasured distance compiles nothing; that
the checks pass correct outputs and catch a wrong secret or a perturbed gate; that the selection rule shows only
strict, checked wins; and the whole pipeline on two tiny circuits with fixed canned 4colors results.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-side-by-side&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
