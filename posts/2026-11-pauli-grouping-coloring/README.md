# Measure fewer times: Pauli grouping as graph coloring, priced in compiled gates

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=pauli-grouping-coloring&utm_content=readme-top)

## What you will learn

- How to build the conflict graph of a Hamiltonian in a few lines of NumPy (both rules).
- How to write K-coloring as a model with exact one-hot rows, so every constrained answer is checked, and how to walk
  K down to find fewer groups.
- How to build the diagonalization circuit of a commuting group, check it in plain Python, and compile it for a real
  device description.
- How to combine group count, a modeled shot allocation and compiled gate counts into one cost, and why a smaller
  group count alone can mislead.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=pauli-grouping-coloring&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=pauli-grouping-coloring&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer,quantum]" qiskit qiskit-ibm-runtime matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

3. Optional: Hamiltonians from HamLib (Sawaya et al. 2024). `hamlib_to_text.py` turns a HamLib HDF5 file into the
   text files `run.py` reads (`pip install h5py`). The built-in Fermi-Hubbard grids need nothing.

## Step 1: terms and the conflict graph

```python
import grouping as gp

terms = gp.fermi_hubbard(3, 2)                    # [(label, coefficient), ...], qubit 0 rightmost
labels = [t[0] for t in terms]
edges = gp.conflict_edges(labels, "gc")           # pairs that anticommute: they may not share a group
```

Two Paulis commute when they differ, both non-identity, on an even number of qubits. `conflict_edges` computes that
for every pair at once from the (x, z) bit matrices: one integer matrix product per chunk of rows.

## Step 2: coloring with exact rows

```python
import fourcolors.qubo

k = 4                                             # try one group fewer than the best baseline found
m = gp.coloring_model(len(terms), edges, k)       # x[i, c] = 1: term i goes to group c
#   rows:      sum_c x[i, c] == 1 for every term (exact), and term 0 in group 0 (a symmetry cut)
#   objective: the number of conflicting pairs that share a group
r = fourcolors.qubo.solve(m, time_limit_s=10, seed=1)
colors = gp.decode(r.solution, len(terms), k)
groups = gp.groups_from_colors(colors)
assert gp.n_conflicts(colors, edges) == 0 and gp.valid_grouping(labels, groups, "gc")
```

The "exactly one group per term" rule is a row, not a penalty, so there is no weight to tune, and the service checks
every constrained answer against its rows. The conflicts are the objective: an answer with objective 0 is a valid
grouping into at most `k` groups, and `valid_grouping` checks it again in plain Python, pair by pair.

`run.py` starts at one group fewer than the better of two baselines and walks down, one job per K, until no
conflict-free answer comes back:

## Step 3: a measurement circuit per group

```python
qc = gp.diagonalization_circuit([labels[i] for i in groups[0]])
assert gp.diagonalizes(qc, [labels[i] for i in groups[0]])      # every term becomes a Z-string
```

The circuit comes from the group's stabilizer state: take an independent subset of the group, build the Clifford that
prepares a state they stabilize, and invert it. After it, every term of the group is a string of Z's, so one
computational-basis measurement reads them all. `diagonalizes` checks this in plain Python by conjugating each term
through the circuit's Clifford tableau. A QWC group needs no two-qubit gate and is not compiled.

## Step 4: price each group on a chip

```python
import fourcolors.quantum.compiler as compiler
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

target = targets.from_qiskit(FakeFez())           # the device description with its published calibration
res = compiler.compile(qc, target=target, time_budget_s=3, seed=1, approximation_mode="exact")
two_qubit = res.metrics["two_qubit_gates"]
```

Then one number per pipeline, labeled as a model:

- **modeled shot share** of each group, in proportion to its modeled standard deviation (the root of the sum of
  its squared coefficients, covariances set to zero);
- **modeled two-qubit gates per shot** = the sum over groups of shot share times compiled two-qubit gates.

```bash
python run.py --out runs/grouping --hamlib-dir hamlib_txt --time-limit 10 --budget 3
python plot.py runs/grouping --figs figures
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check both commutation rules against Pauli matrices, the vectorized conflict graph against the pairwise rule,
the coloring model's rows and objective by enumerating every assignment of a small graph, the baselines' validity,
the diagonalization circuits on random commuting groups (including a dependent one, XX, YY, ZZ), the shot model by
hand, the OpenFermion text parser, and the two-site Hubbard spectrum. The script plumbing runs on fixed canned
answers.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
