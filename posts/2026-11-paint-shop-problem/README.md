# The paint shop problem: one car line, solved as a QUBO and compiled as QAOA

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=paint-shop-problem&utm_content=readme-top), [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=paint-shop-problem&utm_content=readme-top)

## The model

A line of 2n bodies; each of n cars appears twice; one of its two bodies is painted white (0) and the other red
(1). One binary variable per car: `x[c]` is the color of car c's **first** body, and its second body gets
`1 - x[c]`. With `s[i]` the color at position i, the number of color changes is

```
changes(x) = sum over i of [s[i] != s[i+1]] = sum over i of (s[i] + s[i+1] - 2 s[i] s[i+1])
```

and since each `s[i]` is either `x[c]` or `1 - x[c]`, this is a QUBO with small integer coefficients whose energy
**is** the number of color changes. No penalty weights, nothing to tune.

```python
import paintshop as ps

word = ps.random_word(12, seed=1)        # each of 12 cars twice, in a random order
m = ps.model(word)                       # fourcolors.qubo Model, BINARY, minimize
x = [0] * 12
print(m.energy(x), ps.color_changes(word, x))   # the same number, by construction
```

Substituting `z = 1 - 2x` gives the Ising form `H = const + sum J_cd z_c z_d`, with one coupling per pair of cars
that stand next to each other somewhere on the line (couplings can cancel and vanish). Each car has at most four
neighbors, so the coupling graph is a chain with long-range jumps: easy to describe, awkward to place on a chip.

**Multi-car.** N bodies, each of a car class k; class k needs exactly `q[k]` black bodies. `x[i] = 1` if body i is
black; minimize `sum [x[i] != x[i+1]]` subject to one **exact row** per class, `sum over class k of x[i] = q[k]`.
The rows go to the optimizer as constraints, not as penalty terms, and every constrained answer is checked here.

```python
classes, quotas = ps.random_multicar(200, n_classes=4, seed=1)
m = ps.multicar_model(classes, quotas)   # the switch count plus 4 exact rows
```

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=paint-shop-problem&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-ibm-runtime numpy matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

The scripts import `../../shared/examplekit.py` and `../../shared/circuitkit.py`; run them from this directory.

### Step 1: a line you can check by hand

```bash
python solve.py --out runs/small --kind binary --sizes 12 16 20 --time-limit 1 --seeds 1
```

For each instance, `solve.py` sends the model, recounts the color changes from the returned 0/1 answer (never
trusting the service's number alone), and prints it beside the greedy heuristic and the exact optimum:

The exact optimum comes from `ps.brute_force`, which enumerates every coloring with car 0 white (flipping every
car gives the same count); it is fine up to about 22 cars.

### Step 2: sizes nobody enumerates

```bash
python solve.py --out runs/bpsp --kind binary --sizes 100 1000 10000 --time-limit 5 --seeds 1 2 3
```

At 10,000 cars there is no exact reference, so the comparison is with the greedy heuristic: a car's first body
takes the color of the body before it, and its second gets the other color. Greedy is fast and good; beating it
consistently is what a solver has to show.

### Step 3: the multi-car line, with exact quota rows

```bash
python solve.py --out runs/multi --kind multicar --sizes 16 200 2000 --classes 4 --time-limit 5 --seeds 1 2 3
```

Each row records `quotas_met` (checked here, class by class) next to the service's own feasibility flag, the switch
count, the sequence-order heuristic (each class paints its first `q[k]` bodies black), and, for N <= 16, the exact
optimum.

### Step 4: compile the QAOA circuit for a heavy-hex chip

```python
import circuitkit as ck
import fourcolors
from fourcolors.quantum import targets
from qiskit_ibm_runtime.fake_provider import FakeFez

word = ps.random_word(100, seed=1)
qc = ps.qaoa_circuit(word, p=1)          # one rzz per coupled pair of cars, measurements included
fez = targets.from_qiskit(FakeFez())     # the device's description and its published calibration
res = fourcolors.quantum.compiler.compile(qc, target=fez, time_budget_s=10, seed=1,
                                          approximation_mode="exact", verify="skip")
print(ck.circuit_metrics(res.to_qiskit()))   # two_qubit_gates, two_qubit_depth, depth, ...
```

The post compiles circuits; it does not optimize QAOA angles or run them. The compile cost is the same for any
angles, so `qaoa_circuit` uses a fixed linear ramp. `compile.py` runs both arms at 20, 50 and 100 cars, p = 1 and 2:

```bash
python compile.py --out runs/qaoa
python plot.py runs/small runs/bpsp runs/multi runs/qaoa --figs figs
```

The Qiskit arm is `transpile(qc, target=FakeFez().target, optimization_level=3, seed_transpiler=s)` with the
published calibration as-is, repeated with new seeds until the same number of seconds is spent, best kept.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check, on hand-made lines and by enumeration, that the QUBO energy and the Ising energy both equal the
color-change count, that each car's two bodies always get different colors, that the brute force finds the true
optimum and greedy never beats it, that the QAOA circuit has one rzz per coupling per layer and couplings of degree
at most four, that the multi-car model's rows accept exactly the colorings that meet the quotas, that the selection
rules show only strict wins, and the plumbing of `solve.py`, `compile.py` and `plot.py` with canned answers.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=paint-shop-problem&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
