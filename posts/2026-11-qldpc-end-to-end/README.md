# A qLDPC code end to end: bound its distance, compile its syndrome round onto its own chip

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qldpc-end-to-end&utm_content=readme-top), [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=qldpc-end-to-end&utm_content=readme-top)

## What you will learn

- How to build a BB code's check matrices from two polynomials, and check its parameters in a few lines of GF(2).
- How to write "the least-weight logical operator" as a model with exact parity rows (mod-2 constraints as integer
  rows with slack bits), so every constrained answer is a certified upper bound on the distance.
- How to describe a code's Tanner graph as a custom chip, and compile a syndrome round onto it and onto heavy-hex.
- How to check a compiled syndrome round without trusting the compiler: inject errors and read the syndrome by
  Pauli propagation.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qldpc-end-to-end&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qldpc-end-to-end&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer,quantum]" qiskit qiskit-ibm-runtime matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

No dataset is downloaded: the codes are built from their polynomials.

## Step 1: the code

```python
import codes as cd

hx, hz, reported_d = cd.code("bb_72_12_6")   # A = x^3 + y + y^2, B = y^3 + x + x^2 on a 6 x 6 torus
assert cd.is_css(hx, hz) and cd.n_logical(hx, hz) == 12
L = cd.x_logicals(hx, hz)                     # 12 X-type logical operators
```

`H_X = [A | B]` and `H_Z = [B^T | A^T]`, where `x` and `y` are cyclic shifts. The unit tests compute n and k for all
five codes of the paper ([[72,12,6]], [[90,8,10]], [[108,8,10]], [[144,12,12]], [[288,12,18]]) and check them. The
distances are the paper's; we never claim to compute one.

## Step 2: the distance as exact rows

A Z-type error `e` is a logical operator when it commutes with every X check (`H_X e = 0 mod 2`) and anticommutes
with some X logical `l` (`l . e = 1 mod 2`). Mod-2 rows become exact integer rows with a few slack bits each:

```python
import fourcolors.qubo

m = cd.distance_model(hx, L[0])
#   minimize  sum_j e_j
#   subject to, exactly:  sum_{j in row} e_j - 2 (s_0 + 2 s_1) = 0   for each of the 36 X checks
#                         sum_{j in l}   e_j - 2 (t_0 + 2 t_1 + ...) = 1
r = fourcolors.qubo.solve(m, time_limit_s=10, seed=1)
e = cd.error_of(r.solution, 72)
assert cd.certify(hx, L[0], e)                # plain Python: a nontrivial Z logical of weight e.sum()
```

That is the integer-programming formulation used in the literature [CHECK: confirm in the Methods of Bravyi et al.
that their distances were computed this way before saying so], written as exact rows: no penalty weight, and every
constrained answer is checked by the service against its rows. Any answer that meets them all is a certified logical
operator, so its weight is an upper bound on the distance. `run.py` solves one model per X logical; the least weight
over all 12 is the distance whenever every job returns its optimum, because every nontrivial Z logical anticommutes
with at least one of them. The post reports bounds and compares them with the paper's distance; it does not claim
optimality.

## Step 3: the chip the code wants

```python
from fourcolors.quantum import targets

desc = cd.tanner_description(hx, hz)           # 144 qubits, one cz coupler per check-data pair: 432 couplers
```

72 data qubits, 36 X-check and 36 Z-check ancillas, and a coupler for every incidence of the Tanner graph: degree 6
everywhere, with long edges on any flat drawing (`tanner.png`). The description is built with the client's
`targets.build_description`. It has no calibration: it is a design, not a device.

## Step 4: one syndrome round, two chips

```python
import fourcolors.quantum.compiler as compiler
from qiskit_ibm_runtime.fake_provider import FakeFez

qc = cd.syndrome_round(hx, hz)                 # 432 CNOTs, 72 measurements
own = compiler.compile(qc, target=desc, time_budget_s=60, seed=1, approximation_mode="exact")
hh = compiler.compile(qc, target=targets.from_qiskit(FakeFez()), time_budget_s=60, seed=1,
                      approximation_mode="exact")
```

The round is a plain sequential schedule: every Z check, then every X check. It is correct for any CSS code, but it
is not the interleaved depth-8 schedule of Bravyi et al.; the post compares compilers on the same input, not
schedules.

## Step 5: check the compiled round yourself

```bash
python run.py --out runs/qldpc --stage full
python plot.py runs/qldpc --figs figures
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests compute n and k for all five BB codes and the 3 x 3 toric code, find the toric code's distance (3) by
enumeration, enumerate every assignment of a small distance model to show its feasible set is exactly the set of
certified logicals and its optimum the true one, check the Tanner chip's degrees, and check that the syndrome check
passes the plain round and a Qiskit-routed round and fails a round with one CNOT removed. The script plumbing runs
on fixed canned answers.

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
