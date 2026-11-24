# Compile once, bind a thousand times

**Docs:** [4colors Quantum Compiler](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=parametric-compile-once&utm_content=readme-top)

## Step by step

### Setup

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=parametric-compile-once&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[qiskit]" qiskit-aer qiskit-ibm-runtime qiskit-qasm3-import scipy matplotlib pyyaml
export FOURCOLORS_KEY_FILE=/path/to/key-file   # the key is read from the file, in the process
```

The scripts also import `../../shared/examplekit.py` and the shared QAOA code in `../../qaoa-blog/qaoakit/` (ansatze, the
equivalence check, both compilers at a matched budget).

### Step 1: compile once, through the Qiskit plugin

```python
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime.fake_provider import FakeFez
from fourcolors.quantum import compiler

pm = compiler.generate_pass_manager(FakeFez(), time_budget_s=30, seed=1,      # the job's target: FakeFez's own
                                    approximation_mode="exact")              # description and calibration
out = pm.run(ansatz)
record = pm.property_set["fourcolors_result"]                                 # job, billed_s, verification
```

Or without the plugin: `res = compiler.compile(ansatz, targets.from_qiskit(FakeFez()), time_budget_s=30, seed=1,
approximation_mode="exact")` (`from fourcolors.quantum import targets`), then `out = res.to_qiskit()`. A target is a
description of the device, never a name. `time_budget_s` is the one knob: the service returns the best circuit it
found by then; with no budget it returns its instant answer. `res.parameters` lists the names to bind, in your
circuit's order. (The post's runs give both 4colors routes the same map, the device's 169 working couplers as a
description: `ak.device_target()`. Qiskit gets FakeFez's published target: `ak.qiskit_target()`.)

### Step 2: bind many times

```python
for values in steps:                                   # values: one float per parameter
    bound = out.assign_parameters(dict(zip(ansatz.parameters, values)))   # by your own Parameter objects
    # or by position: out.assign_parameters(values), in ansatz.parameters order
```

### Step 3: check it

### Step 4: an OpenQASM text input is bound by name

A circuit sent as OpenQASM text comes back with new parameters, and Qiskit sorts them by name (`t0, t1, t10, t11,
t2, ...`). Bind them by name:

```python
out = compiler.compile(qasm3_text, targets.from_qiskit(FakeFez()), time_budget_s=30).to_qiskit()
bound = out.assign_parameters({p: values_by_name[p.name] for p in out.parameters})
```

(`bind_by_position_ok`). Keep the result of `compile()`: the service does not keep your parameter names.

> **Three ways to bind wrong**
>
> 1. **New objects for the same circuit.** A new `ParameterVector` has new identities, even with the same names.
>    Bind with the `Parameter` objects of the circuit you compiled, or by position on a Qiskit input's result.
> 2. **Position after a text round trip.** OpenQASM text in means parameters out sorted by name (`t10` before
>    `t2`). Bind by name.
> 3. **A nonlinear expression.** `a**2` becomes one more generated name in the circuit text (`res.circuit`), so a
>    positional bind from the text alone raises a length error. `to_qiskit()` on the result still gives your `a`.

### Step 5: the loop

A COBYLA loop on the Aer estimator (noiseless, exact expectation) for QAOA p = 2 on a 12-node 3-regular graph, four
ways: compiled once by 4colors, compiled once by Qiskit, compiled at every step by 4colors, and compiled at every
step by Qiskit. Equivalent circuits give the same energies, so every arm walks the same path; only the compile cost
differs. The compile-once arms run 60 evaluations; the compile-per-step arms stop at 20 (`--per-step-evals 20`),
because compiling at every step is the pattern this post argues against, so their final ratios are lower for that
reason alone.

### Run it

```bash
python run.py --out runs/p1         # defaults: 8 ansatze, 2-480 parameters, budget 30 s, seed 1
python plot.py --out runs/p1        # gates.png, compile_cost.png, loop.png, table.md
```

`results.jsonl` rows (`kind`): `compile` (one per ansatz and route: two-qubit gates, depth, parameter count,
verification label, job id, billed seconds, refusal code), `bindings` (gate count per binding, binding time, checks),
`text_demo`, `loop` (evaluations, compile seconds, billed seconds, energy trace, final ratio against the brute-force
maximum cut).

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that the binding check accounts for routing (a routed circuit passes, a wrong binding fails), that
parameters survive both routes, that the selection rule charts only strict wins per panel, and check the script's
plumbing on fixed circuits: every route including the instant comparison, the text demo (by name passes, by
position fails at 12 parameters) and the four loop arms walking the same path.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=parametric-compile-once&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
