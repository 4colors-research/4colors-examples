"""Time budgets for the compiler: compile one circuit with the budget the compiler picks, then with 20 seconds.

    pip install fourcolors
    python tutorials/compiler/time-budgets/time_budgets.py

Needs a key (FOURCOLORS_API_KEY or FOURCOLORS_KEY_FILE): start a free trial at
    https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-time-budgets&utm_content=script
and copy it.

The code between the `region docs:` lines is the docs tutorial, word for word
(https://docs.4colors-research.com/start/tutorials/compiler-time-budgets/).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
import examplekit  # noqa: E402

examplekit.load_key()

# region docs:target
from fourcolors.quantum import targets

target = targets.design_lattice("heavy_hex_156_cz")
print(target.description["n_qubits"], target.description["two_qubit_gate"])
# endregion

# region docs:circuit
import random

n, n_cx = 25, 250
rnd = random.Random(3)
lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n}] q;"]
for _ in range(n_cx):
    a, b = rnd.sample(range(n), 2)
    lines += [f"rz({rnd.uniform(-3, 3):.4f}) q[{a}];", f"cx q[{a}], q[{b}];"]
circuit = "\n".join(lines) + "\n"
# endregion

# region docs:default-budget
from fourcolors.quantum import compiler

first = compiler.compile(circuit, target)
print(first.metrics["two_qubit_gates"], first.metrics["depth"], first.billed_s)
# endregion

# region docs:streamed
job = compiler.submit(circuit, target, time_budget_s=20)

for res in job.incumbents():
    print(f"two-qubit gates {res.metrics['two_qubit_gates']}, "
          f"depth {res.metrics['depth']}")

best = job.result()
print(best.metrics["two_qubit_gates"], best.billed_s)
# endregion
