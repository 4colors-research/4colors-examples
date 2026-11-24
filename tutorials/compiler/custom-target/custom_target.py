"""Compile for your own device: a 100-qubit QAOA circuit for a bundled heavy-hex design and a 12 x 12 grid you
describe yourself.

    pip install fourcolors
    python tutorials/compiler/custom-target/custom_target.py    # two compiles of 120 s each

Needs a key (FOURCOLORS_API_KEY or FOURCOLORS_KEY_FILE): start a free trial at [TRIAL SIGNUP URL] and copy it.

The code between the `region docs:` lines is the docs tutorial, word for word
(https://docs.4colors-research.com/start/tutorials/compiler-custom-target/). The docs' last two snippets (your
own calibration, and a target read from your provider's backend object) need your device's data, so they are
not part of this script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
import examplekit  # noqa: E402

examplekit.load_key()

# region docs:qaoa
import random

n, rnd = 100, random.Random(5)
while True:  # a random 3-regular graph on 100 nodes
    stubs = [v for v in range(n) for _ in range(3)]
    rnd.shuffle(stubs)
    edges = {tuple(sorted(stubs[k:k + 2])) for k in range(0, len(stubs), 2)}
    if len(edges) == len(stubs) // 2 and all(a != b for a, b in edges):
        break

lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n}] q;", f"bit[{n}] c;"]
lines += [f"h q[{i}];" for i in range(n)]
for gamma, beta in ((0.41, 0.73), (0.82, 0.37)):  # two QAOA layers
    for a, b in sorted(edges):
        lines += [f"cx q[{a}], q[{b}];", f"rz({2 * gamma}) q[{b}];",
                  f"cx q[{a}], q[{b}];"]
    lines += [f"rx({2 * beta}) q[{i}];" for i in range(n)]
lines.append("c = measure q;")
qaoa = "\n".join(lines) + "\n"
# endregion

# region docs:grid
side = 12
couplers = [[r * side + c, r * side + c + 1]
            for r in range(side) for c in range(side - 1)]
couplers += [[r * side + c, (r + 1) * side + c]
             for r in range(side - 1) for c in range(side)]

grid = {
    "format_version": 2,
    "n_qubits": side * side,
    "two_qubit_gate": "cz",
    "couplers": couplers,
}
# endregion

# region docs:compile-both
from fourcolors.quantum import compiler, targets

heavy_hex = targets.design_lattice("heavy_hex_156_cz")

for name, target in (("heavy-hex 156", heavy_hex), ("12x12 grid", grid)):
    res = compiler.compile(qaoa, target, time_budget_s=120)
    print(name, res.metrics["two_qubit_gates"], res.metrics["depth"],
          res.verification.label)
# endregion
