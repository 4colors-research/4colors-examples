"""Quickstart: compile a circuit, then watch better circuits arrive.

    pip install fourcolors
    python quickstart/compiler/quickstart.py   # needs FOURCOLORS_API_KEY or FOURCOLORS_KEY_FILE

No key yet? Start a free trial at
    https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=quickstart-compiler&utm_content=script
and copy your key.

The code between the `region docs:` lines is the docs quickstart, word for word
(https://docs.4colors-research.com/quickstart-compiler/). The target is a device design that ships with the
package; for your own device, see tutorials/compiler/custom-target/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import examplekit  # noqa: E402

examplekit.load_key()

# region docs:compile
from fourcolors.quantum import compiler, targets

ghz = """OPENQASM 3.0;
include "stdgates.inc";
qubit[5] q;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
"""

target = targets.design_lattice("heavy_hex_156_cz")  # a bundled 156-qubit design
res = compiler.compile(ghz, target, time_budget_s=10)

print(res.metrics["two_qubit_gates"], res.metrics["depth"])
print(res.mapping["initial_layout"])  # where each of your qubits was placed
print(res.verification.label)         # what has been checked about the answer
print(res.circuit)                    # the compiled circuit, as OpenQASM 3
# endregion

# region docs:watch
job = compiler.submit(ghz, target, time_budget_s=60)

for res in job.incumbents():  # each better circuit, then the final result
    print(res.metrics["two_qubit_gates"])

# job.stop() ends the compile now and keeps the best circuit so far; Ctrl-C does the same.
# endregion
