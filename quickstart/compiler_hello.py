"""Hello, compiler: a 5-qubit GHZ circuit compiled onto the default hosted device target.

    python quickstart/compiler_hello.py       # needs FOURCOLORS_API_KEY or FOURCOLORS_KEY_FILE

No key yet? Start a free trial at [TRIAL SIGNUP URL] and copy your key.

`time_budget_s` is the seconds the compiler spends looking for a better circuit; the answer is the best it found
in that time. The service checks the answer is equivalent to your circuit (the verification label).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import examplekit  # noqa: E402

import fourcolors  # noqa: E402

GHZ5 = """OPENQASM 3.0;
include "stdgates.inc";
qubit[5] q;
bit[5] c;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
c = measure q;
"""

client = examplekit.client()
res = fourcolors.quantum.compiler.compile(GHZ5, time_budget_s=10, client=client)
print("two-qubit gates:", res.metrics.get("two_qubit_gates"), " depth:", res.metrics.get("depth"))
print("initial layout:", res.mapping.get("initial_layout"))
res.wait_verification(timeout=120)
print("verification:", res.verification.label)
print(res.circuit)
