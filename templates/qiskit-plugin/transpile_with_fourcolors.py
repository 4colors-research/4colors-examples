"""Template: compile a Qiskit circuit with the 4colors Quantum Compiler through Qiskit's own API.

Replace `get_backend()` with your provider's backend (any Qiskit BackendV2).
"""
from qiskit import QuantumCircuit, transpile

import fourcolors


def get_backend():
    from qiskit.providers.fake_provider import GenericBackendV2   # a generic Qiskit device description; use your provider's backend
    return GenericBackendV2(num_qubits=27)


qc = QuantumCircuit(5)
qc.h(0)
for i in range(4):
    qc.cx(i, i + 1)
qc.measure_all()

backend = get_backend()

# Option 1: Qiskit's transpile with all five stages named "fourcolors".
fourcolors.quantum.compiler.set_qiskit_options(time_budget_s=10)
out = transpile(qc, backend, init_method="fourcolors", layout_method="fourcolors", routing_method="fourcolors",
                translation_method="fourcolors", optimization_method="fourcolors")

# Option 2: a pass manager, with the compile record.
pm = fourcolors.quantum.compiler.generate_pass_manager(backend, time_budget_s=10)
out2 = pm.run(qc)
record = pm.property_set["fourcolors_result"]
print(out2.count_ops(), record.verification.label)
