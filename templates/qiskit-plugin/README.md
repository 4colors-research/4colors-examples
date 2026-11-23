# Qiskit plugin template

```bash
pip install "fourcolors[qiskit]"
python transpile_with_fourcolors.py           # needs FOURCOLORS_API_KEY and a Qiskit backend you can use
```

With the `qiskit` extra, the 4colors Quantum Compiler is Qiskit's five compile stages, all named `fourcolors`.
Give all five the value `"fourcolors"` (so Qiskit's own stages do not rewrite the answer), or build the stages with
`fourcolors.quantum.compiler.generate_pass_manager(backend, time_budget_s=...)`. See
[`migrate/from-qiskit-transpile.md`](../../migrate/from-qiskit-transpile.md).
