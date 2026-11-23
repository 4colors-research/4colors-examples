# From Qiskit's `transpile`

You already have:

```python
out = transpile(qc, backend, optimization_level=3)
```

With `pip install "fourcolors[qiskit]"`, keep the same call and name the 4colors stages:

```python
import fourcolors

fourcolors.quantum.compiler.set_qiskit_options(time_budget_s=30)
out = transpile(qc, backend, init_method="fourcolors", layout_method="fourcolors", routing_method="fourcolors",
                translation_method="fourcolors", optimization_method="fourcolors")
```

or build a pass manager:

```python
pm = fourcolors.quantum.compiler.generate_pass_manager(backend, time_budget_s=30)
out = pm.run(qc)
record = pm.property_set["fourcolors_result"]     # metrics, verification label, job
```

What changes:

- **Time, not levels.** There are no optimization levels; what buys a better circuit is `time_budget_s`, the
  seconds spent looking. The answer is the best found in that time.
- **All five stages.** Without `init_method="fourcolors"`, Qiskit's own init stage may rewrite your circuit first;
  with only layout and routing named, Qiskit's translation and optimization run afterward and may change the answer.
- **Checked answers.** Each compiled circuit carries a verification label (`verified`, `approximate` with its
  measured distance, `unverified` with a reason, or `pending` while the check runs).
- **Exact by default.** Nothing is approximated unless you ask (`approximation_mode="per_block"` or
  `max_approximation_error=...`).

Details: the [compiler docs](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=migrate&utm_content=qiskit).
