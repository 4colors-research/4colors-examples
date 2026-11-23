# From dimod samplers

You already have:

```python
sampleset = SomeSampler().sample(bqm, num_reads=100)
```

With `pip install "fourcolors[ocean]"`:

```python
import fourcolors

sampleset = fourcolors.FourColorsSampler().sample(bqm, time_limit_s=10)
best = sampleset.first
```

What changes:

- **One best answer, not many reads.** The service returns the single best sample found within `time_limit_s`;
  `num_reads` above 1 is refused.
- **Time is the knob.** `time_limit_s` is the search time you pay for. Longer usually means better, up to a point:
  the result's trajectory shows when the last improvement came.
- **Constraints stay constraints.** A `ConstrainedQuadraticModel` with binary variables and linear hard constraints
  goes as it is, no penalty weights to tune; the service checks every row exactly
  (`sampleset.info["constraints"]` is the authoritative verdict).
- **No dimod needed** for plain models: `fourcolors.qubo.solve(Q, time_limit_s=..., linear=...)` takes numpy and
  scipy matrices and dicts.

Template: [`templates/dimod-sampler/`](../templates/dimod-sampler/).
