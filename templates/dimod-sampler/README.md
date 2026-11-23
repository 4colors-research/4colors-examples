# dimod sampler template

1. Start a free trial at [TRIAL SIGNUP URL] and copy your key.
2. Run the template:

```bash
pip install "fourcolors[ocean]"
export FOURCOLORS_API_KEY=...                 # the key from step 1
python sample_bqm.py
```

`fourcolors.FourColorsSampler()` is a dimod sampler: `sample(bqm, time_limit_s=...)` for a
`BinaryQuadraticModel`, and `sample(cqm, ...)` for a `ConstrainedQuadraticModel` with binary variables and linear
hard constraints. The SampleSet holds the single best sample found. See [`migrate/from-dimod-samplers.md`](../../migrate/from-dimod-samplers.md).
