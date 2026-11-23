# dimod sampler template

```bash
pip install "fourcolors[ocean]"
FOURCOLORS_OFFLINE=1 python sample_bqm.py     # no key: prints the model and exits
python sample_bqm.py                          # with FOURCOLORS_API_KEY
```

`fourcolors.FourColorsSampler()` is a dimod sampler: `sample(bqm, time_limit_s=...)` for a
`BinaryQuadraticModel`, and `sample(cqm, ...)` for a `ConstrainedQuadraticModel` with binary variables and linear
hard constraints. The SampleSet holds the single best sample found. See [`migrate/from-dimod-samplers.md`](../../migrate/from-dimod-samplers.md).
