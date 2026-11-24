"""Template: solve a dimod BinaryQuadraticModel with the 4colors Optimizer for QUBO.

Needs a key (FOURCOLORS_API_KEY): start a free trial at
    https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=templates&utm_content=script
and copy your key.
"""
import dimod

bqm = dimod.BinaryQuadraticModel({"a": -1, "b": -1, "c": -1}, {("a", "b"): 2, ("b", "c"): 2}, 0.0, "BINARY")
print(bqm)

import fourcolors  # noqa: E402

sampleset = fourcolors.FourColorsSampler().sample(bqm, time_limit_s=1)
best = {v: int(x) for v, x in sampleset.first.sample.items()}   # plain ints (numpy types print oddly)
print(best, sampleset.first.energy)
