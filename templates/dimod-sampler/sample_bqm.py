"""Template: solve a dimod BinaryQuadraticModel with the 4colors Optimizer for QUBO.

Needs a key (FOURCOLORS_API_KEY): start a free trial at [TRIAL SIGNUP URL] and copy your key.
"""
import dimod

bqm = dimod.BinaryQuadraticModel({"a": -1, "b": -1, "c": -1}, {("a", "b"): 2, ("b", "c"): 2}, 0.0, "BINARY")
print(bqm)

import fourcolors  # noqa: E402

sampleset = fourcolors.FourColorsSampler().sample(bqm, time_limit_s=1)
print(sampleset.first.sample, sampleset.first.energy)
