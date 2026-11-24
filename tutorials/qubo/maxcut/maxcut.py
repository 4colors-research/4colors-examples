"""Max-Cut on a large random graph: write it as a QUBO, stream better cuts, check the answer yourself.

    pip install "fourcolors[optimizer]"
    python tutorials/qubo/maxcut/maxcut.py    # 2,000 nodes, 6,000 edges, 30 s

Needs a key (FOURCOLORS_API_KEY or FOURCOLORS_KEY_FILE): start a free trial at [TRIAL SIGNUP URL] and copy it.

The code between the `region docs:` lines is the docs tutorial, word for word
(https://docs.4colors-research.com/start/tutorials/qubo-maxcut/), and the notebook next to this file runs the same
code. Press Ctrl-C while it runs: the job stops on the service and you keep the best cut found so far.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
import examplekit  # noqa: E402

examplekit.load_key()

# region docs:build
import random

n, m = 2000, 6000
rnd = random.Random(7)
edges = set()
while len(edges) < m:
    i, j = rnd.sample(range(n), 2)
    edges.add((min(i, j), max(i, j)))

Q = {e: 2 for e in edges}  # quadratic terms: 2 x_i x_j per edge
linear = [0] * n           # linear terms: -1 per edge end
for i, j in edges:
    linear[i] -= 1
    linear[j] -= 1
# endregion

# region docs:solve
import fourcolors

job = fourcolors.qubo.submit(Q, time_limit_s=30, linear=linear)  # returns at once

for best in job.incumbents():  # each better solution as it is found
    print(f"cut {-best.objective} of {m} edges")

result = job.result()
print(result.status, -result.objective)
# endregion

# region docs:check
x = result.solution  # one 0/1 value per variable
cut = sum(1 for i, j in edges if x[i] != x[j])
print(f"{cut} edges cross between groups of {x.count(0)} and {x.count(1)} nodes")
# endregion

assert cut == -result.objective, "the service's objective and our own count of cut edges agree"

# region docs:trajectory
for step in result.trajectory[-5:]:
    print(f"t={step['t_search_s']:.2f} s  objective {step['objective']}")
# endregion
