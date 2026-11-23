"""Hello, QUBO: the maximum cut of a 5-node ring.

    FOURCOLORS_OFFLINE=1 python quickstart/qubo_hello.py      # no key: a local stand-in answers
    python quickstart/qubo_hello.py                           # with FOURCOLORS_API_KEY or FOURCOLORS_KEY_FILE

Edge (i, j) is cut when x_i != x_j. Minimizing sum over edges of (2 x_i x_j - x_i - x_j) maximizes the cut,
and the cut size is minus that objective.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import examplekit  # noqa: E402

import fourcolors.qubo  # noqa: E402

edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
Q = {e: 2 for e in edges}                       # 2 x_i x_j per edge
linear = [0] * 5
for i, j in edges:                              # -x_i - x_j per edge
    linear[i] -= 1
    linear[j] -= 1

client = examplekit.client()
result = fourcolors.qubo.solve(Q, time_limit_s=1, linear=linear, seed=1, client=client)

x = result.solution
cut = sum(1 for i, j in edges if x[i] != x[j])  # check the answer yourself
print(f"groups {x}, cut {cut} of {len(edges)} edges (objective {result.objective}, engine "
      f"{(result.engine or {}).get('version')})")
assert cut == -result.objective
