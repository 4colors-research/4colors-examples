"""The model both scripts stream: Max-Cut on a random 3-regular graph, large enough that improvements keep coming
for a while. cut(x) = -E(x) (examplekit.maxcut_qubo)."""
from __future__ import annotations

import fourcolors.qubo

import examplekit


def build(n: int, degree: int = 3, seed: int = 5):
    edges = examplekit.random_regular(n, degree, seed)
    Q, lin = examplekit.maxcut_qubo(edges)
    return edges, fourcolors.qubo.to_model({k: int(v) for k, v in Q.items()}, [int(v) for v in lin], n=n)


def cut(edges, x) -> int:
    return int(examplekit.cut_value(edges, x))
