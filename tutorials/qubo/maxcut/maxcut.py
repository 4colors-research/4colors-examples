"""Max-Cut on a large random graph: write it as a QUBO, stream better cuts, check the answer yourself.

    python maxcut.py                                  # 2,000 nodes, 6,000 edges, 30 s (needs a key)
    FOURCOLORS_OFFLINE=1 python maxcut.py --n 14 --m 30 --time-limit 1     # no key: tiny graph, local stand-in

Press Ctrl-C while it runs: the job stops on the service and you keep the best cut found so far.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
import examplekit  # noqa: E402

import fourcolors.qubo  # noqa: E402


def random_graph(n: int, m: int, seed: int = 7):
    rnd = random.Random(seed)
    edges = set()
    while len(edges) < m:
        i, j = rnd.sample(range(n), 2)
        edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def maxcut_qubo(n: int, edges):
    """E(x) = sum over edges of (2 x_i x_j - x_i - x_j); the cut size is -E(x)."""
    Q = {e: 2 for e in edges}
    linear = [0] * n
    for i, j in edges:
        linear[i] -= 1
        linear[j] -= 1
    return Q, linear


def cut_size(edges, x) -> int:
    return sum(1 for i, j in edges if x[i] != x[j])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--m", type=int, default=6000)
    p.add_argument("--time-limit", type=float, default=30)
    p.add_argument("--seed", type=int, default=7, help="graph seed")
    a = p.parse_args(argv)

    edges = random_graph(a.n, a.m, a.seed)
    Q, linear = maxcut_qubo(a.n, edges)
    client = examplekit.client()

    job = fourcolors.qubo.submit(Q, time_limit_s=a.time_limit, linear=linear, client=client)  # returns at once
    for best in job.incumbents():                     # each better solution as it is found
        print(f"cut {-best.objective} of {len(edges)} edges")
    result = job.result()

    x = result.solution                               # one 0/1 value per variable
    cut = cut_size(edges, x)
    print(f"{result.status}: {cut} edges cross between groups of {x.count(0)} and {x.count(1)} nodes")
    assert cut == -result.objective, "the service's objective and our own count agree"

    for step in result.trajectory[-5:]:
        print(f"t={step['t_search_s']:.2f} s  objective {step['objective']}")
    return result, edges


if __name__ == "__main__":
    main()
