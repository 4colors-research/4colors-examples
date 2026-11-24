"""The referee: reference values for published QPU Max-Cut benchmarks, and the benchmarks' own circuits.

Two published application benchmarks score a QPU on Max-Cut against a reference value:

* **Q-score** (Martiel, Ayral and Allouche, IEEE Trans. Quantum Eng. 2, 2021): Max-Cut on Erdős-Rényi graphs
  G(n, 1/2). The score compares the QPU's average cut C with an *asymptotic formula* for random graphs instead of
  each instance's optimum:  beta(n) = (C - n^2/8) / (lambda * n^(3/2)),  lambda = 0.178, and the QPU passes size n
  when beta > 0.2 [CHECK the constants against the paper]. The formula n^2/8 + lambda n^(3/2) is the expected
  optimum as n grows, so at finite n the real optimum of each instance can sit above or below it.
* **LR-QAOA** (Montañez-Barrera, Michielsen and Bernal Neira, arXiv:2502.06471, 2025): linear-ramp QAOA on up to
  156 qubits and 24 processors, on 1D chains, native layouts and fully connected graphs [CHECK which reference
  value the paper's ratio uses, per figure].

This module builds the instances (`er_half`, `fc_weighted`), the Q-score arithmetic (`expected_optimum`,
`qscore_beta`), an exhaustive Max-Cut for small n (`exact_maxcut`, a certificate) and the LR-QAOA circuit
(`lr_qaoa`). The best-known values for larger n come from the 4colors Optimizer for QUBO, labeled as "best found",
never "optimal", unless a certificate closes.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

Edge = Tuple[int, int]
LAMBDA = 0.178
PASS_BETA = 0.2


def er_half(n: int, seed: int) -> List[Edge]:
    """An Erdős-Rényi G(n, 1/2) graph (unweighted), as Q-score uses."""
    rng = random.Random(seed)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]


def fc_weighted(n: int, seed: int, levels: int = 10) -> Tuple[List[Edge], Dict[Edge, int]]:
    """A fully connected graph with integer weights 1..levels (uniform), in the shape of the fully connected
    benchmark instances [CHECK the paper's weight distribution]. Integer weights keep every cut value exact."""
    rng = random.Random(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return edges, {e: rng.randint(1, levels) for e in edges}


def expected_optimum(n: int) -> float:
    """Q-score's reference: the asymptotic expected maximum cut of G(n, 1/2), n^2/8 + lambda n^(3/2)."""
    return n * n / 8 + LAMBDA * n ** 1.5


def qscore_beta(cut: float, n: int) -> float:
    """beta(n) = (cut - n^2/8) / (lambda n^(3/2)): 0 for a random cut on average, 1 at the asymptotic optimum."""
    return (cut - n * n / 8) / (LAMBDA * n ** 1.5)


def beta_needed(n: int, optimum: float) -> float:
    """The beta a perfect solver (always the optimum) would score on this instance: the instance's own ceiling."""
    return qscore_beta(optimum, n)


def cut_value(edges: Sequence[Edge], x: Sequence[int], weights: Optional[Dict[Edge, int]] = None) -> int:
    return sum((1 if weights is None else weights[e]) for e in edges if x[e[0]] != x[e[1]])


def maxcut_model(n: int, edges: Sequence[Edge], weights: Optional[Dict[Edge, int]] = None):
    """Max-Cut for the 4colors Optimizer for QUBO: maximize sum_w (x_i + x_j - 2 x_i x_j), integer coefficients,
    no rows."""
    import fourcolors.qubo

    Q: Dict[Edge, int] = {}
    lin = [0] * n
    for e in edges:
        w = 1 if weights is None else weights[e]
        i, j = e
        lin[i] += w; lin[j] += w
        Q[(i, j)] = Q.get((i, j), 0) - 2 * w
    return fourcolors.qubo.to_model(Q, lin, n=n, sense="max")


def exact_maxcut(n: int, edges: Sequence[Edge], weights: Optional[Dict[Edge, int]] = None,
                 chunk: int = 1 << 18) -> Tuple[int, List[int]]:
    """The maximum cut by enumeration (node n-1 fixed to side 0), in chunks so memory stays small. A certificate
    for n <= about 26."""
    import numpy as np

    if n > 28:
        raise ValueError("exhaustive Max-Cut is for n <= 28")
    a = np.array([e[0] for e in edges]); b = np.array([e[1] for e in edges])
    w = np.array([1 if weights is None else weights[e] for e in edges], dtype=np.int64)
    best, arg = -1, None
    total = 1 << (n - 1)
    shifts = np.arange(n - 1, dtype=np.int64)
    for start in range(0, total, chunk):
        idx = np.arange(start, min(total, start + chunk), dtype=np.int64)
        bits = ((idx[:, None] >> shifts) & 1).astype(np.int8)
        bits = np.concatenate([bits, np.zeros((len(idx), 1), dtype=np.int8)], axis=1)
        cuts = ((bits[:, a] != bits[:, b]).astype(np.int64) * w).sum(axis=1)
        k = int(cuts.argmax())
        if cuts[k] > best:
            best, arg = int(cuts[k]), bits[k].tolist()
    return best, arg


def lr_ramp(p: int, delta_gamma: float = 0.6, delta_beta: float = 0.6) -> Tuple[List[float], List[float]]:
    """Linear-ramp angles: gamma_k = (k / p) * delta_gamma, beta_k = (1 - k / p) * delta_beta, k = 1..p
    [CHECK the paper's exact ramp and its delta values]."""
    gam = [(k / p) * delta_gamma for k in range(1, p + 1)]
    bet = [(1 - k / p) * delta_beta for k in range(1, p + 1)]
    return gam, bet


def lr_qaoa(n: int, edges: Sequence[Edge], weights: Optional[Dict[Edge, int]] = None, p: int = 3,
            measure: bool = True):
    """LR-QAOA for Max-Cut: H on every qubit, then p layers of rzz(gamma_k w_e) on every edge and rx(2 beta_k) on
    every qubit. The cost layer's gates commute, so their order is the compiler's to choose."""
    from qiskit import QuantumCircuit

    wmax = max((weights or {}).values(), default=1) or 1
    gam, bet = lr_ramp(p)
    qc = QuantumCircuit(n, name=f"lrqaoa_n{n}_p{p}")
    qc.h(range(n))
    for g, b in zip(gam, bet):
        for e in edges:
            w = 1 if weights is None else weights[e] / wmax
            qc.rzz(g * w, e[0], e[1])
        qc.rx(2 * b, range(n))
    if measure:
        qc.measure_all()
    return qc
