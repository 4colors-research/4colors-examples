"""Aircraft loading: choose which containers fly and where, within weight and balance limits.

The problem, in the shape of the aircraft-loading studies (for example Kaushik et al., "Quantum Computing for
Optimizing Aircraft Loading", arXiv:2504.01567, 2025, [CHECK the exact model against the paper]):

* positions 0..P-1 along the hold, each with an arm d_j (its distance from the reference point, in integer units,
  negative forward) and a structural weight limit cap_j (lower toward the ends);
* containers 0..N-1 with integer weights (kg); x[i, j] = 1 when container i goes to position j, only for pairs with
  w_i <= cap_j;
* each container at most once; each position at most one container;
* the payload limit: sum w_i x_ij <= W;
* the center of gravity inside the envelope [lo, hi]:  lo * (sum w x) <= sum w_i d_j x_ij <= hi * (sum w x), i.e.
  two linear rows  sum w_i (d_j - lo) x_ij >= 0  and  sum w_i (d_j - hi) x_ij <= 0;
* maximize the loaded weight.

`model` writes it for the 4colors Optimizer for QUBO as a linear objective plus exact rows (no penalty weights, no
slack variables). `milp_reference` solves the same rows with HiGHS. `penalty_qubo` writes the textbook penalty form
a QAOA circuit is built from (the inequality rows squared without slack, so every pair of variables interacts), and
`qaoa_circuit` builds that circuit: the "compile cost of the quantum formulation" side of the post.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import fourcolors.qubo


@dataclass
class Hold:
    arms: List[int]                 # d_j, integer units
    caps: List[int]                 # kg per position
    weights: List[int]              # kg per container
    payload: int                    # W, kg
    lo: int                         # CG envelope, arm units
    hi: int
    pairs: List[Tuple[int, int]]    # allowed (container, position), the variables in order

    @property
    def n(self) -> int:
        return len(self.pairs)


def generate(n_containers: int, n_positions: int, seed: int = 1, payload_frac: float = 0.7, envelope: int = 1) -> Hold:
    """Positions with arms -(P-1), -(P-3), ..., P-1 (step 2), caps 2,600 kg in the middle falling linearly to
    1,400 kg at the ends; container weights 600-2,400 kg; payload limit `payload_frac` of the total container
    weight; CG envelope [-envelope, +envelope]."""
    rng = random.Random(seed)
    arms = [2 * j - (n_positions - 1) for j in range(n_positions)]
    caps = [2600 - 25 * round(48 * abs(a) / max(1, n_positions - 1)) for a in arms]
    weights = [rng.randrange(600, 2401, 50) for _ in range(n_containers)]
    pairs = [(i, j) for i in range(n_containers) for j in range(n_positions) if weights[i] <= caps[j]]
    return Hold(arms, caps, weights, int(payload_frac * sum(weights)), -envelope, envelope, pairs)


def paper_scale(qubits: int, seed: int = 1) -> Hold:
    """A hold whose model has as many variables as possible but at most `qubits` (the studied sizes were 12 to
    28 qubits): the (containers, positions) grid with N <= P whose allowed pairs come closest from below."""
    cands = [generate(nc, npos, seed=seed) for nc in range(2, 8) for npos in range(nc, 12)]
    return max((h for h in cands if h.n <= qubits), key=lambda h: (h.n, len(h.weights)))


def _rows(h: Hold):
    """Every row as (terms {var: coef}, lo, hi)."""
    rows = []
    for i in range(len(h.weights)):
        t = {k: 1 for k, (ci, _) in enumerate(h.pairs) if ci == i}
        if len(t) > 1:
            rows.append((t, None, 1))
    for j in range(len(h.arms)):
        t = {k: 1 for k, (_, pj) in enumerate(h.pairs) if pj == j}
        if len(t) > 1:
            rows.append((t, None, 1))
    rows.append(({k: h.weights[i] for k, (i, _) in enumerate(h.pairs)}, None, h.payload))
    rows.append(({k: h.weights[i] * (h.arms[j] - h.lo) for k, (i, j) in enumerate(h.pairs)}, 0, None))
    rows.append(({k: h.weights[i] * (h.arms[j] - h.hi) for k, (i, j) in enumerate(h.pairs)}, None, 0))
    return rows


def model(h: Hold):
    m = fourcolors.qubo.to_model({}, [h.weights[i] for i, _ in h.pairs], n=h.n, sense="max")
    for terms, lo, hi in _rows(h):
        terms = {k: v for k, v in terms.items() if v != 0}
        if terms:
            m.add_constraint(terms, lo=lo, hi=hi)
    return m


def decode(h: Hold, x) -> Dict[int, int]:
    """{container: position} for the chosen pairs (a container chosen twice keeps its first position; `check`
    reports it)."""
    out: Dict[int, int] = {}
    for k, xi in enumerate(x):
        if xi:
            i, j = h.pairs[k]
            out.setdefault(i, j)
    return out


def check(h: Hold, x) -> dict:
    """Verify a 0/1 vector in plain Python: one position per container, one container per position, structural
    caps, payload, and the center of gravity inside the envelope."""
    chosen = [h.pairs[k] for k, xi in enumerate(x) if xi]
    problems = []
    if len({i for i, _ in chosen}) < len(chosen):
        problems.append("a container in two positions")
    if len({j for _, j in chosen}) < len(chosen):
        problems.append("two containers in one position")
    if any(h.weights[i] > h.caps[j] for i, j in chosen):
        problems.append("structural cap exceeded")
    load = sum(h.weights[i] for i, _ in chosen)
    moment = sum(h.weights[i] * h.arms[j] for i, j in chosen)
    if load > h.payload:
        problems.append("payload exceeded")
    if not (h.lo * load <= moment <= h.hi * load):
        problems.append("center of gravity outside the envelope")
    return {"valid": not problems, "load_kg": load, "cg": (moment / load) if load else 0.0,
            "containers_loaded": len(chosen), "problems": problems}


def milp_reference(h: Hold, time_limit_s: float = 60.0) -> dict:
    """HiGHS (scipy.optimize.milp) on the same rows: a labeled exact reference."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    rows = _rows(h)
    A = lil_matrix((len(rows), h.n))
    lo = np.full(len(rows), -np.inf); hi = np.full(len(rows), np.inf)
    for r, (terms, l, u) in enumerate(rows):
        for k, v in terms.items():
            A[r, k] = v
        if l is not None:
            lo[r] = l
        if u is not None:
            hi[r] = u
    c = -np.array([h.weights[i] for i, _ in h.pairs], dtype=float)
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(h.n), bounds=Bounds(0, 1),
               options={"time_limit": time_limit_s, "disp": False})
    x = None if res.x is None else [int(round(v)) for v in res.x]
    return {"reference_value": None if x is None else int(round(-res.fun)), "optimal": res.status == 0,
            "reference_x": x, "reference_status": int(res.status)}


def brute_force(h: Hold) -> int:
    """The optimum by enumeration (for tests; n <= 20)."""
    assert h.n <= 20
    best = 0
    for x in itertools.product((0, 1), repeat=h.n):
        c = check(h, x)
        if c["valid"]:
            best = max(best, c["load_kg"])
    return best


def penalty_qubo(h: Hold, A: float = 2.0) -> Tuple[Dict[Tuple[int, int], float], List[float]]:
    """The textbook penalty form: minimize -sum w x / w_max + A * sum_rows (violation proxy)^2, with every row's
    left side squared against its bound and no slack variables (a common slack-free shape). Only its interaction
    graph and term count matter for the compile; the coefficients are scaled to O(1)."""
    wmax = max(h.weights)
    Q: Dict[Tuple[int, int], float] = {}
    lin = [-h.weights[i] / wmax for i, _ in h.pairs]

    def add_square(terms: Dict[int, float], target: float):
        ks = sorted(terms)
        for a in ks:
            lin[a] += A * (terms[a] ** 2 - 2 * target * terms[a])
        for a, b in itertools.combinations(ks, 2):
            Q[(a, b)] = Q.get((a, b), 0.0) + 2 * A * terms[a] * terms[b]
    for terms, lo, hi in _rows(h):
        scale = max(abs(v) for v in terms.values()) or 1
        t = {k: v / scale for k, v in terms.items() if v}
        bound = (hi if hi is not None else lo) / scale
        add_square(t, bound)
    return {k: v for k, v in Q.items() if abs(v) > 1e-12}, lin


def qaoa_circuit(h: Hold, p: int = 1, gamma: float = 0.4, beta: float = 0.3, measure: bool = True):
    """p layers of QAOA on the penalty form: H on every qubit, then per layer rzz on every interacting pair, rz on
    every qubit with a linear term, and an rx mixer. Fixed angles: only the gate structure is compiled."""
    from qiskit import QuantumCircuit

    Q, lin = penalty_qubo(h)
    n = h.n
    qc = QuantumCircuit(n, name=f"aircraft_qaoa_n{n}_p{p}")
    qc.h(range(n))
    for _ in range(p):
        for (a, b), v in sorted(Q.items()):
            qc.rzz(gamma * v / 2, a, b)
        for a in range(n):
            field = lin[a] / 2 + sum(v / 4 for (u, w), v in Q.items() if a in (u, w))
            if field:
                qc.rz(gamma * field, a)
        qc.rx(2 * beta, range(n))
    if measure:
        qc.measure_all()
    return qc
