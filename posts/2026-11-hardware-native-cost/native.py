"""What "hardware-native" costs: sparsify a problem until it fits the chip, or route the real one.

A hardware-native demo keeps only the problem's edges that land on the chip's couplers, so its QAOA circuit needs no
SWAPs. That is a sparsification of the problem. This module builds the three sparsifiers the post compares, all at
the same edge budget k (the number of edges the native placement keeps):

* native: place problem nodes on distinct device qubits so that as much edge weight as possible lands on working
  couplers (calibration-aware by default: an edge on a coupler counts |w| (1 - error)); keep exactly the edges that
  land on couplers. This is the reshaping step a hardware demo does by hand; it is not a compiler, and nothing
  here routes or compiles a circuit.
* random: k edges chosen uniformly at random.
* spectral: k edges sampled with probability proportional to |w_e| times the effective resistance R_e (Spielman and
  Srivastava's importance weights), without replacement; the kept edges keep their original weights.

Each sparsified instance is solved as Max-Cut and its answer is scored on the ORIGINAL graph: what fraction of the
full problem's cut (the optimizer's answer on the full graph) survives. Each one's QAOA circuit is compiled too, so
both sides of the trade-off (answer quality, two-qubit gates) sit on one chart.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int]


# ---- problems ------------------------------------------------------------------------------------

def problem(family: str, n: int, seed: int) -> Tuple[List[Edge], Dict[Edge, int]]:
    """(edges, integer weights): "regular3" (weights 1) or "weighted_er" (G(n, 6/n), weights 1..10)."""
    import examplekit

    if family == "regular3":
        edges = examplekit.random_regular(n, 3, seed)
        return edges, {e: 1 for e in edges}
    if family == "weighted_er":
        rnd = random.Random(seed + 7)
        edges = examplekit.gnp(n, 6.0 / n, seed)
        return edges, {e: rnd.randint(1, 10) for e in edges}
    raise ValueError(family)


PROBLEMS = [("regular3", 20, 20), ("regular3", 50, 50), ("regular3", 100, 100), ("regular3", 150, 150),
            ("weighted_er", 50, 50), ("weighted_er", 100, 100), ("weighted_er", 150, 150)]


def cut(edges: Sequence[Edge], weights: Dict[Edge, int], x: Sequence[int]) -> int:
    return int(sum(weights[e] for e in edges if x[e[0]] != x[e[1]]))


def qubo_model(n: int, edges: Sequence[Edge], weights: Dict[Edge, int]):
    """Max-Cut as a minimization QUBO over all n nodes (isolated nodes included): E(x) = -cut(x)."""
    import examplekit
    import fourcolors.qubo

    Q, lin = examplekit.maxcut_qubo(edges, {e: float(weights[e]) for e in edges})
    lin = list(lin) + [0.0] * (n - len(lin))
    return fourcolors.qubo.to_model(Q, lin, n=n)


# ---- the device ----------------------------------------------------------------------------------

def device_graph(target, gate: str = "cz") -> Tuple[int, Dict[Edge, float]]:
    """(n_qubits, {coupler: error}) for the WORKING couplers of a Qiskit Target (failed couplers dropped); the error
    is the worse of the two directions, 0 if none is published."""
    import circuitkit as ck

    bad = ck.failed_couplers(target, gate=gate)
    err: Dict[Edge, float] = {}
    for qargs, props in (target[gate] or {}).items():
        if qargs is None:
            continue
        e = tuple(sorted(int(q) for q in qargs))
        if e in bad:
            continue
        v = 0.0 if props is None or props.error is None else float(props.error)
        err[e] = max(err.get(e, 0.0), v)
    return target.num_qubits, err


# ---- the native placement ------------------------------------------------------------------------

def _edge_score(w: float, coupler_err: Optional[float], aware: bool) -> float:
    if coupler_err is None:
        return 0.0
    return abs(w) * ((1.0 - coupler_err) if aware else 1.0)


def placement_score(place: Sequence[int], edges, weights, couplers: Dict[Edge, float], aware: bool = True) -> float:
    return sum(_edge_score(weights[(a, b)], couplers.get(tuple(sorted((place[a], place[b])))), aware)
               for a, b in edges)


def native_placement(n: int, edges: Sequence[Edge], weights: Dict[Edge, int], n_qubits: int,
                     couplers: Dict[Edge, float], *, seed: int = 1, iters: int = 20000,
                     aware: bool = True) -> List[int]:
    """A placement (problem node -> distinct device qubit) that keeps much edge weight on working couplers: a BFS
    seed (problem BFS order onto device BFS order), then a local search of swaps and moves to free qubits, keeping
    any change that does not lower the score, with a short random walk at the start. Deterministic for a seed."""
    if n > n_qubits:
        raise ValueError("more problem nodes than device qubits")
    rnd = random.Random(seed)
    padj: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        padj[a].append(b)
        padj[b].append(a)
    dadj: List[List[int]] = [[] for _ in range(n_qubits)]
    for a, b in couplers:
        dadj[a].append(b)
        dadj[b].append(a)

    def bfs(adj, start, size):
        order, seen = [], set()
        for s in [start] + list(range(size)):
            if s in seen:
                continue
            queue = [s]
            seen.add(s)
            while queue:
                v = queue.pop(0)
                order.append(v)
                for u in sorted(adj[v]):
                    if u not in seen:
                        seen.add(u)
                        queue.append(u)
        return order

    pstart = max(range(n), key=lambda v: (len(padj[v]), -v))
    dstart = max(range(n_qubits), key=lambda q: (len(dadj[q]), -abs(q - n_qubits // 2)))
    porder = bfs(padj, pstart, n)
    dorder = bfs(dadj, dstart, n_qubits)
    place = [0] * n
    for v, q in zip(porder, dorder):
        place[v] = q
    owner = {q: v for v, q in enumerate(place)}

    def local(v, q):
        """Score of node v's edges if v sat on qubit q (other nodes where they are)."""
        return sum(_edge_score(weights[tuple(sorted((v, u)))], couplers.get(tuple(sorted((q, place[u])))), aware)
                   for u in padj[v] if place[u] != q)

    for it in range(iters):
        v = rnd.randrange(n)
        q = rnd.randrange(n_qubits)
        if q == place[v]:
            continue
        u = owner.get(q)
        qv = place[v]
        before = local(v, qv) + (local(u, q) if u is not None else 0.0)
        place[v] = q
        if u is not None:
            place[u] = qv
        after = local(v, q) + (local(u, qv) if u is not None else 0.0)
        if after + 1e-12 >= before or it < iters // 50:
            owner[q] = v
            if u is not None:
                owner[qv] = u
            else:
                owner.pop(qv, None)
        else:
            place[v] = qv
            if u is not None:
                place[u] = q
    return place


def native_edges(place: Sequence[int], edges: Sequence[Edge], couplers: Dict[Edge, float]) -> List[Edge]:
    """The problem edges whose two nodes sit on a working coupler."""
    return [e for e in edges if tuple(sorted((place[e[0]], place[e[1]]))) in couplers]


# ---- the two other sparsifiers -------------------------------------------------------------------

def random_edges(edges: Sequence[Edge], k: int, seed: int) -> List[Edge]:
    rnd = random.Random(seed)
    return sorted(rnd.sample(list(edges), k))


def effective_resistance(n: int, edges: Sequence[Edge], weights: Dict[Edge, int]) -> np.ndarray:
    """R_e for each edge, from the pseudo-inverse of the |w|-weighted Laplacian."""
    L = np.zeros((n, n))
    for a, b in edges:
        w = abs(float(weights[(a, b)]))
        L[a, a] += w; L[b, b] += w; L[a, b] -= w; L[b, a] -= w
    P = np.linalg.pinv(L)
    return np.array([P[a, a] + P[b, b] - 2 * P[a, b] for a, b in edges])


def spectral_edges(n: int, edges: Sequence[Edge], weights: Dict[Edge, int], k: int, seed: int) -> List[Edge]:
    """k edges sampled without replacement, probability proportional to |w_e| R_e."""
    r = effective_resistance(n, edges, weights)
    p = np.array([abs(float(weights[e])) for e in edges]) * np.clip(r, 1e-12, None)
    p = p / p.sum()
    idx = np.random.default_rng(seed).choice(len(edges), size=k, replace=False, p=p)
    return sorted(edges[i] for i in idx)


# ---- QAOA circuits -------------------------------------------------------------------------------

def qaoa_circuit(n_qubits: int, edges: Sequence[Edge], weights: Dict[Edge, int], p: int = 1,
                 qubit_of: Optional[Sequence[int]] = None, nodes: Optional[Sequence[int]] = None,
                 measure: bool = True):
    """QAOA for weighted Max-Cut (fixed linear-ramp angles): H on the problem's qubits, p layers of rzz(-gamma w_e)
    per edge and rx(2 beta). With `qubit_of`, problem node v acts on qubit qubit_of[v] of an n_qubits-wide circuit
    (the native circuit, already on device qubits); otherwise node v is qubit v."""
    from qiskit import QuantumCircuit

    nodes = list(range(n_qubits)) if nodes is None else list(nodes)
    q = (lambda v: v) if qubit_of is None else (lambda v: int(qubit_of[v]))  # noqa: E731
    qc = QuantumCircuit(n_qubits, name=f"qaoa_{len(edges)}e_p{p}")
    used = [q(v) for v in nodes]
    qc.h(used)
    for k in range(p):
        g, b = 0.6 * (k + 0.5) / p, 0.6 * (1 - (k + 0.5) / p)
        for a, c in edges:
            qc.rzz(-g * float(weights[(a, c)]), q(a), q(c))
        qc.rx(2 * b, used)
    if measure:
        from qiskit import ClassicalRegister

        creg = ClassicalRegister(len(used), "c")
        qc.add_register(creg)
        for i, u in enumerate(used):
            qc.measure(u, creg[i])
    return qc


def on_couplers(qc, couplers: Dict[Edge, float]) -> bool:
    """True when every two-qubit gate of `qc` (by its qubit indices) sits on a working coupler."""
    for inst in qc.data:
        if inst.operation.num_qubits == 2:
            a, b = (qc.find_bit(x).index for x in inst.qubits)
            if tuple(sorted((a, b))) not in couplers:
                return False
    return True
