"""A two-call pre-flight check: is this optimization instance worth QPU time?

Call 1, the 4colors Optimizer: one QUBO solve with its anytime trajectory. It gives the classical reference value
and a cheap hardness signal: how many seconds of service search time the optimizer needed to reach the value it
ended with, and whether 1 s and the full budget gave the same value.

Call 2, the 4colors Quantum Compiler: one compile of the instance's QAOA circuit for the device, with a time budget.
It gives the two-qubit gate count, the two-qubit depth and an estimated success probability (ESP): the product of
(1 - error) over every gate and measurement, from the device's published calibration. ESP is a model, never a
measurement.

The verdict puts the instance in one of four boxes, with lines the reader sets:

                       ESP >= esp_line (shallow)          ESP < esp_line (deep)
  t_final >  easy_s    "worth a look"                     "hard classically, too noisy to tell"
  t_final <= easy_s    "classically easy: a QPU can at    "classically easy and too noisy: a QPU run can
                         best confirm it"                   only confirm noise"

The instance families are Max-Cut and Ising-type problems of the kind QAOA and annealing papers use: random
3-regular graphs, Sherrington-Kirkpatrick (SK) models with +-1 couplings, square grids and sparse weighted random
graphs. They are generated here; none is a paper's own data.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

Edge = Tuple[int, int]

EASY_S = 1.0          # default "classically easy" line: final value reached within 1 s of service search time
ESP_LINE = 0.1        # default "shallow enough" line on the modeled ESP


# ---- instances -----------------------------------------------------------------------------------

def regular3(n: int, seed: int) -> Tuple[List[Edge], Dict[Edge, int]]:
    import examplekit

    edges = examplekit.random_regular(n, 3, seed)
    return edges, {e: 1 for e in edges}


def sk(n: int, seed: int) -> Tuple[List[Edge], Dict[Edge, int]]:
    """Sherrington-Kirkpatrick: every pair coupled, weight +1 or -1 at random."""
    rnd = random.Random(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return edges, {e: rnd.choice((-1, 1)) for e in edges}


def grid(side: int, seed: int) -> Tuple[List[Edge], Dict[Edge, int]]:
    """A side x side square grid with +-1 weights (a planar Ising spin glass)."""
    rnd = random.Random(seed)
    idx = lambda r, c: r * side + c          # noqa: E731
    edges = sorted([(idx(r, c), idx(r, c + 1)) for r in range(side) for c in range(side - 1)] +
                   [(idx(r, c), idx(r + 1, c)) for r in range(side - 1) for c in range(side)])
    return edges, {e: rnd.choice((-1, 1)) for e in edges}


def weighted_er(n: int, p: float, seed: int) -> Tuple[List[Edge], Dict[Edge, int]]:
    """G(n, p) with integer weights 1..10."""
    import examplekit

    rnd = random.Random(seed + 104729)
    edges = examplekit.gnp(n, p, seed)
    return edges, {e: rnd.randint(1, 10) for e in edges}


def instances() -> List[dict]:
    """The post's 30 instances: {id, family, n, edges, weights}. Sizes stay within one 156-qubit device."""
    out = []
    for n in (16, 24, 32, 48, 64, 80, 96, 112, 128, 150):
        e, w = regular3(n, seed=n)
        out.append({"id": f"reg3_n{n}", "family": "regular3", "n": n, "edges": e, "weights": w})
    for n in (10, 14, 18, 22, 26, 30, 40, 50):
        e, w = sk(n, seed=n)
        out.append({"id": f"sk_n{n}", "family": "sk", "n": n, "edges": e, "weights": w})
    for side in (4, 6, 8, 10, 12):
        e, w = grid(side, seed=side)
        out.append({"id": f"grid_{side}x{side}", "family": "grid", "n": side * side, "edges": e, "weights": w})
    for n in (20, 40, 60, 80, 100, 120, 150):
        e, w = weighted_er(n, 4.0 / n, seed=n)
        out.append({"id": f"er_n{n}", "family": "weighted_er", "n": n, "edges": e, "weights": w})
    return out


# ---- the two models ------------------------------------------------------------------------------

def qubo_model(n: int, edges: Sequence[Edge], weights: Dict[Edge, int]):
    """Max-Cut as a minimization QUBO, E(x) = -cut(x) (examplekit.maxcut_qubo)."""
    import examplekit
    import fourcolors.qubo

    Q, lin = examplekit.maxcut_qubo(edges, {e: float(weights[e]) for e in edges})
    lin = list(lin) + [0.0] * (n - len(lin))
    return fourcolors.qubo.to_model(Q, lin, n=n)


def cut(edges: Sequence[Edge], weights: Dict[Edge, int], x: Sequence[int]) -> int:
    return int(sum(weights[e] for e in edges if x[e[0]] != x[e[1]]))


def qaoa_circuit(n: int, edges: Sequence[Edge], weights: Dict[Edge, int], p: int = 1, measure: bool = True):
    """QAOA for weighted Max-Cut: H on every qubit, then p layers of rzz(-gamma w_e) per edge and rx(2 beta).
    Fixed linear-ramp angles: the check compiles the circuit, it does not optimize or run it."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, name=f"qaoa_n{n}_p{p}")
    qc.h(range(n))
    for k in range(p):
        g, b = 0.6 * (k + 0.5) / p, 0.6 * (1 - (k + 0.5) / p)
        for a, c in edges:
            qc.rzz(-g * float(weights[(a, c)]), a, c)
        qc.rx(2 * b, range(n))
    if measure:
        qc.measure_all()
    return qc


# ---- the two signals and the verdict -------------------------------------------------------------

def classical_signal(trajectory: Sequence[Sequence[float]], final: Optional[float], *, budget_s: float,
                     early_s: float = EASY_S) -> dict:
    """From an anytime trajectory [[t_search_s, objective], ...] (minimization) and the final objective:
    t_final_s, the service search time of the first point at the final value (None if the trajectory never shows
    it), the value at `early_s`, and whether it already equals the final value. Service time, not wall time."""
    pts = sorted((float(t), float(v)) for t, v in trajectory if t is not None and v is not None)
    if final is None:
        return {"t_final_s": None, "value_at_early": None, "early_equals_final": None, "budget_s": budget_s,
                "trajectory_points": len(pts)}
    t_final = next((t for t, v in pts if v <= final), None)
    early = [v for t, v in pts if t <= early_s]
    v_early = min(early) if early else None
    return {"t_final_s": t_final, "value_at_early": v_early, "early_s": early_s,
            "early_equals_final": None if v_early is None else v_early <= final,
            "budget_s": budget_s, "trajectory_points": len(pts)}


def verdict(t_final_s: Optional[float], esp: Optional[float], *, easy_s: float = EASY_S,
            esp_line: float = ESP_LINE) -> str:
    """One of four boxes (module docstring), or "unknown" when a signal is missing."""
    if t_final_s is None or esp is None:
        return "unknown"
    easy = t_final_s <= easy_s
    shallow = esp >= esp_line
    if easy and shallow:
        return "classically easy: a QPU can at best confirm it"
    if easy:
        return "classically easy and too noisy: a QPU run can only confirm noise"
    if shallow:
        return "worth a look"
    return "hard classically at this budget, too noisy to tell"


def crosses_line(esp_4colors: Optional[float], esp_other: Optional[float], esp_line: float = ESP_LINE) -> bool:
    """True when the 4colors compile is on the shallow side of the ESP line and the other compile is not."""
    return (esp_4colors is not None and esp_other is not None and esp_4colors >= esp_line > esp_other)
