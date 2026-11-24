"""Domain observables from output distributions. A distribution is {bitstring: probability or count} with
QUBIT 0 FIRST (`from_counts` turns Qiskit's count keys, clbit 0 last, into that), so bit j is logical qubit j.

Each `measure(spec, dist)` returns the headline value and its extras; `error(spec, noisy, ideal)` is the post's
error for the family (the number charted), and `standard_error` the shot noise on it.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]

import domain_circuits as circuits


def from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    """Qiskit counts (clbit 0 last, register spaces) as probabilities keyed qubit 0 first."""
    total = sum(counts.values())
    out: Dict[str, float] = {}
    for k, v in counts.items():
        key = k.replace(" ", "")[::-1]
        out[key] = out.get(key, 0.0) + v / total
    return out


def occupations(dist: Dict[str, float], n: int) -> List[float]:
    return [sum(p for k, p in dist.items() if k[j] == "1") for j in range(n)]


def sector_probability(dist: Dict[str, float], n_spatial: int, n_electrons: int) -> float:
    """Probability of the right particle number and spin: n_electrons / 2 ones in each spin block."""
    half = n_electrons // 2
    return sum(p for k, p in dist.items() if k[:n_spatial].count("1") == half and k[n_spatial:].count("1") == half)


def feasible_probability(dist: Dict[str, float], k: int) -> float:
    return sum(p for key, p in dist.items() if key.count("1") == k)


def feasible_mean_cost(dist: Dict[str, float], k: int, mu, sigma) -> Optional[float]:
    feas = {key: p for key, p in dist.items() if key.count("1") == k}
    tot = sum(feas.values())
    if tot <= 0:
        return None
    return sum(p * circuits.portfolio_cost([int(b) for b in key], mu, sigma) for key, p in feas.items()) / tot


def magnetization(dist: Dict[str, float]) -> float:
    """Mean <Z> over the qubits: +1 for |0>, -1 for |1>."""
    n = len(next(iter(dist)))
    return sum(p * (n - 2 * key.count("1")) / n for key, p in dist.items())


def measure(spec: dict, dist: Dict[str, float]) -> dict:
    """The family's observable(s) on one distribution."""
    obs = spec["observable"]
    if obs == "occupations":
        n = 2 * spec["n_spatial"]
        return {"occupations": occupations(dist, n),
                "sector_probability": sector_probability(dist, spec["n_spatial"], spec["n_electrons"])}
    if obs == "feasibility":
        mu, sigma = circuits.portfolio_data(spec["n"], spec["seed"])
        return {"feasible_probability": feasible_probability(dist, spec["k"]),
                "feasible_mean_cost": feasible_mean_cost(dist, spec["k"], mu, sigma)}
    if obs == "magnetization":
        return {"magnetization": magnetization(dist)}
    raise KeyError(obs)


def error(spec: dict, noisy: dict, ideal: dict) -> float:
    """The charted error: largest occupation error; 1 - P(feasible) relative to the ideal; |m - m_ideal|."""
    obs = spec["observable"]
    if obs == "occupations":
        return max(abs(a - b) for a, b in zip(noisy["occupations"], ideal["occupations"]))
    if obs == "feasibility":
        return abs(ideal["feasible_probability"] - noisy["feasible_probability"])
    return abs(noisy["magnetization"] - ideal["magnetization"])


def standard_error(spec: dict, noisy: dict, shots: int) -> float:
    """Shot noise on the charted error (binomial for a probability; the largest occupation's binomial SE;
    a bound of 1/sqrt(shots) for the magnetization, whose single-shot values lie in [-1, 1])."""
    obs = spec["observable"]
    if obs == "occupations":
        return max(math.sqrt(p * (1 - p) / shots) for p in noisy["occupations"])
    if obs == "feasibility":
        p = noisy["feasible_probability"]
        return math.sqrt(p * (1 - p) / shots)
    return 1 / math.sqrt(shots)
