"""EV smart charging as a maximum-weight independent set, with exact rows.

The problem, in the shape studied by Dalyac et al. (EPJ Quantum Technology 8, 12, 2021) and run on neutral atoms by
Leclerc et al. (Phys. Rev. A 111, 032611, 2025) [CHECK the model details against both papers]:

* electric vehicles arrive at a depot with an arrival slot, a departure slot, a charge they need (a number of
  slots at a charger's power) and a priority;
* a *task* is one way to serve one vehicle: (vehicle, charger, start slot), running `length` slots inside the
  vehicle's window;
* two tasks conflict when they serve the same vehicle, or use the same charger in an overlapping slot;
* pick a set of tasks with no conflict that maximizes the delivered value (priority x energy).

That is a maximum-weight independent set (MWIS) on the conflict graph. Here it is written for the 4colors Optimizer
for QUBO as a linear objective plus exact rows: one "at most one task" row per vehicle and one per (charger, slot).
Those rows are cliques of the conflict graph, so they say exactly what the pairwise edges say, with far fewer rows.
`pairwise_model` writes the textbook form (one row per conflict edge) for comparison, and `penalty_bqm` the
penalty form a QUBO sampler without rows needs (each conflict edge costs more than any task is worth).

An optional site power cap (`site_kw`) adds one row per slot: the chargers running in a slot may draw at most the
site's limit. It is a knapsack-type row, not a conflict edge, and goes in the same way.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import fourcolors.qubo


@dataclass
class Instance:
    n_slots: int
    chargers_kw: List[int]                      # power of each charger, kW (integers)
    vehicles: List[dict]                        # arrival, departure, length, priority
    tasks: List[Tuple[int, int, int]] = field(default_factory=list)      # (vehicle, charger, start)
    site_kw: Optional[int] = None

    def task_slots(self, t: int) -> range:
        v, c, s = self.tasks[t]
        return range(s, s + self.vehicles[v]["length"])

    def weight(self, t: int) -> int:
        """Delivered value: priority x energy (kW x slots)."""
        v, c, s = self.tasks[t]
        return self.vehicles[v]["priority"] * self.chargers_kw[c] * self.vehicles[v]["length"]


def generate(n_vehicles: int, n_chargers: int, n_slots: int = 48, seed: int = 1, starts_per_vehicle: int = 4,
             chargers_per_vehicle: int = 3, site_kw: Optional[int] = None) -> Instance:
    """A synthetic depot: windows of 4-16 slots (a slot is 30 min), needs of 2-6 slots, priorities 1-3, chargers of
    7, 11 or 22 kW. Each vehicle gets up to `starts_per_vehicle` start slots on up to `chargers_per_vehicle`
    chargers, so the tasks grow linearly with the vehicles (about 12 per vehicle)."""
    rng = random.Random(seed)
    chargers = [rng.choice([7, 11, 22]) for _ in range(n_chargers)]
    vehicles = []
    for _ in range(n_vehicles):
        length = rng.randint(2, 6)
        window = rng.randint(max(length, 4), min(16, n_slots))
        arr = rng.randint(0, n_slots - window)
        vehicles.append({"arrival": arr, "departure": arr + window, "length": length, "priority": rng.randint(1, 3)})
    tasks = []
    for v, veh in enumerate(vehicles):
        starts = list(range(veh["arrival"], veh["departure"] - veh["length"] + 1))
        rng.shuffle(starts)
        cs = rng.sample(range(n_chargers), min(chargers_per_vehicle, n_chargers))
        for s in sorted(starts[:starts_per_vehicle]):
            for c in sorted(cs):
                tasks.append((v, c, s))
    return Instance(n_slots, chargers, vehicles, tasks, site_kw)


def for_tasks(n_tasks: int, seed: int = 1, **kw) -> Instance:
    """An instance of about `n_tasks` tasks: vehicles = n_tasks / 12, one charger per 4 vehicles (at least 2)."""
    nv = max(2, round(n_tasks / 12))
    return generate(nv, max(2, nv // 4), seed=seed, **kw)


def clique_groups(inst: Instance) -> List[List[int]]:
    """The 'at most one' groups: tasks of one vehicle; tasks using one charger in one slot."""
    groups: Dict[tuple, List[int]] = {}
    for t, (v, c, s) in enumerate(inst.tasks):
        groups.setdefault(("v", v), []).append(t)
        for slot in inst.task_slots(t):
            groups.setdefault(("c", c, slot), []).append(t)
    return [g for g in groups.values() if len(g) > 1]


def conflict_edges(inst: Instance) -> List[Tuple[int, int]]:
    """The conflict graph's edges (pairs inside any clique group), sorted, each once."""
    edges = set()
    for g in clique_groups(inst):
        edges.update(itertools.combinations(sorted(g), 2))
    return sorted(edges)


def _power_rows(inst: Instance, m):
    if inst.site_kw is None:
        return
    for slot in range(inst.n_slots):
        terms = {t: inst.chargers_kw[inst.tasks[t][1]] for t in range(len(inst.tasks)) if slot in inst.task_slots(t)}
        if terms and sum(terms.values()) > inst.site_kw:
            m.add_constraint(terms, hi=inst.site_kw)


def model(inst: Instance):
    """Maximize sum weight x; one exact row (sum x <= 1) per clique group; the power cap if set."""
    n = len(inst.tasks)
    m = fourcolors.qubo.to_model({}, [inst.weight(t) for t in range(n)], n=n, sense="max")
    for g in clique_groups(inst):
        m.add_constraint({t: 1 for t in g}, hi=1)
    _power_rows(inst, m)
    return m


def pairwise_model(inst: Instance):
    """The textbook MWIS form: one row x_u + x_v <= 1 per conflict edge."""
    n = len(inst.tasks)
    m = fourcolors.qubo.to_model({}, [inst.weight(t) for t in range(n)], n=n, sense="max")
    for u, v in conflict_edges(inst):
        m.add_constraint({u: 1, v: 1}, hi=1)
    _power_rows(inst, m)
    return m


def penalty_bqm(inst: Instance):
    """(linear, quadratic, P): minimize -sum w x + P sum_edges x_u x_v with P = max weight + 1, the penalty form a
    sampler without rows needs. Any P above the largest weight keeps the optimum (removing one end of a violated
    edge never loses value). The power cap has no exact penalty form here and is not included."""
    n = len(inst.tasks)
    w = [inst.weight(t) for t in range(n)]
    P = max(w) + 1
    return {t: -w[t] for t in range(n)}, {e: P for e in conflict_edges(inst)}, P


def decode(inst: Instance, x) -> List[Tuple[int, int, int]]:
    return [inst.tasks[t] for t in range(len(inst.tasks)) if x[t]]


def check(inst: Instance, schedule: List[Tuple[int, int, int]]) -> dict:
    """Verify a schedule in plain Python, with no reference to the model: each vehicle at most once, inside its
    window; no charger used twice in a slot; the site cap in every slot. Returns validity and the value."""
    seen_v, busy, problems = set(), set(), []
    draw = [0] * inst.n_slots
    value = 0
    for v, c, s in schedule:
        veh = inst.vehicles[v]
        if v in seen_v:
            problems.append(f"vehicle {v} twice")
        seen_v.add(v)
        if s < veh["arrival"] or s + veh["length"] > veh["departure"]:
            problems.append(f"vehicle {v} outside its window")
        for slot in range(s, s + veh["length"]):
            if (c, slot) in busy:
                problems.append(f"charger {c} double-booked at slot {slot}")
            busy.add((c, slot))
            if 0 <= slot < inst.n_slots:
                draw[slot] += inst.chargers_kw[c]
            else:
                problems.append(f"slot {slot} outside the day")
        value += veh["priority"] * inst.chargers_kw[c] * veh["length"]
    if inst.site_kw is not None and max(draw, default=0) > inst.site_kw:
        problems.append("site power cap exceeded")
    return {"valid": not problems, "value": value, "served": len(seen_v), "problems": problems[:5],
            "peak_kw": max(draw, default=0)}


def milp_reference(inst: Instance, time_limit_s: float = 60.0) -> dict:
    """An exact reference with HiGHS (through scipy.optimize.milp): the optimum when it closes in time, else its
    best value and proven upper bound. A labeled reference the post compares against, not a rival."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n = len(inst.tasks)
    groups = clique_groups(inst)
    rows = [(g, {t: 1 for t in g}, 1) for g in groups]
    if inst.site_kw is not None:
        for slot in range(inst.n_slots):
            terms = {t: inst.chargers_kw[inst.tasks[t][1]] for t in range(n) if slot in inst.task_slots(t)}
            if terms and sum(terms.values()) > inst.site_kw:
                rows.append((None, terms, inst.site_kw))
    A = lil_matrix((max(len(rows), 1), n))
    ub = np.zeros(max(len(rows), 1))
    for i, (_, terms, hi) in enumerate(rows):
        for t, a in terms.items():
            A[i, t] = a
        ub[i] = hi
    c = -np.array([inst.weight(t) for t in range(n)], dtype=float)
    res = milp(c, constraints=LinearConstraint(A.tocsr(), -np.inf, ub), integrality=np.ones(n),
               bounds=Bounds(0, 1), options={"time_limit": time_limit_s, "disp": False})
    best = None if res.x is None else int(round(-res.fun))
    bound = getattr(res, "mip_dual_bound", None)
    bound = None if bound is None or not np.isfinite(bound) else -float(bound)
    return {"reference_value": best, "upper_bound": bound, "optimal": res.status == 0,
            "reference_status": int(res.status), "reference_message": str(res.message)[:200],
            "reference_x": None if res.x is None else [int(round(v)) for v in res.x]}
