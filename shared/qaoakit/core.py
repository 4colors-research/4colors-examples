"""QAOA blog harness: instances, QAOA variant circuits, angle tuning and noisy scoring.

Bit order everywhere is Qiskit's: bit i of a basis-state index is qubit (or clbit) i.
"""
import math

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Both compilers get the same input circuit, written in these gates.
INPUT_BASIS = ["h", "x", "sx", "rx", "ry", "rz", "cx", "rzz"]
DEVICE_BASIS = ["cz", "rz", "sx", "x"]
VARIANTS = ["std-p1", "std-p2", "std-p3", "ma-p1", "warm-p1", "xy-p1", "xy-p2"]


# ---------------------------------------------------------------- instances

def instances():
    """21 instances: 15 MaxCut graphs (unweighted 3-regular, weighted G(n, 0.3)) and 6 max-3-cut graphs."""
    out = []
    for n in (12, 16, 20):
        for s in (1, 2, 3):
            g = nx.random_regular_graph(3, n, seed=s)
            out.append(dict(id=f"reg3-n{n}-s{s}", kind="maxcut", nodes=n,
                            edges=[(int(a), int(b), 1.0) for a, b in sorted(g.edges)]))
    for n in (12, 16, 20):
        for s in (1, 2):
            g = nx.gnp_random_graph(n, 0.3, seed=100 + s)
            rng = np.random.default_rng(100 + s)
            out.append(dict(id=f"gnp-n{n}-s{s}", kind="maxcut", nodes=n,
                            edges=[(int(a), int(b), round(float(rng.uniform(0.1, 1.0)), 3))
                                   for a, b in sorted(g.edges)]))
    for n in (4, 5, 6):
        for s in (1, 2):
            g = nx.gnp_random_graph(n, 0.6, seed=200 + s)
            out.append(dict(id=f"m3c-n{n}-s{s}", kind="max3cut", nodes=n,
                            edges=[(int(a), int(b), 1.0) for a, b in sorted(g.edges)]))
    return out


def n_qubits(inst):
    return inst["nodes"] * (3 if inst["kind"] == "max3cut" else 1)


def cost_vector(inst):
    """Score of every basis state: the cut weight; for max-3-cut, 0 when any node is not one-hot."""
    nq = n_qubits(inst)
    idx = np.arange(2 ** nq, dtype=np.int64)
    bit = lambda q: (idx >> q) & 1
    cost = np.zeros(2 ** nq)
    if inst["kind"] == "maxcut":
        for a, b, w in inst["edges"]:
            cost += w * (bit(a) ^ bit(b))
        return cost
    feasible = np.ones(2 ** nq, dtype=bool)
    for v in range(inst["nodes"]):
        feasible &= (bit(3 * v) + bit(3 * v + 1) + bit(3 * v + 2)) == 1
    for a, b, w in inst["edges"]:
        same = sum(bit(3 * a + c) & bit(3 * b + c) for c in range(3))
        cost += w * (1 - same)
    return np.where(feasible, cost, 0.0)


def local_optimum(inst, seed=0):
    """A 1-flip local optimum of MaxCut from a random start: the warm-start solution."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, inst["nodes"])
    nbr = [[] for _ in range(inst["nodes"])]
    for a, b, w in inst["edges"]:
        nbr[a].append((b, w))
        nbr[b].append((a, w))
    improved = True
    while improved:
        improved = False
        for v in range(inst["nodes"]):
            gain = sum(w * (1 if x[u] == x[v] else -1) for u, w in nbr[v])
            if gain > 1e-12:
                x[v] ^= 1
                improved = True
    return [int(b) for b in x]


# ---------------------------------------------------------------- circuits

def _cost_layer(qc, inst, gamma):
    """e^{-i gamma C}, up to a global phase (and, for max-3-cut, a phase constant on one-hot states)."""
    if inst["kind"] == "maxcut":
        for k, (a, b, w) in enumerate(inst["edges"]):
            g = gamma[k] if isinstance(gamma, (list, tuple)) else gamma
            qc.rzz(-g * w, a, b)
    else:
        for a, b, w in inst["edges"]:
            for c in range(3):
                qc.rzz(-gamma * w / 2, 3 * a + c, 3 * b + c)


def _w3(qc, q0, q1, q2):
    """(|100> + |010> + |001>) / sqrt(3) on three qubits."""
    qc.ry(2 * math.asin(1 / math.sqrt(3)), q0)
    qc.x(q0)
    qc.cry(math.pi / 2, q0, q1)
    qc.x(q0)
    qc.x(q2)
    qc.cx(q0, q2)
    qc.cx(q1, q2)


def build(inst, variant, x0=None, eps=0.25):
    """Parametrized circuit (no measurements) and its parameter vector."""
    nq = n_qubits(inst)
    qc = QuantumCircuit(nq)
    kind, p = variant.split("-p")[0], int(variant.split("-p")[1])
    if kind == "std":
        th = ParameterVector("t", 2 * p)
        qc.h(range(nq))
        for k in range(p):
            _cost_layer(qc, inst, th[2 * k])
            qc.rx(2 * th[2 * k + 1], range(nq))
    elif kind == "ma":
        m = len(inst["edges"])
        th = ParameterVector("t", m + nq)
        qc.h(range(nq))
        _cost_layer(qc, inst, [th[k] for k in range(m)])
        for v in range(nq):
            qc.rx(2 * th[m + v], v)
    elif kind == "warm":
        th = ParameterVector("t", 2 * p)
        theta = [2 * math.asin(math.sqrt(1 - eps if b else eps)) for b in x0]
        for v in range(nq):
            qc.ry(theta[v], v)
        for k in range(p):
            _cost_layer(qc, inst, th[2 * k])
            for v in range(nq):
                qc.ry(-theta[v], v)
                qc.rz(-2 * th[2 * k + 1], v)
                qc.ry(theta[v], v)
    elif kind == "xy":
        th = ParameterVector("t", 2 * p)
        for v in range(inst["nodes"]):
            _w3(qc, 3 * v, 3 * v + 1, 3 * v + 2)
        for k in range(p):
            _cost_layer(qc, inst, th[2 * k])
            for v in range(inst["nodes"]):
                for a, b in ((0, 1), (1, 2), (0, 2)):
                    qc.append(XXPlusYYGate(2 * th[2 * k + 1]), [3 * v + a, 3 * v + b])
    else:
        raise ValueError(variant)
    return qc, th


def applies(inst, variant):
    return variant.startswith("xy") == (inst["kind"] == "max3cut")


def input_circuit(qc_bound):
    """The circuit both compilers receive: INPUT_BASIS gates plus measure_all (clbit k = qubit k)."""
    qc = transpile(qc_bound, basis_gates=INPUT_BASIS, optimization_level=0)
    qc.measure_all()
    return qc


# ---------------------------------------------------------------- simulation

def _sim(noise=None, threads=0):
    return AerSimulator(method="statevector", noise_model=noise, max_parallel_threads=threads)


def ideal_probs(qc_bound, sim=None):
    sim = sim or _sim()
    qc = transpile(qc_bound, sim, optimization_level=0)
    qc.save_probabilities()
    return sim.run(qc, shots=1).result().data()["probabilities"]


def tune_circuit(qc, params, cost, restarts=6, seed=0, init=None, maxiter=1500, lo=0.0, hi=math.pi,
                 sign=+1.0):
    """COBYLA on the noiseless expected score of any parametric circuit (no measurements); maximizes
    sign * <cost>. Returns (best angles, best expectation, number of circuit evaluations)."""
    qc = qc.copy()
    qc.save_probabilities()
    sim = _sim(threads=1)
    tqc = transpile(qc, sim, optimization_level=0)
    calls = [0]

    def neg(x):
        calls[0] += 1
        bound = tqc.assign_parameters(dict(zip(params, x)))
        return -sign * float(sim.run(bound, shots=1).result().data()["probabilities"] @ cost)

    rng = np.random.default_rng(seed)
    starts = [np.asarray(init, float)] if init is not None else []
    while len(starts) < restarts:
        starts.append(rng.uniform(lo, hi, len(params)))
    best = None
    for x in starts:
        r = minimize(neg, x, method="COBYLA", options=dict(maxiter=maxiter, rhobeg=0.3))
        if best is None or r.fun < best.fun:
            best = r
    return [float(v) for v in best.x], -sign * float(best.fun), calls[0]


def tune(inst, variant, cost, x0=None, restarts=6, seed=0, init=None, maxiter=1500):
    """Maximize the noiseless expected score; returns (best angles, best expectation)."""
    qc, th = build(inst, variant, x0)
    angles, e, _ = tune_circuit(qc, th, cost, restarts=restarts, seed=seed, init=init, maxiter=maxiter)
    return angles, e


def noise_model(e2, e1=None):
    nm = NoiseModel(basis_gates=DEVICE_BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(e2, 2), ["cz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(e2 / 10 if e1 is None else e1, 1), ["sx", "x"])
    return nm


def compact(compiled, n_clbits):
    """Drop idle physical qubits and measurements; return (circuit, compact qubit of each clbit)."""
    phys_of_clbit = {}
    used = set()
    for ins in compiled.data:
        qs = [compiled.find_bit(q).index for q in ins.qubits]
        if ins.operation.name == "measure":
            phys_of_clbit[compiled.find_bit(ins.clbits[0]).index] = qs[0]
        if ins.operation.name not in ("barrier", "delay"):
            used.update(qs)
    order = sorted(used)
    pos = {q: i for i, q in enumerate(order)}
    out = QuantumCircuit(len(order))
    for ins in compiled.data:
        if ins.operation.name in ("measure", "barrier", "delay"):
            continue
        out.append(ins.operation, [pos[compiled.find_bit(q).index] for q in ins.qubits])
    return out, [pos[phys_of_clbit[k]] for k in range(n_clbits)]


def readout(probs, n, e_ro):
    """Symmetric readout flips with probability e_ro on every bit."""
    if not e_ro:
        return np.asarray(probs)
    m = np.array([[1 - e_ro, e_ro], [e_ro, 1 - e_ro]])
    t = np.asarray(probs).reshape([2] * n)
    for ax in range(n):
        t = np.moveaxis(np.tensordot(m, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def noisy_probs(compiled, n_clbits, e2, e_ro=0.01, trajectories=300, seed=1, threads=0):
    """Output distribution over the measured bits (clbit k = bit k) of a compiled circuit under depolarizing
    gate noise (e2 on CZ, e2/10 on SX and X) and symmetric readout error e_ro. Averaged over `trajectories`
    noisy statevector runs (exact when e2 == 0). Returns (probabilities, active qubits)."""
    qc, meas = compact(compiled, n_clbits)
    qc.save_probabilities(qubits=meas)
    noisy = e2 > 0
    sim = _sim(noise_model(e2) if noisy else None, threads)
    res = sim.run(qc, shots=trajectories if noisy else 1, seed_simulator=seed).result()
    return readout(res.data()["probabilities"], n_clbits, e_ro), qc.num_qubits


def noisy_expectation(compiled, n_clbits, cost, e2, e_ro=0.01, trajectories=300, seed=1, threads=0):
    """Expected score of a compiled circuit under depolarizing gate noise and readout error.

    Probabilities are averaged over `trajectories` noisy statevector runs (exact when e2 == 0).
    """
    probs, active = noisy_probs(compiled, n_clbits, e2, e_ro, trajectories, seed, threads)
    return float(probs @ cost), active


def noisy_ratio(compiled, n_clbits, cost, opt, e2, e_ro=0.01, trajectories=100, batches=5, threads=1):
    """Noisy approximation ratio <cost>/opt: mean and standard error over `batches` independent batches of
    `trajectories` trajectories (seeds 1..batches). Also the batch-averaged distribution, for other metrics."""
    ars, acc = [], None
    for b in range(batches):
        probs, active = noisy_probs(compiled, n_clbits, e2, e_ro, trajectories, seed=b + 1, threads=threads)
        ars.append(float(probs @ cost) / opt)
        acc = probs if acc is None else acc + probs
    mean = sum(ars) / batches
    se = (sum((x - mean) ** 2 for x in ars) / (batches - 1) / batches) ** 0.5 if batches > 1 else None
    return dict(mean=mean, se=se, active_qubits=active), acc / batches


def two_qubit_stats(qc):
    ops = qc.count_ops()
    n2 = sum(1 for ins in qc.data if len(ins.qubits) == 2 and ins.operation.name not in ("barrier",))
    d2 = qc.depth(lambda ins: len(ins.qubits) == 2 and ins.operation.name != "barrier")
    return dict(two_qubit_gates=n2, two_qubit_depth=d2, depth=qc.depth(), ops={k: int(v) for k, v in ops.items()})


# ---------------------------------------------------------------- fixed angles, fast p-layer simulator

# Tree-limit fixed angles for 3-regular Max-Cut (Wurtz and Lykov, "Fixed-angle conjectures for the quantum
# approximate optimization algorithm on regular MaxCut graphs", PRA 104, 052419 (2021), arXiv:2107.00677,
# Table I), in this file's convention: e^{-i beta B} e^{-i gamma C}, C = sum_edges (1 - Z Z)/2, B = sum X.
# Checked here, not only transcribed: p = 1 is the closed-form optimum (tan gamma = 1/sqrt 2, beta = pi/8,
# cut fraction 0.6925 on the tree); p = 2 is checked on the 14-node tree light cone in the tests (0.7559).
# p = 3 (tree value 0.7924) matches Table I as printed (gamma 0.422, 0.798, 0.937; beta 0.609, 0.459, 0.235;
# read from the arXiv PDF on 2 Nov 2026); its 30-node light cone is too large to check here.
FIXED_ANGLES_3REG = {
    1: ([0.6155], [0.3927]),
    2: ([0.4877, 0.8979], [0.5550, 0.2921]),
    3: ([0.4220, 0.7979, 0.9372], [0.6089, 0.4598, 0.2357]),
}


def fixed_angles(p):
    """FIXED_ANGLES_3REG[p] in build()'s order (gamma_1, beta_1, gamma_2, beta_2, ...)."""
    g, b = FIXED_ANGLES_3REG[p]
    return [x for pair in zip(g, b) for x in pair]


def qaoa_statevector(n, cost, angles, x0=None, eps=0.25):
    """Standard (or warm-start) QAOA state as a numpy vector, without a circuit: phases e^{-i gamma C} on the
    cost vector, then the mixer on every qubit. Same convention as build(); fast enough to tune n <= 20."""
    if x0 is None:
        psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
        thetas = None
    else:                                          # prod_v ry(theta_v)|0>; np.kron puts qubit v above 0..v-1
        thetas = warm_thetas(x0, eps)
        psi = np.array([1.0 + 0j])
        for v in range(n):
            psi = np.kron(np.array([math.cos(thetas[v] / 2), math.sin(thetas[v] / 2)]), psi)
    psi = psi.reshape([2] * n)                     # axis k is qubit n-1-k
    for k in range(len(angles) // 2):
        g, b = angles[2 * k], angles[2 * k + 1]
        psi = psi * np.exp(-1j * g * cost).reshape([2] * n)
        for v in range(n):
            ax = n - 1 - v
            if thetas is None:                     # rx(2 beta) = cos b I - i sin b X
                m = np.array([[math.cos(b), -1j * math.sin(b)], [-1j * math.sin(b), math.cos(b)]])
            else:                                  # ry(theta) rz(-2 beta) ry(-theta)
                m = _ry(thetas[v]) @ np.diag([np.exp(1j * b), np.exp(-1j * b)]) @ _ry(-thetas[v])
            psi = np.moveaxis(np.tensordot(m, psi, axes=([1], [ax])), 0, ax)
    return psi.reshape(-1)


def _ry(a):
    return np.array([[math.cos(a / 2), -math.sin(a / 2)], [math.sin(a / 2), math.cos(a / 2)]])


def tune_numpy(n, cost, p, restarts=4, seed=0, init=None, maxiter=400, x0=None):
    """Angles maximizing the noiseless <cost> of p-layer QAOA (numpy simulator). Returns (angles, expectation,
    evaluations)."""
    calls = [0]

    def neg(x):
        calls[0] += 1
        return -float(np.abs(qaoa_statevector(n, cost, x, x0)) ** 2 @ cost)

    rng = np.random.default_rng(seed)
    starts = [np.asarray(init, float)] if init is not None else []
    while len(starts) < restarts:
        starts.append(rng.uniform(0, math.pi / 2, 2 * p))
    best = None
    for x in starts:
        r = minimize(neg, x, method="COBYLA", options=dict(maxiter=maxiter, rhobeg=0.2))
        if best is None or r.fun < best.fun:
            best = r
    return [float(v) for v in best.x], -float(best.fun), calls[0]


def maxcut_cost(n, edges):
    """Cut value of every basis state (bit i = node i) for (a, b, w) edges; signed weights allowed."""
    idx = np.arange(2 ** n, dtype=np.int64)
    cost = np.zeros(2 ** n)
    for a, b, w in edges:
        cost += w * (((idx >> a) ^ (idx >> b)) & 1)
    return cost


def random_ratio(cost):
    """Expected score of a uniformly random bit string over the best score (the zero-gate baseline)."""
    return float(cost.mean() / cost.max())


def best_of_shots(probs, cost, shots):
    """Expected best score among `shots` independent samples of `probs` (exact, from the sorted scores)."""
    order = np.argsort(cost)
    c, p = cost[order], np.clip(np.asarray(probs)[order], 0, None)
    p = p / p.sum()
    vals, inv = np.unique(c, return_inverse=True)
    pv = np.bincount(inv, weights=p, minlength=len(vals))
    F = np.cumsum(pv)
    Fm = np.concatenate([[0.0], F[:-1]])
    return float(np.sum(vals * (np.clip(F, 0, 1) ** shots - np.clip(Fm, 0, 1) ** shots)))


# ---------------------------------------------------------------- recursive QAOA (p = 1)

def zz_from_probs(n, probs, pairs):
    """<Z_a Z_b> for each pair from a distribution over n bits (bit i = node i)."""
    idx = np.arange(2 ** n, dtype=np.int64)
    return [float(np.sum(probs * (1 - 2 * (((idx >> a) ^ (idx >> b)) & 1)))) for a, b in pairs]


def ideal_correlations(n, edges, seed=0):
    """p = 1 QAOA tuned on the reduced Max-Cut (numpy), then <Z_a Z_b> on every edge. Returns (zz, angles)."""
    cost = maxcut_cost(n, edges)
    angles, _, _ = tune_numpy(n, cost, 1, restarts=4, seed=seed)
    probs = np.abs(qaoa_statevector(n, cost, angles)) ** 2
    return zz_from_probs(n, probs, [(a, b) for a, b, _ in edges]), angles


def rqaoa(n, edges, correlations=None, stop_at=8, seed=0, on_step=None):
    """Recursive QAOA at p = 1 (Bravyi et al. 2020) for Max-Cut with signed weights.

    Each step computes <Z_a Z_b> on every edge (`correlations(n, edges, step)` -> list; default: the ideal
    p = 1 state with tuned angles), takes the edge with the largest |<Z_a Z_b>| (ties: first edge), fixes
    z_b = sign * z_a and removes b: edge (b, k, w) becomes (a, k, sign * w). Once `stop_at` nodes remain, the
    reduced problem is solved by enumeration and the eliminated nodes are filled back in.
    Returns (assignment as 0/1 list over the original nodes, list of step records)."""
    correlations = correlations or (lambda m, es, step: ideal_correlations(m, es, seed=seed)[0])
    alive = list(range(n))                          # reduced index -> original node
    cur = [(a, b, float(w)) for a, b, w in edges]
    fixed = []                                      # (original node b, original node a, sign)
    steps = []
    while len(alive) > stop_at and cur:
        m = len(alive)
        zz = correlations(m, cur, len(steps))
        k = int(np.argmax(np.abs(zz)))
        a, b, _ = cur[k]
        sgn = 1 if zz[k] >= 0 else -1
        fixed.append((alive[b], alive[a], sgn))
        steps.append(dict(step=len(steps), nodes=m, edges=len(cur), zz=float(zz[k]),
                          eliminated=alive[b], kept=alive[a], sign=sgn))
        if on_step:
            on_step(steps[-1])
        merged = {}
        for u, v, w in cur:
            u, v, w = (a if u == b else u), (a if v == b else v), (w * sgn if b in (u, v) else w)
            if u == v:
                continue                            # z_a^2 = 1: a constant, not a coupling
            key = (min(u, v), max(u, v))
            merged[key] = merged.get(key, 0.0) + w
        relabel = {old: new for new, old in enumerate(i for i in range(m) if i != b)}
        cur = [(relabel[u], relabel[v], w) for (u, v), w in sorted(merged.items()) if abs(w) > 1e-12]
        alive = [alive[i] for i in range(m) if i != b]
    m = len(alive)
    rest = maxcut_cost(m, cur) if cur else np.zeros(2 ** m)
    best = int(np.argmax(rest))
    z = {alive[i]: 1 - 2 * ((best >> i) & 1) for i in range(m)}   # bit 1 -> spin -1
    for b, a, sgn in reversed(fixed):
        z[b] = sgn * z[a]
    return [0 if z[v] == 1 else 1 for v in range(n)], steps


# ---------------------------------------------------------------- warm start with parametric thetas

def build_warm_parametric(inst, p=1):
    """Warm-start QAOA (Egger et al. 2021) with the warm-start angles as parameters too, so one compile serves
    every warm start: bind `t` (gamma, beta per layer) and `w` (theta_v = warm_thetas(x0)). No measurements."""
    nq = n_qubits(inst)
    th, tw = ParameterVector("t", 2 * p), ParameterVector("w", nq)
    qc = QuantumCircuit(nq)
    for v in range(nq):
        qc.ry(tw[v], v)
    for k in range(p):
        _cost_layer(qc, inst, th[2 * k])
        for v in range(nq):
            qc.ry(-tw[v], v)
            qc.rz(-2 * th[2 * k + 1], v)
            qc.ry(tw[v], v)
    return qc, th, tw


def warm_thetas(x0, eps=0.25):
    return [2 * math.asin(math.sqrt(1 - eps if b else eps)) for b in x0]


def interp(angles, p):
    """INTERP initialization (Zhou et al. 2020): p-layer angles from (p-1)-layer ones, in build()'s order."""
    g, b = angles[0::2], angles[1::2]

    def up(v):
        v = [0.0] + list(v) + [0.0]
        return [(i / p) * v[i] + ((p - i) / p) * v[i + 1] for i in range(p)]
    return [x for pair in zip(up(g), up(b)) for x in pair]


def bits_index(x):
    """Basis-state index of a 0/1 assignment (bit i = x[i])."""
    return int(sum(int(v) << i for i, v in enumerate(x)))
