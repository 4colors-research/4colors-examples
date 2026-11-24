"""The 29 circuit families of the compiler report card, three sizes each, from plain Qiskit.

Attribution: the family builders below are copied, with the changes listed under GENERATOR_CHANGES, from the
4colors blog circuit generators (internal, pinned versions):
  - the domain generator (tfim2d, xxz1d, hubbard1d,
    hubbard2d, kicked_ising_hh, uccsd, lucj, hea_full, schwinger, qae, portfolio_xy, zzkernel, repcode, surface,
    grover);
  - the campaign generator (ising, qaoa, heisenberg,
    uccsd excitations, bv_ka, cuccaro, draper, ghz, qv, qft, lrising, ising1d, qaoa3r, qaoaer).
The copy lives here so the post's circuits never change when that branch does: the generator changed between
earlier internal runs, so every run of this post records GENERATOR_VERSION, the commit of this file and the
sha256 of every circuit it compiled (`instance_sha256`).

Every circuit has the SHAPE of its domain workload (the same gates and interaction graph as the real algorithm).
Angles and amplitudes are fixed stand-ins, so a gate count is a resource result, never a chemistry or physics result.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

GENERATOR_VERSION = "report-card-1"
GENERATOR_CHANGES = (
    "uccsd starts from the Hartree-Fock reference (X on the occupied spin-orbitals); "
    "hubbard1d and hubbard2d start from a half-filled Neel-type product reference (one electron per site, "
    "alternating spin), so neither acts trivially on |0...0>; "
    "instances are numbered 101 and 102 (fresh: never used by the earlier internal runs, which used 0); "
    "no other change to any builder.")
#: The gates every input is lowered to before it is sent (as in the copied generator), plus reset for the QEC
#: families. Both compilers get the same lowered circuit.
BASIS = ["cx", "rz", "rx", "ry", "h", "x", "sx", "p", "cp", "swap", "ccx", "reset"]
INSTANCES = (101, 102)


@dataclass(frozen=True)
class Family:
    name: str
    domain: str
    sizes: Tuple[int, int, int]      # the family's own size parameter (see each builder)
    mechanism: str                   # one line for the post: the structure the compile can use (a hypothesis)
    build: Callable[[int, int], Tuple["object", dict]]


# ---- shared pieces (copied) -----------------------------------------------------------------------

def _qc(n):
    from qiskit import QuantumCircuit

    return QuantumCircuit(n)


def trotter(n, terms, steps, dt=0.1):
    """First-order Trotter of sum_k c_k P_k: one PauliEvolutionGate per term, synthesized when lowered."""
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    qc = _qc(n)
    for _ in range(steps):
        for label, c in terms:
            qc.append(PauliEvolutionGate(SparsePauliOp(label, c), time=dt), range(n))
    return qc


def pstr(n, ops):
    """Pauli label (Qiskit order, qubit n-1 first) from {qubit: 'X'|'Y'|'Z'}."""
    s = ["I"] * n
    for q, p in ops.items():
        s[n - 1 - q] = p
    return "".join(s)


def hop(n, a, b):
    """Jordan-Wigner hopping a^dag_a a_b + h.c. between qubits a < b: (XZ..ZX + YZ..ZY) / 2."""
    z = {k: "Z" for k in range(a + 1, b)}
    return [(pstr(n, {a: "X", b: "X", **z}), 0.5), (pstr(n, {a: "Y", b: "Y", **z}), 0.5)]


def ising(n, edges, steps, dt=0.2, h=1.0):
    qc = _qc(n)
    for _ in range(steps):
        for u, v in edges:
            qc.cx(u, v); qc.rz(2 * dt, v); qc.cx(u, v)
        qc.rx(2 * h * dt, range(n))
    return qc


def heisenberg(n, edges, steps, dt=0.2):
    qc = _qc(n)
    for _ in range(steps):
        for a, b in edges:
            qc.rxx(2 * dt, a, b); qc.ryy(2 * dt, a, b); qc.rzz(2 * dt, a, b)
    return qc


def qaoa(graph, p):
    n = graph.number_of_nodes(); qc = _qc(n)
    qc.h(range(n))
    g, b = ([0.6155], [0.3927]) if p == 1 else ([0.6155 * (k + 1) / p for k in range(p)],
                                                [0.3927 * (p - k) / p for k in range(p)])
    for layer in range(p):
        for u, v in graph.edges():
            qc.cx(u, v); qc.rz(2 * g[layer], v); qc.cx(u, v)
        qc.rx(2 * b[layer], range(n))
    return qc


def grid_edges(k):
    """Nearest-neighbor bonds of an OPEN k x k square grid (no wrap-around), site i*k+j."""
    return ([(i * k + j, i * k + j + 1) for i in range(k) for j in range(k - 1)]
            + [(i * k + j, (i + 1) * k + j) for i in range(k - 1) for j in range(k)])


def uccsd_ops(n, ne, rng, amp):
    """Spin-conserving singles and doubles over ne/2 + ne/2 electrons (even qubits alpha, odd beta), Jordan-Wigner,
    one first-order Trotter step, amplitudes uniform in [-amp, amp]."""
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    def a(j):
        z = "I" * (n - j - 1)
        return SparsePauliOp([z + "X" + "Z" * j, z + "Y" + "Z" * j], [0.5, 0.5j])
    occ = [q for q in range(n) if q < ne]; virt = [q for q in range(n) if q >= ne]
    exc = [((i,), (v,)) for i in occ for v in virt if i % 2 == v % 2]
    exc += [((i, j), (v, w)) for i, j in itertools.combinations(occ, 2) for v, w in itertools.combinations(virt, 2)
            if sorted([i % 2, j % 2]) == sorted([v % 2, w % 2])]
    qc = _qc(n)
    for q in occ:                      # CHANGED: Hartree-Fock reference, the occupied spin-orbitals filled
        qc.x(q)
    for frm, to in exc:
        op = None
        for q in to:
            x = a(q).adjoint(); op = x if op is None else op.compose(x)
        for q in frm:
            op = op.compose(a(q))
        g = (op - op.adjoint()).simplify()
        h = SparsePauliOp(g.paulis, (g.coeffs * 1j).real)
        qc.append(PauliEvolutionGate(h, time=rng.uniform(-amp, amp)), range(n))
    return qc


# ---- builders (size parameter -> (circuit without measurements, meta)) ----------------------------

def b_tfim2d(k, inst):
    """Transverse-field Ising on an open k x k grid, 2 Trotter steps (n = k^2)."""
    return ising(k * k, grid_edges(k), 2), {"grid": [k, k], "boundary": "open"}


def b_xxz1d(n, inst):
    t = []
    for i in range(n - 1):
        t += [(pstr(n, {i: "X", i + 1: "X"}), 1.0), (pstr(n, {i: "Y", i + 1: "Y"}), 1.0),
              (pstr(n, {i: "Z", i + 1: "Z"}), 0.5)]
    return trotter(n, t, 3), {}


def b_heis1d(n, inst):
    return heisenberg(n, [(i, i + 1) for i in range(n - 1)], 2), {}


def b_heis2d(n, inst):
    k = round(n ** 0.5)
    return heisenberg(k * k, grid_edges(k), 2), {"grid": [k, k], "boundary": "open"}


def _neel(qc, qubits_up, qubits_down):
    """CHANGED: a half-filled product reference: one electron per site, alternating spin."""
    for s, q in enumerate(qubits_up):
        if s % 2 == 0:
            qc.x(q)
    for s, q in enumerate(qubits_down):
        if s % 2 == 1:
            qc.x(q)


def b_hubbard1d(L, inst):
    """Fermi-Hubbard chain, L sites, spin-interleaved JW (qubit 2i up, 2i+1 down), t = 1, U = 4, 2 steps (n = 2L)."""
    n = 2 * L; t = []
    for i in range(L - 1):
        for s in (0, 1):
            t += hop(n, 2 * i + s, 2 * (i + 1) + s)
    t += [(pstr(n, {2 * i: "Z", 2 * i + 1: "Z"}), 1.0) for i in range(L)]
    qc = _qc(n)
    _neel(qc, [2 * i for i in range(L)], [2 * i + 1 for i in range(L)])
    return qc.compose(trotter(n, t, 2)), {"sites": L, "reference": "neel"}


def b_hubbard2d(w, inst):
    """Fermi-Hubbard on a 2 x w ladder, snake JW ordering per spin (up 0..2w-1, down 2w..4w-1), 1 step."""
    m = 2 * w; n = 2 * m
    snake = [r * w + (c if r == 0 else w - 1 - c) for r in range(2) for c in range(w)]
    idx = {s: i for i, s in enumerate(snake)}
    edges = ([(c, c + 1) for c in range(w - 1)] + [(w + c, w + c + 1) for c in range(w - 1)]
             + [(c, w + c) for c in range(w)])
    t = []
    for a, b in edges:
        ia, ib = sorted((idx[a], idx[b]))
        for off in (0, m):
            t += hop(n, ia + off, ib + off)
    t += [(pstr(n, {i: "Z", i + m: "Z"}), 1.0) for i in range(m)]
    qc = _qc(n)
    _neel(qc, list(range(m)), list(range(m, n)))
    return qc.compose(trotter(n, t, 1)), {"ladder": [2, w], "reference": "neel"}


def b_kicked_ising_hh(n, inst):
    """Kicked Ising on an n-qubit connected piece of the FakeFez heavy-hex map (working couplers, BFS from a fixed
    qubit), relabelled 0..n-1, 4 steps of rzz(-pi/2) on every edge + rx(0.5). Native: needs zero SWAPs."""
    import networkx as nx
    from qiskit_ibm_runtime.fake_provider import FakeFez

    tg = FakeFez().target
    cm = nx.Graph([tuple(q) for q, p in tg["cz"].items() if p is None or p.error is None or p.error < 0.99])
    start = sorted(cm.nodes)[inst % len(cm)]
    keep = list(nx.bfs_tree(cm, start))[:n]; sub = cm.subgraph(keep); lab = {q: i for i, q in enumerate(keep)}
    edges = sorted((min(lab[a], lab[b]), max(lab[a], lab[b])) for a, b in sub.edges())
    qc = _qc(n)
    for _ in range(4):
        qc.rx(0.5, range(n))
        for a, b in edges:
            qc.rzz(-math.pi / 2, a, b)
    return qc, {"graph": [list(e) for e in edges]}


def b_ising1d(n, inst):
    return ising(n, [(i, i + 1) for i in range(n - 1)], 2), {}


def b_lrising(n, inst):
    return ising(n, [(i, j) for i in range(n) for j in range(i + 1, n)], 1), {}


def b_uccsd(n, inst):
    """UCCSD shape (JW, spin-conserving singles + doubles, one Trotter step) from the Hartree-Fock reference,
    n spin-orbitals, 4 electrons (an active space), fixed amplitudes."""
    rng = random.Random(1000 * n + inst); ne = 4 if n >= 8 else 2
    return uccsd_ops(n, ne, rng, 0.1), {"electrons": ne, "reference": "hartree_fock"}


def b_lucj(norb, inst):
    """LUCJ shape (orbital rotation, diagonal Coulomb, inverse rotation), n = 2 norb, half-filling reference."""
    n = 2 * norb; rng = np.random.default_rng(1000 * norb + inst); qc = _qc(n)
    for k in range(norb // 2):
        qc.x(k); qc.x(norb + k)

    def givens(sign):
        for layer in range(norb):
            for i in range(layer % 2, norb - 1, 2):
                for off in (0, norb):
                    th = sign * float(rng.uniform(-0.3, 0.3))
                    qc.rxx(th, i + off, i + 1 + off); qc.ryy(th, i + off, i + 1 + off)
    givens(1)
    for i in range(norb - 1):
        for off in (0, norb):
            qc.rzz(float(rng.uniform(-0.2, 0.2)), i + off, i + 1 + off)
    for i in range(0, norb, 4):
        qc.rzz(float(rng.uniform(-0.2, 0.2)), i, norb + i)
    givens(-1)
    return qc, {"orbitals": norb}


def b_hea_full(n, inst):
    from qiskit.circuit.library import efficient_su2

    c = efficient_su2(n, reps=2, entanglement="full")
    return c.assign_parameters(np.random.default_rng(1000 * n + inst).uniform(-np.pi, np.pi, c.num_parameters)), {}


def b_schwinger(N, inst):
    """Lattice Schwinger model (staggered fermions, Gauss law eliminated: long-range ZZ), N sites, 1 step."""
    n = N; t = []
    for i in range(N - 1):
        t += hop(n, i, i + 1)
    t += [(pstr(n, {i: "Z"}), 0.25 * (-1) ** i) for i in range(N)]
    for i, j in itertools.combinations(range(N - 1), 2):
        t.append((pstr(n, {i: "Z", j: "Z"}), 0.5 * (N - 1 - j)))
    return trotter(n, t, 1), {}


def b_qae(m, inst):
    """Amplitude estimation shape (option pricing): m uncertainty qubits with a log-normal (StatePreparation),
    a linear payoff as controlled RY onto one objective qubit, then Grover power Q^2. n = m + 1."""
    from qiskit.circuit.library import StatePreparation, grover_operator

    x = np.linspace(0.05, 3, 2 ** m); p = np.exp(-(np.log(x) - 0.2) ** 2 / 0.32) / x; p /= p.sum()
    a = _qc(m + 1); a.append(StatePreparation(np.sqrt(p)), range(m))
    a.ry(0.1, m)
    for i in range(m):
        a.cry(0.2 * 2 ** i / 2 ** m, i, m)
    orc = _qc(m + 1); orc.z(m)
    q = grover_operator(orc, state_preparation=a)
    qc = a.copy()
    for _ in range(2):
        qc.compose(q, inplace=True)
    return qc, {"grover_power": 2}


def b_portfolio_xy(n, inst):
    """Portfolio QAOA shape: dense covariance ZZ on every pair, XY ring mixer, p = 1, first n/2 assets in."""
    rng = np.random.default_rng(1000 * n + inst); qc = _qc(n)
    for k in range(n // 2):
        qc.x(k)
    for i, j in itertools.combinations(range(n), 2):
        qc.rzz(float(rng.uniform(0.05, 0.3)), i, j)
    for i in range(n):
        qc.rxx(0.4, i, (i + 1) % n); qc.ryy(0.4, i, (i + 1) % n)
    return qc, {}


def b_zzkernel(n, inst):
    """Quantum-kernel fidelity circuit: ZZFeatureMap(x) then its inverse at x', full entanglement, reps 2."""
    from qiskit.circuit.library import zz_feature_map

    rng = np.random.default_rng(1000 * n + inst)
    fm = zz_feature_map(n, reps=2, entanglement="full")
    qc = fm.assign_parameters(rng.uniform(0, 2, n)).compose(fm.assign_parameters(rng.uniform(0, 2, n)).inverse())
    return qc, {}


def b_repcode(d, inst):
    n = 2 * d - 1; qc = _qc(n)
    for _ in range(d):
        for a in range(1, n, 2):
            qc.cx(a - 1, a); qc.cx(a + 1, a)
        for a in range(1, n, 2):
            qc.reset(a)
    return qc, {"distance": d}


def b_surface(d, inst):
    """Rotated surface code, distance d, d syndrome rounds, standard N/Z CX order; d^2 data + d^2-1 ancillas."""
    data = {(r, c): r * d + c for r in range(d) for c in range(d)}; anc = []
    for r in range(d + 1):
        for c in range(d + 1):
            kind = "X" if (r + c) % 2 == 0 else "Z"
            nb = [(r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c)]
            nb = [data[x] if x in data else None for x in nb]
            k = sum(x is not None for x in nb)
            if k == 4 or (k == 2 and ((kind == "X" and r in (0, d)) or (kind == "Z" and c in (0, d)))):
                anc.append((kind, nb))
    n = d * d + len(anc); qc = _qc(n)
    order = {"X": [0, 1, 2, 3], "Z": [0, 2, 1, 3]}
    for _ in range(d):
        for k, (kind, _nb) in enumerate(anc):
            if kind == "X":
                qc.h(d * d + k)
        for step in range(4):
            for k, (kind, nb) in enumerate(anc):
                q = nb[order[kind][step]]
                if q is None:
                    continue
                a = d * d + k
                if kind == "X":
                    qc.cx(a, q)
                else:
                    qc.cx(q, a)
        for k, (kind, _nb) in enumerate(anc):
            if kind == "X":
                qc.h(d * d + k)
            qc.reset(d * d + k)
    return qc, {"distance": d, "ancillas": len(anc)}


def b_grover(n, inst):
    from qiskit.circuit.library import grover_operator

    s = random.Random(1000 * n + inst).getrandbits(n)
    orc = _qc(n)
    for i in range(n):
        if not (s >> i) & 1:
            orc.x(i)
    orc.h(n - 1); orc.mcx(list(range(n - 1)), n - 1); orc.h(n - 1)
    for i in range(n):
        if not (s >> i) & 1:
            orc.x(i)
    qc = _qc(n); qc.h(range(n)); g = grover_operator(orc)
    for _ in range(2):
        qc.compose(g, inplace=True)
    return qc, {"answers": [format(s, f"0{n}b")]}


def _qaoa_graph(kind, n, inst):
    import networkx as nx

    seed = 1000 * n + inst
    if kind == "3r":
        return nx.random_regular_graph(3, n, seed=seed)
    return nx.gnp_random_graph(n, {"er01": 0.1, "er03": 0.3}[kind], seed=seed)


def _qaoa_builder(kind, p):
    def build(n, inst):
        g = _qaoa_graph(kind, n, inst)
        return qaoa(g, p), {"graph": sorted(map(list, g.edges())), "p": p}
    return build


def b_bv_ka(n, inst):
    m = n - 1; s = random.Random(1000 * n + inst).getrandbits(m) | 1; qc = _qc(n)
    qc.x(m); qc.h(range(n))
    for i in range(m):
        if (s >> i) & 1:
            qc.cx(i, m)
    qc.h(range(m)); qc.h(m); qc.x(m)
    return qc, {"answers": [format(s, f"0{m}b").rjust(n, "0")]}


def b_cuccaro(n, inst):
    from qiskit.circuit.library import CDKMRippleCarryAdder

    return CDKMRippleCarryAdder((n - 2) // 2, kind="full"), {"bits": (n - 2) // 2}


def b_draper(n, inst):
    from qiskit.circuit.library import DraperQFTAdder

    return DraperQFTAdder(n // 2, kind="fixed"), {"bits": n // 2}


def b_ghz(n, inst):
    qc = _qc(n); qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc, {}


def b_qv(n, inst):
    from qiskit.circuit.library import QuantumVolume

    return QuantumVolume(n, seed=1000 * n + inst), {}


def b_qft(n, inst):
    from qiskit.circuit.library import QFT

    return QFT(n), {}


# ---- the registry -----------------------------------------------------------------------------------

def _f(name, domain, sizes, mechanism, build):
    return name, Family(name, domain, sizes, mechanism, build)


FAMILIES: Dict[str, Family] = dict([
    _f("tfim2d", "materials", (4, 6, 8), "commuting ZZ bonds that can be reordered into hardware matchings", b_tfim2d),
    _f("xxz1d", "materials", (16, 32, 64), "a 1D chain: embeds on a path with no SWAPs", b_xxz1d),
    _f("heis1d", "materials", (16, 32, 64), "a 1D chain: embeds on a path with no SWAPs", b_heis1d),
    _f("heis2d", "materials", (16, 36, 64), "XX+YY+ZZ on each bond of a 2D grid: one 3-CZ block per bond", b_heis2d),
    _f("hubbard1d", "materials", (4, 8, 16), "nearest-neighbor hopping on a chain (spin-interleaved)", b_hubbard1d),
    _f("hubbard2d", "materials", (3, 5, 8), "2D hopping with Jordan-Wigner strings", b_hubbard2d),
    _f("kicked_ising_hh", "materials", (27, 64, 127), "already native on heavy-hex: zero SWAPs possible",
       b_kicked_ising_hh),
    _f("ising1d", "materials", (16, 32, 64), "a 1D chain: embeds on a path with no SWAPs", b_ising1d),
    _f("lrising", "materials", (8, 12, 16), "all-to-all commuting ZZ: routing dominates", b_lrising),
    _f("uccsd", "chemistry", (12, 16, 20), "long Pauli-exponential CNOT ladders that can be refit to the chip",
       b_uccsd),
    _f("lucj", "chemistry", (8, 16, 26), "built nearest-neighbor for hardware: little to route", b_lucj),
    _f("hea_full", "chemistry", (8, 16, 24), "an all-to-all entangler layer of commuting CZs", b_hea_full),
    _f("schwinger", "hep", (16, 32, 64), "long-range ZZ from the eliminated gauge field", b_schwinger),
    _f("qae", "finance", (4, 6, 8), "deep controlled rotations with little reordering freedom", b_qae),
    _f("portfolio_xy", "finance", (12, 20, 32), "dense ZZ cost layer plus an XY ring mixer", b_portfolio_xy),
    _f("zzkernel", "qml", (8, 16, 24), "commuting ZZ feature map on every pair", b_zzkernel),
    _f("repcode", "qec", (5, 11, 25), "a 1D syndrome pattern: embeds on a path", b_repcode),
    _f("surface", "qec", (3, 5, 7), "degree-4 stabilizer checks on a degree-3 lattice", b_surface),
    _f("grover", "primitives", (5, 6, 7), "multi-controlled-Z synthesis on a small patch", b_grover),
    _f("qaoa3r_p1", "optimization", (16, 32, 64), "one commuting ZZ cost layer on a 3-regular graph",
       _qaoa_builder("3r", 1)),
    _f("qaoa3r_p2", "optimization", (16, 32, 64), "two commuting ZZ cost layers on a 3-regular graph",
       _qaoa_builder("3r", 2)),
    _f("qaoaer01", "optimization", (16, 32, 64), "commuting ZZ cost layer on a sparse random graph",
       _qaoa_builder("er01", 1)),
    _f("qaoaer03", "optimization", (16, 24, 32), "commuting ZZ cost layer on a denser random graph",
       _qaoa_builder("er03", 1)),
    _f("bv_ka", "primitives", (24, 48, 96), "a fan-in of commuting CNOTs into one qubit", b_bv_ka),
    _f("cuccaro", "arithmetic", (18, 34, 66), "a local ripple of Toffolis along a line", b_cuccaro),
    _f("draper", "arithmetic", (8, 16, 32), "QFT-based addition: dense controlled phases", b_draper),
    _f("ghz", "primitives", (16, 32, 64), "a CNOT chain: embeds on a path", b_ghz),
    _f("qv", "benchmarking", (8, 12, 16), "random SU(4) layers on random pairs", b_qv),
    _f("qft", "primitives", (8, 16, 32), "all-to-all controlled phases", b_qft),
])


def make(family: str, size: int, inst: int, measure: bool = True):
    """(circuit, meta): the family's circuit lowered to BASIS at optimization level 0 (as the copied generator
    did), with measure_all when `measure` (the post compiles measured circuits)."""
    from qiskit import transpile

    fam = FAMILIES[family]
    qc, meta = fam.build(size, inst)
    qc = transpile(qc, basis_gates=BASIS, optimization_level=0)
    if measure:
        qc.measure_all()
    qc.name = circuit_id(family, qc.num_qubits, inst)
    return qc, {**meta, "family": family, "domain": fam.domain, "size": size, "instance": inst,
                "generator_version": GENERATOR_VERSION}


def circuit_id(family: str, n_qubits: int, inst: int) -> str:
    return f"{family}-n{n_qubits}-i{inst}"


def instance_sha256(qc) -> str:
    """sha256 of the circuit's OpenQASM 3 text, exactly as written to the run's inputs."""
    from qiskit import qasm3

    return hashlib.sha256(qasm3.dumps(qc).encode()).hexdigest()


def plan(families=None, sizes_index=(0, 1, 2), instances=INSTANCES) -> List[Tuple[str, int, int]]:
    """Every (family, size, instance) of a run, in a fixed order."""
    names = list(families or FAMILIES)
    return [(f, FAMILIES[f].sizes[i], inst) for f in names for i in sizes_index for inst in instances]
