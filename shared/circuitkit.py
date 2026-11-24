"""Helpers shared by the quantum-compiler blog-post examples (the QUBO posts use examplekit.py alone).

Everything here is plain Qiskit plus the public `fourcolors` client:

* circuit generators (Bernstein-Vazirani, QAOA Max-Cut, Draper adder, QFT, a hand-built UCCSD ansatz) and
  graphs;
* ONE metric function, `circuit_metrics`, applied the same way to Qiskit's output and to `res.to_qiskit()`;
* the device: FakeFez as Qiskit sees it, the same map restricted to its working couplers, its failed couplers,
  and its calibration in the service's `calibration` format;
* the two arms: `qiskit_trials` (level 3, seeds until the budget is spent) and `compile_4colors`
  (`fourcolors.quantum.compiler.compile`), both returning plain row fields;

Qiskit, networkx and the fake backends are imported inside the functions that need them.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

Pair = Tuple[int, int]

#: A coupler whose calibrated two-qubit error is at least this is treated as failed (IBM marks dead couplers 1.0).
FAILED_ERROR = 0.999
#: Instructions that are not gates: never counted as gates, never in the depth.
NON_GATES = frozenset({"barrier", "delay", "measure", "reset"})
#: The heavy-hex posts compile for FakeFez: its description and its published calibration, built by the client
#: (`fourcolors.quantum.targets.from_qiskit`), as a customer sends any device. There are no named targets: a
#: target is a description, the calibration the caller's own.
#: This label is the `coupler_set` of those 4colors rows.
HEAVY_HEX_DEVICE = "fakefez"

#: Every result row has these keys (None when they do not apply), so plot scripts never guess.
ROW_KEYS = ("kind", "post", "circuit_id", "family", "qubits", "input_2q", "arm", "coupler_set", "seed",
            "budget_s", "two_qubit_gates", "two_qubit_depth", "depth", "gates_on_failed_couplers",
            "verification_label", "verification_distance", "verification_reason", "engine_version",
            "billed_s", "job_id", "status", "reason_code", "refusal_code", "refusal_message", "wall_s")


# ---- metrics: one function for both arms ----------------------------------------------------------

def is_two_qubit(inst) -> bool:
    """A two-qubit gate (not a barrier, measurement or delay)."""
    op = inst.operation
    return op.num_qubits == 2 and op.name not in NON_GATES


def _pair(qc, inst) -> Pair:
    a, b = sorted(qc.find_bit(q).index for q in inst.qubits)
    return a, b


def circuit_metrics(qc, failed: Iterable[Sequence[int]] = ()) -> Dict[str, int]:
    """Two-qubit gates, two-qubit depth, depth (every gate except barrier/delay/measure/reset, rz included)
    and two-qubit gates on `failed` couplers. Qubit k of the circuit is physical qubit k: true for Qiskit's
    output on a target and for `res.to_qiskit()`."""
    bad = {tuple(sorted(int(x) for x in e)) for e in failed}
    n2 = on_bad = 0
    for inst in qc.data:
        if is_two_qubit(inst):
            n2 += 1
            if bad and _pair(qc, inst) in bad:
                on_bad += 1
    return {"two_qubit_gates": n2,
            "two_qubit_depth": qc.depth(is_two_qubit) if n2 else 0,
            "depth": qc.depth(lambda inst: inst.operation.name not in NON_GATES),
            "gates_on_failed_couplers": on_bad}


def coupler_counts(qc) -> List[List[int]]:
    """[[a, b, two-qubit gates on (a, b)], ...] for a circuit on physical qubits, sorted (for heat maps)."""
    c: Counter = Counter(_pair(qc, inst) for inst in qc.data if is_two_qubit(inst))
    return [[a, b, n] for (a, b), n in sorted(c.items())]


def interaction_edges(qc) -> List[List[int]]:
    """The logical circuit's interaction graph: [[a, b, two-qubit gates between a and b], ...]."""
    return coupler_counts(qc)


# ---- circuits -------------------------------------------------------------------------------------

def bernstein_vazirani(secret: str, measure: bool = True):
    """Bernstein-Vazirani for `secret` (a 0/1 string, secret[i] belongs to qubit i) on len(secret) + 1 qubits.
    The last qubit is the ancilla in |->; every 1 bit is one CNOT from its qubit to the ancilla, so the
    interaction graph is a star centered on the ancilla. A noiseless run measures `secret` (Qiskit prints
    bit 0 last: the counts key is secret[::-1])."""
    from qiskit import QuantumCircuit

    if not secret or set(secret) - {"0", "1"}:
        raise ValueError("secret must be a non-empty string of 0 and 1")
    n = len(secret)
    qc = QuantumCircuit(n + 1, n if measure else 0, name=f"bv_{n}")
    qc.x(n)
    qc.h(range(n + 1))
    for i, bit in enumerate(secret):
        if bit == "1":
            qc.cx(i, n)
    qc.h(range(n))
    if measure:
        qc.measure(range(n), range(n))
    return qc


def random_secret(n: int, seed: int) -> str:
    import random

    rnd = random.Random(seed)
    return "".join(rnd.choice("01") for _ in range(n))


# Fixed angles for p = 1..3 on 3-regular graphs (Wurtz and Lykov, PRA 104, 052419 (2021), Table I). They only
# set the rotation angles; the gate count and the routing problem do not depend on them.
FIXED_ANGLES = {1: ([0.6155], [0.3927]),
                2: ([0.4877, 0.8979], [0.5550, 0.2921]),
                3: ([0.4220, 0.7979, 0.9372], [0.6089, 0.4598, 0.2357])}


def qaoa_maxcut(n: int, edges: Iterable[Sequence[int]], p: int = 1, gammas=None, betas=None,
                measure: bool = False):
    """QAOA for Max-Cut: H on every qubit, then p layers of rzz(-gamma) on each edge and rx(2 beta) on each
    qubit. One rzz per edge per layer: the logical circuit has p * len(edges) two-qubit gates."""
    from qiskit import QuantumCircuit

    edges = [tuple(int(x) for x in e) for e in edges]
    if gammas is None or betas is None:
        if p in FIXED_ANGLES:
            gammas, betas = FIXED_ANGLES[p]
        else:                                  # a linear ramp for deeper circuits
            gammas = [0.75 * (k + 0.5) / p for k in range(p)]
            betas = [0.75 * (1 - (k + 0.5) / p) for k in range(p)]
    qc = QuantumCircuit(n, name=f"qaoa_n{n}_p{p}")
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for a, b in edges:
            qc.rzz(-float(gamma), a, b)
        qc.rx(2 * float(beta), range(n))
    if measure:
        qc.measure_all()
    return qc


def barabasi_albert(n: int, m: int, seed: int) -> List[Pair]:
    """A Barabasi-Albert graph (networkx): each new node attaches to m existing ones, so a few hubs get
    many edges. Benchpress-style QAOA inputs use this family."""
    import networkx as nx

    g = nx.barabasi_albert_graph(n, m, seed=seed)
    return sorted(tuple(sorted(e)) for e in g.edges())


def draper_adder(n_bits: int, a: Optional[int] = None, b: Optional[int] = None):
    """Draper QFT adder |a>|b> -> |a>|a + b mod 2^n_bits> on 2 n_bits qubits (qiskit.synthesis.adder_qft_d00,
    kind "fixed"), decomposed until no gate acts on more than two qubits. `a` and `b` prepare the inputs
    with X gates (qubits 0..n-1 hold a, n..2n-1 hold b, little-endian)."""
    from qiskit import QuantumCircuit
    from qiskit.synthesis import adder_qft_d00

    core = adder_qft_d00(n_bits, kind="fixed")
    qc = QuantumCircuit(core.num_qubits, name=f"adder_{n_bits}b")
    for value, offset in ((a, 0), (b, n_bits)):
        if value:
            for k in range(n_bits):
                if (value >> k) & 1:
                    qc.x(offset + k)
    qc.compose(core, inplace=True)
    for _ in range(8):
        if all(len(inst.qubits) <= 2 for inst in qc.data):
            break
        qc = qc.decompose()
    return qc


def qft(n: int):
    """The QFT on n qubits (h and controlled phases; no final swaps, a relabeling every compiler absorbs)."""
    from qiskit.synthesis import synth_qft_full

    qc = synth_qft_full(n, do_swaps=False)
    qc.name = f"qft_{n}"
    return qc


# ---- UCCSD (compiler-custom-chips, compiler-noisy-observables) ---------------------------------------
# A UCCSD ansatz circuit built by hand (Jordan-Wigner, block spin ordering), with a Hartree-Fock reference.
#
#     qc = ck.uccsd(n_spatial=6, n_electrons=4, seed=1)    # the LiH STO-3G register: 12 qubits, 92 excitations
#
# Spin orbitals: alpha 0..m-1, then beta m..2m-1 (qiskit-nature's block ordering). The Hartree-Fock reference puts
# the electrons in the lowest orbitals of each spin (X gates), so the ansatz acts on a real reference state, not the
# vacuum: on |0...0> every excitation operator is zero and the whole circuit would be the identity.
#
# Each excitation (singles and doubles, spin-preserving) is one factor exp(theta (T - T^dagger)), first-order
# Trotterized: the Pauli strings of a factor commute, so each factor is exact. The amplitudes theta are fixed and
# seeded, not optimized: the circuit has the structure (and the gate cost) of the UCCSD ansatz for that register,
# but it does not prepare a molecule's ground state. The circuit is lowered to cx, rz, sx, x and h with Qiskit at
# optimization level 0, so the input is the textbook CNOT-ladder form.

UccsdExcitation = Tuple[Tuple[int, ...], Tuple[int, ...]]


def uccsd_hf_occupied(n_spatial: int, n_electrons: int) -> List[int]:
    """The occupied spin orbitals of the Hartree-Fock reference (closed shell: half the electrons per spin)."""
    if n_electrons % 2 or not 0 < n_electrons < 2 * n_spatial:
        raise ValueError("closed shell only: an even number of electrons, fewer than 2 * n_spatial")
    k = n_electrons // 2
    return list(range(k)) + [n_spatial + i for i in range(k)]


def uccsd_excitations(n_spatial: int, n_electrons: int) -> List[UccsdExcitation]:
    """Spin-preserving singles and doubles from the Hartree-Fock reference, as (occupied, virtual) tuples."""
    k = n_electrons // 2
    occ = {"a": list(range(k)), "b": [n_spatial + i for i in range(k)]}
    vir = {"a": list(range(k, n_spatial)), "b": [n_spatial + i for i in range(k, n_spatial)]}
    out: List[UccsdExcitation] = []
    for s in "ab":
        out += [((o,), (v,)) for o in occ[s] for v in vir[s]]
    for s in "ab":                                                  # same-spin doubles
        out += [(o, v) for o in itertools.combinations(occ[s], 2) for v in itertools.combinations(vir[s], 2)]
    out += [((oa, ob), (va, vb)) for oa in occ["a"] for ob in occ["b"] for va in vir["a"] for vb in vir["b"]]
    return out


def _uccsd_creation(j: int, n: int):
    from qiskit.quantum_info import SparsePauliOp

    zs = list(range(j))
    return SparsePauliOp.from_sparse_list([("Z" * j + "X", zs + [j], 0.5), ("Z" * j + "Y", zs + [j], -0.5j)], n)


def uccsd_generator(exc: UccsdExcitation, n: int):
    """The Hermitian H = i (T - T^dagger) of one excitation, so that exp(-i theta H) = exp(theta (T - T^dagger))."""
    occ, vir = exc
    t = None
    for v in vir:
        t = _uccsd_creation(v, n) if t is None else t.compose(_uccsd_creation(v, n))
    for o in reversed(occ):
        t = t.compose(_uccsd_creation(o, n).adjoint())
    h = (1j * (t - t.adjoint())).simplify(atol=1e-12)
    if max(abs(c.imag) for c in h.coeffs) > 1e-9:
        raise AssertionError("generator is not Hermitian")
    from qiskit.quantum_info import SparsePauliOp

    return SparsePauliOp(h.paulis, h.coeffs.real)


def uccsd_amplitudes(n_exc: int, seed: int, lo: float = 0.05, hi: float = 0.2) -> List[float]:
    """Fixed amplitudes: magnitude uniform in [lo, hi], random sign, from `seed`."""
    rng = random.Random(seed)
    return [rng.choice((-1, 1)) * rng.uniform(lo, hi) for _ in range(n_exc)]


def uccsd(n_spatial: int, n_electrons: int, seed: int = 1, *, reference: bool = True,
          thetas: Optional[Sequence[float]] = None, measure: bool = True):
    """The UCCSD circuit on 2 * n_spatial qubits (see the section comment above). `reference=False` leaves out the
    Hartree-Fock X gates (the old generator's bug: the circuit then acts as the identity on |0...0>)."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate

    n = 2 * n_spatial
    exc = uccsd_excitations(n_spatial, n_electrons)
    thetas = list(thetas) if thetas is not None else uccsd_amplitudes(len(exc), seed)
    qc = QuantumCircuit(n, name=f"uccsd_{n}")
    if reference:
        for q in uccsd_hf_occupied(n_spatial, n_electrons):
            qc.x(q)
    for e, th in zip(exc, thetas):
        qc.append(PauliEvolutionGate(uccsd_generator(e, n), time=th), range(n))
    out = transpile(qc, basis_gates=["cx", "rz", "sx", "x", "h"], optimization_level=0)
    out.name = qc.name
    if measure:
        out.measure_all()
    return out


# ---- the device: FakeFez -------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def fez():
    """Qiskit's FakeFez backend (a 156-qubit Heron r2 snapshot from qiskit-ibm-runtime), loaded once."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    return FakeFez()


def failed_couplers(target, threshold: float = FAILED_ERROR, gate: str = "cz") -> Set[Pair]:
    """Couplers whose calibrated `gate` error is >= threshold in either direction."""
    out: Set[Pair] = set()
    for qargs, props in (target[gate] or {}).items():
        if qargs is not None and props is not None and props.error is not None and props.error >= threshold:
            out.add(tuple(sorted(int(q) for q in qargs)))
    return out


def device_couplers(target, gate: str = "cz") -> List[Pair]:
    return sorted({tuple(sorted(int(q) for q in qargs)) for qargs in (target[gate] or {}) if qargs is not None})


def restrict_target(target, drop: Iterable[Sequence[int]]):
    """A copy of a Qiskit Target without the two-qubit couplers in `drop` (both directions); every other
    instruction keeps its calibrated error and duration, so Qiskit's noise-aware passes see the same numbers."""
    from qiskit.transpiler import Target

    drop = {tuple(sorted(int(x) for x in e)) for e in drop}
    new = Target(num_qubits=target.num_qubits, dt=target.dt, qubit_properties=target.qubit_properties,
                 concurrent_measurements=getattr(target, "concurrent_measurements", None))
    for name in target.operation_names:
        op, props = target.operation_from_name(name), target[name]
        if isinstance(op, type) or props is None or None in props:     # control flow: no qargs
            new.add_instruction(op, name=name)
            continue
        keep = {q: p for q, p in props.items() if not (len(q) == 2 and tuple(sorted(q)) in drop)}
        new.add_instruction(op, keep, name=name)
    return new


@functools.lru_cache(maxsize=1)
def _heavy_hex_sets():
    t = fez().target
    bad = failed_couplers(t)
    return restrict_target(t, bad), t, frozenset(bad)


def heavy_hex_coupler_sets():
    """The Qiskit arm's two views of the heavy-hex device: {"full": FakeFez as shipped, its published
    calibration as-is (the primary Qiskit arm, owner 15 Nov: the rival gets the pure calibration), "working":
    the same Target without the failed couplers (our cut, a labeled extra arm only)}, and the failed couplers.
    Built once per process."""
    working, full, bad = _heavy_hex_sets()
    return {"working": working, "full": full}, set(bad)


@functools.lru_cache(maxsize=1)
def heavy_hex_target():
    """The 4colors arm's target for the heavy-hex posts: `fourcolors.quantum.targets.from_qiskit(FakeFez())`, a
    `DeviceTarget` with FakeFez's description (156 qubits, 176 couplers, cz) and its published calibration (the
    failed couplers carry error 1.0). The same device and the same numbers as Qiskit's "full" arm."""
    from fourcolors.quantum import targets

    return targets.from_qiskit(fez())


def custom_target(n_qubits: int, couplers: Iterable[Sequence[int]], two_qubit_gate: str = "cz",
                  gates_1q: Sequence[str] = ("rz", "sx", "x")) -> dict:
    """A custom target description in the format the client sends (format_version 2, built by
    `fourcolors.quantum.targets.build_description`): undirected couplers listed once each, sorted, or the
    string "all_to_all"."""
    from fourcolors.quantum import targets

    if couplers == "all_to_all":
        pairs: Any = "all_to_all"
    else:
        pairs = sorted({tuple(sorted(int(x) for x in e)) for e in couplers})
    return targets.build_description(int(n_qubits), two_qubit_gate, gates_1q, pairs)


def qiskit_target_from_custom(custom: dict):
    """The Qiskit Target with the same qubits, couplers (both directions) and gates (the two-qubit gate and
    `gates_1q`, default rz, sx, x) as a custom target (format_version 2), for the Qiskit arm. No error rates: both arms see only the map."""
    from qiskit.transpiler import CouplingMap, Target

    gate = custom.get("two_qubit_gate", "cz")
    gates_1q = list(custom.get("gates_1q") or ("rz", "sx", "x"))
    couplers = custom["couplers"]
    if couplers == "all_to_all":
        n = custom["n_qubits"]
        couplers = [[a, b] for a in range(n) for b in range(a + 1, n)]
    cm = CouplingMap([[int(a), int(b)] for a, b in couplers] + [[int(b), int(a)] for a, b in couplers])
    return Target.from_configuration(basis_gates=[gate, *gates_1q, "measure"],
                                     num_qubits=int(custom["n_qubits"]), coupling_map=cm)


def is_connected(n: int, couplers: Iterable[Sequence[int]]) -> bool:
    adj: Dict[int, set] = {q: set() for q in range(n)}
    for a, b in couplers:
        adj[a].add(b)
        adj[b].add(a)
    seen, todo = {0}, [0]
    while todo:
        for nb in adj[todo.pop()]:
            if nb not in seen:
                seen.add(nb)
                todo.append(nb)
    return len(seen) == n


def _err_dur(props) -> Optional[dict]:
    if props is None:
        return None
    out = {}
    if props.error is not None:
        out["error"] = min(1.0, max(0.0, float(props.error)))
    if props.duration is not None:
        out["duration_s"] = float(props.duration)
    return out or None


def calibration_from_target(target, couplers: Optional[Iterable[Sequence[int]]] = None,
                            source: str = "Qiskit FakeFez snapshot") -> dict:
    """The service's `calibration` (format_version 1) from a Qiskit Target's error rates and durations:
    t1/t2, readout, sx/x per qubit, and the cz error per coupler (the worse direction). `couplers` limits
    the coupler list to the compile target's own (the service refuses others); a coupler at error >= 0.999
    is also marked `available: false`."""
    keep = None if couplers is None else {tuple(sorted(int(x) for x in e)) for e in couplers}
    qubits = []
    for q in range(target.num_qubits):
        e: Dict[str, Any] = {"qubit": q}
        qp = target.qubit_properties[q] if target.qubit_properties else None
        if qp is not None and qp.t1:
            e["t1_s"] = float(qp.t1)
        if qp is not None and qp.t2:
            e["t2_s"] = float(qp.t2)
        if "measure" in target.operation_names:
            ro = _err_dur(target["measure"].get((q,)))
            if ro:
                e["readout"] = ro
        gates = {g: d for g in ("sx", "x") if g in target.operation_names
                 for d in [_err_dur(target[g].get((q,)))] if d}
        if gates:
            e["gates"] = gates
        qubits.append(e)
    by_pair: Dict[Pair, List[dict]] = {}
    for qargs, props in target["cz"].items():
        by_pair.setdefault(tuple(sorted(qargs)), []).append(_err_dur(props) or {})
    out_couplers = []
    for pair, dirs in sorted(by_pair.items()):
        if keep is not None and pair not in keep:
            continue
        c: Dict[str, Any] = {"qubits": list(pair)}
        errs = [d["error"] for d in dirs if "error" in d]
        durs = [d["duration_s"] for d in dirs if "duration_s" in d]
        if errs:
            c["error"] = max(errs)
        if durs:
            c["duration_s"] = max(durs)
        if errs and max(errs) >= FAILED_ERROR:
            c["available"] = False
        out_couplers.append(c)
    return {"format_version": 1, "source": source, "qubits": qubits, "couplers": out_couplers}


def estimated_esp(qc, target) -> Optional[float]:
    """Estimated success probability: the product of (1 - error) over every gate and measurement that has a
    calibrated error in `target` (rz is virtual, error 0). A gate on a failed element gives 0. None if no
    operation had an error rate. An estimate from a snapshot, never a measured fidelity."""
    logp, seen = 0.0, 0
    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(qc.find_bit(q).index for q in inst.qubits)
        try:
            props = target[name].get(qargs) if name in target.operation_names else None
        except (KeyError, TypeError):
            props = None
        if props is None or props.error is None:
            continue
        seen += 1
        e = min(max(float(props.error), 0.0), 1.0)
        if e >= FAILED_ERROR:
            return 0.0
        logp += math.log1p(-e)
    return math.exp(logp) if seen else None


def lattice_positions(n: int, couplers: Iterable[Sequence[int]]) -> Dict[int, Tuple[float, float]]:
    """(x, y) per qubit for drawing a heavy-hex device numbered row by row (IBM Heron): rows are runs of
    consecutive ids joined by (i, i+1) couplers, the other qubits are bridges drawn between their rows.
    Any other numbering falls back to a fixed spring layout."""
    und = {tuple(sorted(int(x) for x in e)) for e in couplers}
    nbr: Dict[int, set] = {q: set() for q in range(n)}
    for a, b in und:
        nbr[a].add(b)
        nbr[b].add(a)
    rows, cur = [], [0]
    for q in range(1, n):
        if (q - 1, q) in und:
            cur.append(q)
        else:
            rows.append(cur)
            cur = [q]
    rows.append(cur)
    main = [r for r in rows if len(r) > 4]
    bridges = [q for r in rows if len(r) <= 4 for q in r]
    pos: Dict[int, Tuple[float, float]] = {}
    if len(main) >= 2:
        for ri, r in enumerate(main):
            for c, q in enumerate(r):
                pos[q] = (float(c), 2.0 * ri)
        for q in bridges:
            known = [pos[x] for x in nbr[q] if x in pos]
            if known:
                pos[q] = (sum(p[0] for p in known) / len(known), sum(p[1] for p in known) / len(known))
        if len(pos) == n:
            return pos
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(und)
    return {q: (float(x), float(y)) for q, (x, y) in nx.spring_layout(g, seed=1).items()}


# ---- the two arms ---------------------------------------------------------------------------------

def qiskit_trials(qc, target, budget_s: float, seed: int, failed: Iterable[Sequence[int]] = (),
                  max_trials: Optional[int] = None, keep_circuits: bool = False, esp_target=None) -> List[dict]:
    """Qiskit's arm: transpile(optimization_level=3) with seed_transpiler = 1000 * seed + k for k = 0, 1, ...
    until `budget_s` seconds of wall time are spent (at least one trial). Each trial records its seed, its
    own time, the time spent so far and its metrics. `best_within` then picks the best for any budget."""
    from qiskit import transpile

    trials: List[dict] = []
    spent = 0.0
    for k in range(10 ** 6):
        s = 1000 * int(seed) + k
        t0 = time.perf_counter()
        out = transpile(qc, target=target, optimization_level=3, seed_transpiler=s)
        dt = time.perf_counter() - t0
        spent += dt
        trial = {"qiskit_seed": s, "trial_s": dt, "cumulative_s": spent, **circuit_metrics(out, failed)}
        if esp_target is not None:
            trial["estimated_esp"] = estimated_esp(out, esp_target)
        if keep_circuits:
            trial["circuit"] = out
        trials.append(trial)
        if spent >= budget_s or (max_trials is not None and k + 1 >= max_trials):
            break
    return trials


def best_within(trials: List[dict], budget_s: float) -> dict:
    """The best trial (fewest two-qubit gates, then two-qubit depth, then depth) among those finished within
    `budget_s`; the first trial if even that one took longer (then `over_budget` is True)."""
    ok = [t for t in trials if t["cumulative_s"] <= budget_s]
    over = not ok
    pick = min(ok or trials[:1], key=lambda t: (t["two_qubit_gates"], t["two_qubit_depth"], t["depth"],
                                                t["cumulative_s"]))
    return {**pick, "over_budget": over, "trials_in_budget": len(ok)}


def result_fields(res, failed: Iterable[Sequence[int]] = (), keep_circuit: bool = False) -> dict:
    """Row fields from a `fourcolors.quantum.compiler` result. The metrics are `circuit_metrics` on
    `res.to_qiskit()`; the service's own numbers are kept beside them as `service_metrics`."""
    v = res.verification
    out: Dict[str, Any] = {
        "status": res.status, "reason_code": (res.reason or {}).get("code") if isinstance(res.reason, dict) else None,
        "verification_label": v.label, "verification_distance": v.distance, "verification_reason": v.reason,
        "engine_version": (res.engine or {}).get("version"), "engine": dict(res.engine or {}),
        "client_version": client_version(), "billed_s": res.billed_s,
        "job_id": res.job.id if res.job is not None else None, "service_seed": res.seed,
        "service_metrics": dict(res.metrics or {})}
    if res.circuit is not None:
        qc = res.to_qiskit()
        out.update(circuit_metrics(qc, failed))
        if keep_circuit:
            out["circuit"] = qc
    return out


def refusal_fields(err) -> dict:
    """Row fields for a compile the service refused or failed (JobFailed, InvalidRequest): kept as a row,
    never dropped. A refusal is not billed."""
    job = getattr(err, "job", None)
    result = getattr(job, "result", None)
    return {"status": "refused", "refusal_code": err.code, "refusal_message": str(err.message)[:400],
            "refusal_field": err.field, "job_id": getattr(job, "id", None),
            "limit": result.get("limit") if isinstance(result, dict) else None, "billed_s": 0.0}


def compile_4colors(qc, *, target, time_budget_s: float, seed: int, failed: Iterable[Sequence[int]] = (),
                    verify: str = "skip", client=None, keep_circuit: bool = False, **options):
    """The 4colors arm: `fourcolors.quantum.compiler.compile` and its row fields (`result_fields`), or the
    refusal's (`refusal_fields`). Returns (result or None, fields). Errors that are not about this circuit
    (no key, no credit, rate limits, network) are raised, not recorded."""
    import fourcolors
    from fourcolors.errors import InvalidRequest, JobFailed

    t0 = time.perf_counter()
    try:
        res = fourcolors.quantum.compiler.compile(qc, target=target, time_budget_s=time_budget_s, seed=seed,
                                                  verify=verify, client=client, **options)
    except (JobFailed, InvalidRequest) as err:
        return None, {**refusal_fields(err), "wall_s": time.perf_counter() - t0}
    return res, {**result_fields(res, failed, keep_circuit), "wall_s": time.perf_counter() - t0}


def row(**fields) -> dict:
    """A result row with every ROW_KEYS key present."""
    return {**{k: None for k in ROW_KEYS}, **fields}


def circuit_fields(qc, circuit_id: str, family: str) -> dict:
    return {"circuit_id": circuit_id, "family": family, "qubits": qc.num_qubits,
            "input_2q": circuit_metrics(qc)["two_qubit_gates"]}


def qiskit_rows(qc, coupler_sets: Dict[str, Any], budgets: Sequence[float], seed: int, *, post: str,
                circuit: dict, failed: Iterable[Sequence[int]] = (), max_trials: Optional[int] = None,
                with_counts: bool = False, esp_target=None) -> List[dict]:
    """Every Qiskit row for one circuit and seed: the trials ("trial" rows) on each coupler set, run once up to
    the largest budget, then for each budget the best within it on each set ("result" rows) and the lower of
    the sets (coupler_set "lower_of_both", `from_coupler_set` says which)."""
    rows: List[dict] = []
    best: Dict[Tuple[str, float], dict] = {}
    for name, target in coupler_sets.items():
        trials = qiskit_trials(qc, target, max(budgets), seed, failed, max_trials, keep_circuits=with_counts,
                               esp_target=esp_target)
        for t in trials:
            rows.append(row(kind="trial", post=post, **circuit, arm="qiskit", coupler_set=name, seed=seed,
                            budget_s=max(budgets), wall_s=t["trial_s"],
                            **{k: v for k, v in t.items() if k not in ("circuit", "trial_s")}))
        for b in budgets:
            pick = best_within(trials, b)
            extra = {"coupler_counts": coupler_counts(pick["circuit"])} if with_counts else {}
            r = row(kind="result", post=post, **circuit, arm="qiskit", coupler_set=name, seed=seed, budget_s=b,
                    status="finished", engine_version=qiskit_version(), wall_s=pick["cumulative_s"],
                    **{k: v for k, v in pick.items() if k not in ("circuit", "trial_s", "cumulative_s")}, **extra)
            best[(name, b)] = r
            rows.append(r)
    if len(coupler_sets) > 1:
        for b in budgets:
            cands = [best[(n, b)] for n in coupler_sets]
            low = min(cands, key=lambda r: (r["two_qubit_gates"], r["two_qubit_depth"], r["depth"]))
            rows.append({**low, "coupler_set": "lower_of_both", "from_coupler_set": low["coupler_set"]})
    return rows


# ---- run metadata and arguments -------------------------------------------------------------------

def qiskit_version() -> str:
    import qiskit

    return f"qiskit {qiskit.__version__}"


def client_version() -> Optional[str]:
    """The installed `fourcolors` client's version (examplekit.client_version)."""
    import examplekit

    return examplekit.client_version()


def meta_row(post: str, args: argparse.Namespace) -> dict:
    """The first row of every run: versions, host CPU and the arguments (no key, no environment)."""
    import qiskit

    fc = client_version()
    return {"kind": "meta", "post": post, "argv": sys.argv[1:], "args": vars(args), "python": platform.python_version(),
            "qiskit": qiskit.__version__, "fourcolors": fc, "machine": platform.machine(),
            "processor": platform.processor() or None, "cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}


def floats(text: str) -> List[float]:
    return [float(x) for x in str(text).replace(",", " ").split()]


def ints(text: str) -> List[int]:
    return [int(x) for x in str(text).replace(",", " ").split()]


def compiler_parser(description: str, *, budgets: str, seeds: str = "1,2,3,4,5", sizes: Optional[str] = None):
    """--out DIR, --budgets, --seeds, --sizes (comma lists), --max-qiskit-trials, --skip-qiskit, --skip-4colors."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", required=True, help="directory for results.jsonl")
    p.add_argument("--budgets", type=floats, default=floats(budgets), help=f"seconds per compile (default {budgets})")
    p.add_argument("--seeds", type=ints, default=ints(seeds), help=f"seeds, both arms (default {seeds})")
    if sizes is not None:
        p.add_argument("--sizes", type=ints, default=ints(sizes), help=f"circuit sizes (default {sizes})")
    p.add_argument("--max-qiskit-trials", type=int, default=None,
                   help="cap on Qiskit seeds per budget (default: seeds until the budget is spent)")
    p.add_argument("--skip-qiskit", action="store_true", help="run only the 4colors arm")
    p.add_argument("--skip-4colors", action="store_true", help="run only the Qiskit arm (no API calls)")
    return p


def fourcolors_row(qc, *, post: str, circuit: dict, target, coupler_set: str, seed: int, budget_s: float,
                   failed: Iterable[Sequence[int]] = (), with_counts: bool = False, client=None,
                   arm: str = "4colors", **options):
    """Compile once with the 4colors arm and return (result or None, its "result" row). `options` go to
    `compile` (objective, calibration, approximation_mode, verify, ...); the row records them, the
    calibration only as `calibration_given` (a `DeviceTarget` sends its own calibration)."""
    res, f = compile_4colors(qc, target=target, time_budget_s=budget_s, seed=seed, failed=failed, client=client,
                             keep_circuit=with_counts, **options)
    out = f.pop("circuit", None)
    extra = {"coupler_counts": coupler_counts(out)} if out is not None else {}
    shown = {k: v for k, v in options.items() if k != "calibration"}
    return res, row(kind="result", post=post, **circuit, arm=arm, coupler_set=coupler_set, seed=seed,
                    budget_s=budget_s, options=shown,
                    calibration_given="calibration" in options or getattr(target, "calibration", None) is not None,
                    **f, **extra)


# ---- reading and plotting -------------------------------------------------------------------------

def results(rows: List[dict], **match) -> List[dict]:
    """The "result" rows whose fields equal `match`."""
    return [r for r in rows if r.get("kind") == "result" and all(r.get(k) == v for k, v in match.items())]


def table(rows: List[dict], group: Sequence[str], values: Sequence[str] = ("two_qubit_gates", "two_qubit_depth",
                                                                           "gates_on_failed_couplers")) -> str:
    """A Markdown table of the median of `values` per group, with the row count and refusals per group."""
    import examplekit

    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(g) for g in group), []).append(r)
    head = list(group) + ["rows", "refused"] + [f"median {v}" for v in values]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rs = groups[key]
        meds = [examplekit.median([r.get(v) for r in rs]) for v in values]
        refused = sum(1 for r in rs if r.get("refusal_code"))
        lines.append("| " + " | ".join(str(x) for x in (*key, len(rs), refused, *meds)) + " |")
    return "\n".join(lines)


def draw_device(ax, n: int, couplers, counts=None, failed=(), title: str = "", vmax: Optional[float] = None,
                label: str = "two-qubit gates on the coupler"):
    """The device map at `lattice_positions`: unused couplers faint, used ones colored and widened by
    their count (one sequential hue), failed couplers dashed in red ink with an x."""
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import examplekit

    pos = lattice_positions(n, couplers)
    counts = {(a, b): c for a, b, c in (counts or [])}
    bad = {tuple(sorted(e)) for e in failed}
    und = sorted({tuple(sorted(e)) for e in couplers})
    cold = [(pos[a], pos[b]) for a, b in und if counts.get((a, b), 0) == 0 and (a, b) not in bad]
    ax.add_collection(LineCollection(cold, colors=examplekit.GRID, linewidths=1.2, zorder=1))
    hot = [(a, b) for a, b in und if counts.get((a, b), 0) > 0]
    if hot:
        vals = np.array([counts[e] for e in hot], dtype=float)
        top = vmax or float(vals.max())
        cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#0d366b"])
        lc = LineCollection([(pos[a], pos[b]) for a, b in hot], cmap=cmap, norm=Normalize(0, top),
                            linewidths=1.5 + 3.0 * vals / top, zorder=2, capstyle="round")
        lc.set_array(vals)
        ax.add_collection(lc)
        cb = ax.figure.colorbar(lc, ax=ax, fraction=0.025, pad=0.01, panchor=False)
        cb.set_label(label, color=examplekit.INK_2, fontsize=8)
    for a, b in sorted(bad):
        if a in pos and b in pos:
            (x0, y0), (x1, y1) = pos[a], pos[b]
            used = counts.get((a, b), 0)
            ax.plot([x0, x1], [y0, y1], ls="--", lw=2.5 if used else 1.2, color="#e34948", zorder=3)
            ax.plot([(x0 + x1) / 2], [(y0 + y1) / 2], marker="x", ms=8, color="#e34948", zorder=4)
    P = np.array([pos[q] for q in range(n)])
    ax.scatter(P[:, 0], P[:, 1], s=8, color=examplekit.INK_2, zorder=3, linewidths=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    return ax


def stream_rows(qc, *, post: str, circuit: dict, target, budget_s: float, seed: int,
                failed: Iterable[Sequence[int]] = (), client=None, arm: str = "4colors",
                coupler_set: str = HEAVY_HEX_DEVICE,
                **options):
    """Submit one compile and yield an "incumbent" row for each better circuit as it arrives, then one for the
    final result (always last). `t_client_s` is seconds since submit on this machine's clock; `t_since_start_s`
    is this machine's clock minus the job's `started_at` (mixes two clocks). `options` go to `submit`."""
    import fourcolors

    t0 = time.monotonic()
    job = fourcolors.quantum.compiler.submit(qc, target=target, time_budget_s=budget_s, seed=seed,
                                             verify=options.pop("verify", "skip"), client=client, **options)
    for k, res in enumerate(job.incumbents()):
        arrived = time.monotonic() - t0
        started = getattr(res.job, "started_at", None)
        yield row(kind="incumbent", post=post, **circuit, arm=arm, coupler_set=coupler_set, seed=seed,
                  budget_s=budget_s, **result_fields(res, failed), index=k, options=dict(options),
                  t_client_s=arrived, t_since_start_s=(time.time() - started) if started else None)
