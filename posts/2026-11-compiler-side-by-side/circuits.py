"""The twelve input circuits of the side-by-side post (C1-C12), each from a builder fixed by its id.

    import circuits
    qc = circuits.build("bv-n48")            # measurements included (the final layout is read from them)
    circuits.CIRCUITS["bv-n48"]              # {"cid": "C1", "family": ..., "check": "clifford" | "states", ...}

Every input ends with measurements on every logical qubit, so each compiler's output says where each logical
qubit ended up. Two-qubit gates in the inputs: cx (BV), cp (QFT, Draper), rzz (QAOA, Ising, TFIM), rxx/ryy/rzz
(Heisenberg), and the cx of the decomposed Toffolis (Cuccaro). The Heisenberg circuit (C11) is run and recorded
only: no figure uses it.
"""
from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

import examplekit  # noqa: E402
import circuitkit as ck  # noqa: E402

Pair = Tuple[int, int]

#: id -> what it is. `check`: how check.py proves or measures each output ("clifford": the secret comes back
#: with certainty; "states": random product states, a sampled distance; QCEC when installed).
CIRCUITS: Dict[str, dict] = {
    "bv-n48": {"cid": "C1", "family": "bernstein_vazirani", "check": "clifford",
               "why": "a star: the oracle must meet 24 qubits"},
    "bv-n100": {"cid": "C2", "family": "bernstein_vazirani", "check": "clifford", "why": "the same star at scale"},
    "qft-n20": {"cid": "C3", "family": "qft", "check": "states", "why": "all-to-all controlled phases"},
    "qft-n32": {"cid": "C4", "family": "qft", "check": "states", "why": "the upper end of the QFT range"},
    "qaoa3r-n20": {"cid": "C5", "family": "qaoa_3_regular", "check": "states", "why": "random 3-regular graph"},
    "qaoa3r-n32": {"cid": "C6", "family": "qaoa_3_regular", "check": "states", "why": "a larger 3-regular graph"},
    "qaoaba2-n32": {"cid": "C7", "family": "qaoa_barabasi_albert", "check": "states", "why": "hub-heavy graph"},
    "draper-n16": {"cid": "C8", "family": "draper_adder", "check": "states", "why": "QFT adder: depth against count"},
    "cuccaro-n20": {"cid": "C9", "family": "cuccaro_adder", "check": "states", "why": "ripple-carry adder"},
    "tfim2d-n36": {"cid": "C10", "family": "tfim_2d", "check": "states", "why": "2D Trotter step, not heavy-hex"},
    "heis2d-n25": {"cid": "C11", "family": "heisenberg_2d", "check": "states", "why": "XX+YY+ZZ blocks",
                   "record_only": True},
    "ising1d-n32": {"cid": "C12", "family": "ising_1d", "check": "states", "why": "control: a chain fits as is",
                    "control": True},
}


def bv_secret_weight(n: int, weight: int, seed: int) -> str:
    """A secret of n bits with exactly `weight` ones, at positions fixed by `seed`."""
    ones = set(random.Random(seed).sample(range(n), weight))
    return "".join("1" if i in ones else "0" for i in range(n))


def grid_edges(rows: int, cols: int) -> List[Pair]:
    q = lambda r, c: r * cols + c
    return sorted([(q(r, c), q(r, c + 1)) for r in range(rows) for c in range(cols - 1)] +
                  [(q(r, c), q(r + 1, c)) for r in range(rows - 1) for c in range(cols)])


def chain_edges(n: int) -> List[Pair]:
    return [(i, i + 1) for i in range(n - 1)]


def ising_trotter(n: int, edges: Sequence[Pair], steps: int = 1, dt: float = 0.1, h: float = 1.0,
                  xx_yy: bool = False):
    """`steps` first-order Trotter steps of H = sum_edges ZZ (+ XX + YY) + h sum X: per step one rzz per edge
    (and rxx, ryy with `xx_yy`), then rx on every qubit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    for _ in range(steps):
        for a, b in edges:
            if xx_yy:
                qc.rxx(2 * dt, a, b)
                qc.ryy(2 * dt, a, b)
            qc.rzz(2 * dt, a, b)
        qc.rx(2 * h * dt, range(n))
    return qc


def cuccaro_adder(n_bits: int, a: int = 0, b: int = 0):
    """Cuccaro ripple-carry adder (quant-ph/0410184) on 2 n_bits + 2 qubits: qubit 0 the carry-in ancilla,
    then a_i on 1 + 2i and b_i on 2 + 2i, the carry-out on 2 n_bits + 1. |0, a, b, 0> -> |0, a, a + b mod 2^n,
    carry>. `a` and `b` are prepared with X gates; the Toffolis are decomposed into one- and two-qubit gates."""
    from qiskit import QuantumCircuit

    nq = 2 * n_bits + 2
    A = lambda i: 1 + 2 * i
    B = lambda i: 2 + 2 * i
    qc = QuantumCircuit(nq)
    for i in range(n_bits):
        if (a >> i) & 1:
            qc.x(A(i))
        if (b >> i) & 1:
            qc.x(B(i))

    def maj(c, y, x):            # (c, b, a) -> carry into a
        qc.cx(x, y)
        qc.cx(x, c)
        qc.ccx(c, y, x)

    def uma(c, y, x):
        qc.ccx(c, y, x)
        qc.cx(x, c)
        qc.cx(c, y)

    carry = [0] + [A(i) for i in range(n_bits - 1)]
    for i in range(n_bits):
        maj(carry[i], B(i), A(i))
    qc.cx(A(n_bits - 1), nq - 1)
    for i in reversed(range(n_bits)):
        uma(carry[i], B(i), A(i))
    return qc.decompose(["ccx"])


def cuccaro_sum(n_bits: int, bits: str) -> Tuple[int, int]:
    """Read (a + b mod 2^n, carry) from a measured bit string of `cuccaro_adder` (bits[k] = qubit k)."""
    s = sum(int(bits[2 + 2 * i]) << i for i in range(n_bits))
    return s, int(bits[2 * n_bits + 1])


def _measured(qc):
    out = qc.copy()
    if not out.cregs:
        out.measure_all()
    return out


def build(name: str):
    """The input circuit of one id in CIRCUITS, with measurements."""
    if name == "bv-n48":
        return ck.bernstein_vazirani(bv_secret_weight(48, 24, seed=48))
    if name == "bv-n100":
        return ck.bernstein_vazirani(ck.random_secret(100, seed=1))
    if name.startswith("qft-n"):
        return _measured(ck.qft(int(name[5:])))
    if name.startswith("qaoa3r-n"):
        n = int(name[8:])
        return _measured(ck.qaoa_maxcut(n, examplekit.random_regular(n, 3, seed=1)))
    if name == "qaoaba2-n32":
        return _measured(ck.qaoa_maxcut(32, ck.barabasi_albert(32, 2, seed=1)))
    if name == "draper-n16":
        a, b = random.Random(16).randrange(256), random.Random(17).randrange(256)
        return _measured(ck.draper_adder(8, a, b))
    if name == "cuccaro-n20":
        a, b = random.Random(20).randrange(512), random.Random(21).randrange(512)
        return _measured(cuccaro_adder(9, a, b))
    if name == "tfim2d-n36":
        return _measured(ising_trotter(36, grid_edges(6, 6)))
    if name == "heis2d-n25":
        return _measured(ising_trotter(25, grid_edges(5, 5), xx_yy=True))
    if name == "ising1d-n32":
        return _measured(ising_trotter(32, chain_edges(32), steps=2))
    raise KeyError(f"unknown circuit {name!r}; one of {', '.join(CIRCUITS)}")


def control_floor(qc) -> int:
    """For the native-fit control: the CZ count with no routing, 2 per rzz (a ZZ rotation needs 2 CZ)."""
    return 2 * sum(1 for i in qc.data if i.operation.name == "rzz")
