"""The post's circuits, one domain observable each, all rebuilt from their circuit id (seeded, deterministic).

    qc, spec = build("uccsd_12")       # the logical circuit (measured) and what to read from its output

Families (8-12 qubits, so the ideal answer is an exact statevector and the noisy one a density matrix):

* chemistry  `uccsd_8`, `uccsd_12`: UCCSD on 4 spatial orbitals / 2 electrons and on the LiH STO-3G register
  (6 / 4), Hartree-Fock reference, fixed seeded amplitudes (`circuitkit.uccsd`). Observable: the spin-orbital
  occupations; the error is the largest occupation error. Also recorded: the probability of the right particle
  number and spin sector.
* finance  `portfolio_8`, `portfolio_12`: portfolio QAOA with an XY ring mixer, k = n/2 assets out of n, two
  layers, fixed angles, started from the Dicke state (every k-asset portfolio equally). Observable: the probability
  that a sample is feasible (exactly k assets); ideal 1, since the XY mixer and the cost layer conserve k. Also
  recorded: the mean cost of the feasible samples.
* materials  `tfim_3x3_s<steps>`, `tfim_3x4_s<steps>`, steps 1-8: first-order Trotter steps of the 2D transverse
  field Ising model from |0...0>. Observable: the mean magnetization <Z>; the error is |m - m_ideal|.

Every circuit must do something to |0...0>: `trivial_reason` flags a circuit whose ideal output is |0...0> again
(the identity on the vacuum, the old UCCSD and portfolio generators' bug) or one single basis state; `build`
refuses such a circuit. A trivial circuit makes an observable study meaningless, and lets an approximate compiler
delete the whole circuit.
"""
from __future__ import annotations

import itertools
import math
import os
import random
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import circuitkit as ck  # noqa: E402  (../circuitkit.py: the shared UCCSD builder)


#: TFIM parameters: H = -J sum ZZ - h sum X, Trotter step dt.
TFIM_J, TFIM_H, TFIM_DT = 1.0, 2.0, 0.15
TFIM_STEPS = tuple(range(1, 9))
#: Portfolio QAOA: risk aversion q, two layers of fixed angles (not optimized).
PORTFOLIO_Q = 0.5
PORTFOLIO_ANGLES = ((0.4, 0.8), (0.6, 0.3))          # (gammas, betas)
#: The default circuit set of the post.
POST_CIRCUITS = (["uccsd_8", "uccsd_12", "portfolio_8", "portfolio_12"]
                 + [f"tfim_3x3_s{s}" for s in TFIM_STEPS] + [f"tfim_3x4_s{s}" for s in TFIM_STEPS])
FAMILY = {"uccsd": "chemistry", "portfolio": "finance", "tfim": "materials"}


# ---- triviality check -------------------------------------------------------------------------------

def ideal_probabilities(qc) -> Dict[str, float]:
    """Exact output probabilities of the circuit without its measurements, keys with qubit 0 FIRST."""
    from qiskit.quantum_info import Statevector

    sv = Statevector(qc.remove_final_measurements(inplace=False))
    return {k[::-1]: float(v) for k, v in sv.probabilities_dict().items() if v > 1e-14}


def trivial_reason(qc, tol: float = 1e-6) -> Optional[str]:
    """Why the circuit is trivial on |0...0>, or None: "identity on |0>" if the output is |0...0> again,
    "single basis state" if one basis state carries almost all the probability."""
    probs = ideal_probabilities(qc)
    if probs.get("0" * qc.num_qubits, 0.0) > 1 - tol:
        return "identity on |0>"
    if max(probs.values()) > 1 - tol:
        return "single basis state"
    return None


# ---- finance: portfolio QAOA with an XY mixer -------------------------------------------------------

def portfolio_data(n: int, seed: int) -> Tuple[List[float], List[List[float]]]:
    """Expected returns mu (uniform 0-1) and a covariance A A^T / n (A Gaussian), from `seed`."""
    rng = random.Random(seed)
    mu = [rng.random() for _ in range(n)]
    a = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    sigma = [[sum(a[i][t] * a[j][t] for t in range(n)) / n for j in range(n)] for i in range(n)]
    return mu, sigma


def portfolio_cost(x: Sequence[int], mu, sigma, q: float = PORTFOLIO_Q) -> float:
    """Mean-variance cost q x^T Sigma x - mu^T x of a 0/1 portfolio x."""
    n = len(x)
    return q * sum(sigma[i][j] * x[i] * x[j] for i in range(n) for j in range(n)) - sum(mu[i] * x[i] for i in range(n))


def ising_terms(mu, sigma, q: float = PORTFOLIO_Q):
    """The cost as sum a_ij Z_i Z_j + sum b_i Z_i + const (x = (1 - z) / 2), scaled so max |coefficient| = 1."""
    n = len(mu)
    zz: Dict[Tuple[int, int], float] = {}
    z = [0.0] * n
    for i in range(n):
        c = q * sigma[i][i] - mu[i]                 # c x_i
        z[i] += -c / 2
        for j in range(i + 1, n):
            c = 2 * q * sigma[i][j]                  # c x_i x_j
            zz[(i, j)] = c / 4
            z[i] += -c / 4
            z[j] += -c / 4
    scale = max([abs(v) for v in zz.values()] + [abs(v) for v in z])
    return {k: v / scale for k, v in zz.items()}, [v / scale for v in z], scale


def dicke_state(n: int, k: int):
    """Bartschi-Eidenbenz Dicke-state preparation: the equal superposition of all n-bit strings of weight k."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate

    qc = QuantumCircuit(n, name=f"dicke_{n}_{k}")
    qc.x(range(n - k, n))

    def scs(m: int, l: int):                         # split and cyclic shift on qubits [m - l - 1, m - 1]
        top = m - 1
        qc.cx(top - 1, top)
        qc.cry(2 * math.acos(math.sqrt(1 / m)), top, top - 1)
        qc.cx(top - 1, top)
        for i in range(2, l + 1):
            qc.cx(top - i, top)
            qc.append(RYGate(2 * math.acos(math.sqrt(i / m))).control(2, annotated=False), [top, top - i + 1, top - i])
            qc.cx(top - i, top)

    for m in range(n, k, -1):
        scs(m, k)
    for m in range(k, 1, -1):
        scs(m, m - 1)
    return qc


def portfolio_xy(n: int, k: int, seed: int, angles=PORTFOLIO_ANGLES, dicke: bool = True):
    """Portfolio QAOA: Dicke(n, k) start (or |0...0> with dicke=False: the old generator's bug), then per layer
    the cost phase (rzz, rz) and the XY ring mixer (XX+YY on even pairs, odd pairs, then the closing pair)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate

    mu, sigma = portfolio_data(n, seed)
    zz, z, _ = ising_terms(mu, sigma)
    qc = QuantumCircuit(n, name=f"portfolio_{n}")
    if dicke:
        qc.compose(dicke_state(n, k), inplace=True)
    ring = [(i, i + 1) for i in range(0, n - 1, 2)] + [(i, i + 1) for i in range(1, n - 1, 2)] + [(n - 1, 0)]
    for gamma, beta in zip(*angles):
        for (i, j), a in zz.items():
            qc.rzz(2 * gamma * a, i, j)
        for i, b in enumerate(z):
            qc.rz(2 * gamma * b, i)
        for i, j in ring:
            qc.append(XXPlusYYGate(2 * beta), [i, j])
    return qc


# ---- materials: 2D TFIM Trotter steps ---------------------------------------------------------------

def grid_edges(rows: int, cols: int) -> List[Tuple[int, int]]:
    q = lambda r, c: r * cols + c
    return ([(q(r, c), q(r, c + 1)) for r in range(rows) for c in range(cols - 1)]
            + [(q(r, c), q(r + 1, c)) for r in range(rows - 1) for c in range(cols)])


def tfim2d(rows: int, cols: int, steps: int, j: float = TFIM_J, h: float = TFIM_H, dt: float = TFIM_DT):
    """First-order Trotter evolution under H = -J sum ZZ - h sum X from |0...0>, `steps` steps of dt."""
    from qiskit import QuantumCircuit

    n = rows * cols
    qc = QuantumCircuit(n, name=f"tfim_{rows}x{cols}_s{steps}")
    for _ in range(steps):
        for a, b in grid_edges(rows, cols):
            qc.rzz(-2 * j * dt, a, b)
        qc.rx(-2 * h * dt, range(n))
    return qc


# ---- the registry -----------------------------------------------------------------------------------

def lower(qc):
    """The textbook input form: cx, rz, sx, x, h (Qiskit at optimization level 0), measured."""
    from qiskit import transpile

    out = transpile(qc, basis_gates=["cx", "rz", "sx", "x", "h"], optimization_level=0)
    out.name = qc.name
    out.measure_all()
    return out


def build(circuit_id: str, *, check: bool = True):
    """(measured logical circuit, observable spec) for a circuit id; refuses a trivial circuit."""
    if m := re.fullmatch(r"uccsd_(8|12)", circuit_id):
        n_spatial, ne = {"8": (4, 2), "12": (6, 4)}[m.group(1)]
        qc = ck.uccsd(n_spatial, ne, seed=1, measure=False)
        spec = {"observable": "occupations", "n_spatial": n_spatial, "n_electrons": ne}
    elif m := re.fullmatch(r"portfolio_(8|12)", circuit_id):
        n = int(m.group(1))
        qc = lower(portfolio_xy(n, n // 2, seed=n)).remove_final_measurements(inplace=False)
        spec = {"observable": "feasibility", "n": n, "k": n // 2, "seed": n}
    elif m := re.fullmatch(r"tfim_(\d)x(\d)_s(\d+)", circuit_id):
        rows, cols, steps = map(int, m.groups())
        qc = lower(tfim2d(rows, cols, steps)).remove_final_measurements(inplace=False)
        spec = {"observable": "magnetization", "rows": rows, "cols": cols, "steps": steps,
                "t": round(steps * TFIM_DT, 6)}
    else:
        raise KeyError(f"unknown circuit id {circuit_id!r}")
    qc.name = circuit_id
    if check and (why := trivial_reason(qc)):
        raise ValueError(f"{circuit_id} is trivial on |0...0>: {why}")
    qc.measure_all()
    spec["family"] = FAMILY[circuit_id.split("_")[0]]
    return qc, spec


# ---- compiled circuits on disk ----------------------------------------------------------------------

def save_qpy(qc, path):
    """Write one circuit as QPY (Qiskit's binary format, exact); returns the file name."""
    from qiskit import qpy

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        qpy.dump(qc, f)
    return os.path.basename(path)


def load_qpy(path):
    from qiskit import qpy

    with open(path, "rb") as f:
        return qpy.load(f)[0]
