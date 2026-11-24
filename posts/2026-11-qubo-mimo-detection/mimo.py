"""MIMO detection as an Ising model: channels, constellations, the QUBO builder, and two classical baselines.

Uplink: y = H x + n, H is Nr x Nt complex Gaussian (i.i.d. Rayleigh, CN(0, 1) entries), x holds one QAM symbol
per user, n ~ CN(0, sigma^2 I). Maximum-likelihood (ML) detection is  min_x ||y - H x||^2.

Real-valued form: y_r = [Re y; Im y], H_r = [[Re H, -Im H], [Im H, Re H]], and each real symbol component is an
amplitude in {+-1} (QPSK) or {+-1, +-3} (16-QAM). Amplitudes are written as spins, x_r = T s with
T = I kron [2^(k-1), ..., 2, 1] (k = 1 for QPSK, 2 for 16-QAM), so

    ||y_r - H_r T s||^2 = s' G s - 2 b' s + ||y_r||^2,   G = T' H_r' H_r T,  b = T' H_r' y_r.

With s_i^2 = 1 the diagonal of G is a constant, which leaves the Ising model
    E(s) = sum_{i<j} 2 G_ij s_i s_j - 2 sum_i b_i s_i   (+ const = ||y_r||^2 + trace G),
dense (G is a Gram matrix), with N_t * log2(M) spins.

Baselines (local, labeled, never the 4colors service):
- `mmse`: the linear MMSE estimate sliced to the nearest amplitude. Cheap, and what most receivers run.
- `sphere_decode`: a depth-first Schnorr-Euchner sphere decoder on the QR form. It is exact ML when it finishes;
  it counts the tree nodes it visits and stops at `max_nodes` (then the answer is the best leaf found, flagged).

Bits: each real amplitude carries k Gray-coded bits (the usual QAM labeling). `gray_bits_to_spins` and
`spins_to_gray_bits` are the fixed post-hoc transform between the solver's natural-binary spins and those bits.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

MODS = {"qpsk": 1, "16qam": 2}          # bits per real dimension (k); M = 4**k


def bits_per_dim(mod: str) -> int:
    try:
        return MODS[mod.lower()]
    except KeyError:
        raise ValueError(f"modulation {mod!r}: use one of {sorted(MODS)}") from None


def levels(k: int) -> np.ndarray:
    """The real amplitudes of one dimension, ascending: k=1 -> [-1, 1]; k=2 -> [-3, -1, 1, 3]."""
    return np.arange(-(2 ** k - 1), 2 ** k, 2, dtype=np.int64)


def symbol_energy(k: int) -> float:
    """Average energy of one complex symbol (two real dimensions): QPSK 2, 16-QAM 10."""
    return 2.0 * float(np.mean(levels(k) ** 2))


def t_matrix(n_dims: int, k: int) -> np.ndarray:
    """T with x_r = T s: n_dims rows, n_dims * k columns, weights 2^(k-1) ... 1 per dimension."""
    w = 2 ** np.arange(k - 1, -1, -1)
    return np.kron(np.eye(n_dims, dtype=np.int64), w[None, :].astype(np.int64))


# ---- bits <-> spins <-> amplitudes ------------------------------------------------------------------

def spins_to_amplitudes(s, k: int) -> np.ndarray:
    s = np.asarray(s, dtype=np.int64).reshape(-1, k)
    return s @ (2 ** np.arange(k - 1, -1, -1))


def amplitudes_to_spins(a, k: int) -> np.ndarray:
    """Inverse of spins_to_amplitudes for amplitudes in levels(k)."""
    a = np.asarray(a, dtype=np.int64)
    idx = (a + (2 ** k - 1)) // 2                      # 0 .. 2^k - 1, natural binary
    bits = (idx[:, None] >> np.arange(k - 1, -1, -1)) & 1
    return (2 * bits - 1).reshape(-1)


def spins_to_gray_bits(s, k: int) -> np.ndarray:
    """Natural-binary spins (k per dimension, most significant first) to the Gray-coded data bits."""
    b = ((np.asarray(s, dtype=np.int64).reshape(-1, k) + 1) // 2)
    g = b.copy()
    g[:, 1:] = b[:, 1:] ^ b[:, :-1]
    return g.reshape(-1)


def gray_bits_to_spins(g, k: int) -> np.ndarray:
    g = np.asarray(g, dtype=np.int64).reshape(-1, k)
    b = np.bitwise_xor.accumulate(g, axis=1)
    return (2 * b - 1).reshape(-1)


# ---- instances --------------------------------------------------------------------------------------

@dataclass
class Instance:
    """One channel use, in real-valued form. `bits` are the transmitted Gray bits, `s_true` their spins."""
    nt: int
    nr: int
    mod: str
    snr_db: float
    seed: int
    H: np.ndarray            # 2Nr x 2Nt real
    y: np.ndarray            # 2Nr real
    bits: np.ndarray         # 2 Nt k Gray bits
    s_true: np.ndarray       # 2 Nt k spins
    sigma2: float            # complex noise variance

    @property
    def k(self) -> int:
        return bits_per_dim(self.mod)

    @property
    def n_spins(self) -> int:
        return 2 * self.nt * self.k

    @property
    def name(self) -> str:
        return f"{self.mod}-{self.nt}x{self.nr}-snr{self.snr_db:g}-s{self.seed}"


def real_form(Hc: np.ndarray, yc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = np.block([[Hc.real, -Hc.imag], [Hc.imag, Hc.real]])
    return H, np.concatenate([yc.real, yc.imag])


def generate(nt: int, nr: int, mod: str, snr_db: float, seed: int) -> Instance:
    """A seeded i.i.d. Rayleigh channel use. SNR is the average received symbol energy per receive antenna over
    the noise variance: SNR = Nt * Es / sigma^2 (H entries have unit variance)."""
    k = bits_per_dim(mod)
    rng = np.random.default_rng([seed, nt, nr, k, int(round(10 * snr_db))])
    Hc = (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / math.sqrt(2)
    bits = rng.integers(0, 2, size=2 * nt * k)
    s = gray_bits_to_spins(bits, k)
    x_r = spins_to_amplitudes(s, k)                      # [Re x; Im x]
    xc = x_r[:nt] + 1j * x_r[nt:]
    sigma2 = nt * symbol_energy(k) / 10 ** (snr_db / 10)
    n = (rng.standard_normal(nr) + 1j * rng.standard_normal(nr)) * math.sqrt(sigma2 / 2)
    H, y = real_form(Hc, Hc @ xc + n)
    return Instance(nt, nr, mod.lower(), float(snr_db), seed, H, y, bits, s, float(sigma2))


# ---- the Ising model ---------------------------------------------------------------------------------

def ising_terms(inst: Instance) -> Tuple[Dict[Tuple[int, int], float], List[float], float]:
    """(quadratic {(i, j): 2 G_ij} for i < j, linear [-2 b_i], constant) with E(s) + constant = ||y - H T s||^2."""
    T = t_matrix(2 * inst.nt, inst.k).astype(float)
    A = inst.H @ T
    G = A.T @ A
    b = A.T @ inst.y
    n = G.shape[0]
    iu = np.triu_indices(n, 1)
    quad = {(int(i), int(j)): float(2 * G[i, j]) for i, j in zip(*iu) if G[i, j] != 0}
    const = float(inst.y @ inst.y + np.trace(G))
    return quad, [float(-2 * v) for v in b], const


def residual(inst: Instance, s) -> float:
    """||y - H T s||^2 of a spin answer: the one scorer for every arm."""
    x = spins_to_amplitudes(s, inst.k).astype(float)
    r = inst.y - inst.H @ x
    return float(r @ r)


def model(inst: Instance):
    """The fourcolors Model (SPIN, minimize). Its energy plus `ising_terms(...)[2]` is the residual."""
    import fourcolors.qubo

    quad, lin, _ = ising_terms(inst)
    return fourcolors.qubo.to_model(quad, lin, vartype="SPIN")


def packed_model(insts: List[Instance]):
    """Several independent channel uses as one block-diagonal model (one job). Block b holds spins
    offsets[b] .. offsets[b+1]-1; the blocks do not interact, so each is decoded on its own."""
    import fourcolors.qubo

    quad: Dict[Tuple[int, int], float] = {}
    lin: List[float] = []
    offsets = [0]
    for inst in insts:
        q, l, _ = ising_terms(inst)
        o = offsets[-1]
        quad.update({(i + o, j + o): v for (i, j), v in q.items()})
        lin += l
        offsets.append(o + len(l))
    return fourcolors.qubo.to_model(quad, lin, vartype="SPIN"), offsets


# ---- baselines ----------------------------------------------------------------------------------------

def slice_to_levels(x: np.ndarray, k: int) -> np.ndarray:
    lv = levels(k)
    return lv[np.abs(np.asarray(x)[:, None] - lv[None, :]).argmin(axis=1)]


def mmse(inst: Instance) -> np.ndarray:
    """Linear MMSE estimate, sliced; returns spins. Real form: noise sigma^2/2 and energy Es/2 per dimension."""
    k = inst.k
    reg = inst.sigma2 / symbol_energy(k)
    HtH = inst.H.T @ inst.H
    x = np.linalg.solve(HtH + reg * np.eye(HtH.shape[0]), inst.H.T @ inst.y)
    return amplitudes_to_spins(slice_to_levels(x, k), k)


def zero_forcing(inst: Instance) -> np.ndarray:
    k = inst.k
    x = np.linalg.lstsq(inst.H, inst.y, rcond=None)[0]
    return amplitudes_to_spins(slice_to_levels(x, k), k)


def sphere_decode(inst: Instance, max_nodes: int = 2_000_000) -> dict:
    """Schnorr-Euchner depth-first sphere decoder (exact ML when `complete`). Needs Nr >= Nt.

    Returns {spins, residual, nodes, complete, wall_s}. `nodes` counts every tree node visited (a tree node is one
    amplitude tried at one level); it does not depend on the machine, unlike `wall_s`."""
    k = inst.k
    lv = levels(k).astype(float)
    t0 = time.perf_counter()
    Q, R = np.linalg.qr(inst.H)                   # H = Q R, R is m x m upper triangular
    z = Q.T @ inst.y
    const = float(inst.y @ inst.y - z @ z)
    m = R.shape[0]
    best = [math.inf, None]
    x = np.zeros(m)
    nodes = [0]
    complete = True

    def descend(level: int, partial: float):
        nonlocal complete
        # interference from the already fixed levels below (indices > level)
        c = z[level] - R[level, level + 1:] @ x[level + 1:]
        center = c / R[level, level]
        order = lv[np.argsort(np.abs(lv - center), kind="stable")]
        for a in order:
            if nodes[0] >= max_nodes:
                complete = False
                return
            nodes[0] += 1
            d = partial + (c - R[level, level] * a) ** 2
            if d >= best[0]:
                return                            # Schnorr-Euchner order: the rest are farther still
            x[level] = a
            if level == 0:
                best[0], best[1] = d, x.copy()
            else:
                descend(level - 1, d)
        x[level] = 0.0

    descend(m - 1, 0.0)
    wall = time.perf_counter() - t0
    amp = best[1].astype(np.int64)
    s = amplitudes_to_spins(amp, k)
    return {"spins": s, "residual": float(best[0] + const), "nodes": nodes[0], "complete": complete,
            "wall_s": wall}


def brute_force(inst: Instance) -> Tuple[float, np.ndarray]:
    """Exact ML by trying every spin vector (tests only: n_spins <= 16)."""
    n = inst.n_spins
    if n > 16:
        raise ValueError("brute force is for at most 16 spins")
    best, arg = math.inf, None
    for code in range(1 << n):
        s = 2 * ((code >> np.arange(n)) & 1) - 1
        r = residual(inst, s)
        if r < best - 1e-12:
            best, arg = r, s
    return best, arg


# ---- scoring -------------------------------------------------------------------------------------------

def score(inst: Instance, s) -> dict:
    """Bit errors (Gray bits), symbol errors (complex symbols) and the residual of a spin answer."""
    s = np.asarray(s, dtype=np.int64)
    k = inst.k
    bits = spins_to_gray_bits(s, k)
    amp = spins_to_amplitudes(s, k)
    amp_true = spins_to_amplitudes(inst.s_true, k)
    nt = inst.nt
    sym_err = int(np.sum((amp[:nt] != amp_true[:nt]) | (amp[nt:] != amp_true[nt:])))
    return {"bit_errors": int(np.sum(bits != inst.bits)), "n_bits": int(inst.bits.size),
            "symbol_errors": sym_err, "n_symbols": nt, "residual": residual(inst, s)}


def wilson(errors: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson interval of an error rate (used by the chart's selection rule)."""
    if n == 0:
        return (0.0, 1.0)
    p = errors / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def bits_str(s, k: int) -> str:
    return "".join(map(str, spins_to_gray_bits(s, k)))
