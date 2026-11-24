"""The paint shop problem, as a QUBO for the optimizer and as a QAOA circuit for the compiler.

Binary paint shop (BPSP). A line of 2n car bodies passes the paint booth; each of n cars appears twice in the
sequence, and one of its two bodies must be painted white (0), the other red (1). Minimize the number of color
changes along the line. One binary variable per car: x_c is the color of car c's FIRST body; its second body then
gets 1 - x_c. With s_i the color at position i,

    changes(x) = sum_i [s_i != s_{i+1}] = sum_i (s_i + s_{i+1} - 2 s_i s_{i+1}),

and s_i is x_c (first body) or 1 - x_c (second body), so the energy is a QUBO with integer coefficients whose value
is exactly the number of color changes. Flipping every car gives the same count: the optimum is at least two-fold.

As an Ising model (z_c = 1 - 2 x_c, t_i = +z_c on a first body, -z_c on a second): [s_i != s_{i+1}] = (1 - t_i t_{i+1})/2,
so H = const + sum_{c<d} J_cd z_c z_d with J_cd = -1/2 * sum of sign products over the places where c and d sit side by
side. Each car has at most four neighbors in the line, so the coupling graph has degree <= 4: a chain with long-range
jumps. That is the graph the QAOA circuit has to route.

Multi-car paint shop. N bodies in a line, each of a car class k; class k needs exactly q_k of its bodies black (1),
the rest white (0). x_i = 1 if body i is black. Minimize sum_i [x_i != x_{i+1}] subject to one exact row per class,
sum_{i in k} x_i = q_k. The rows go to the optimizer as constraints, not as penalties.

References (the instances here are generated in their style; the papers' data are not public):
* M. Streif, S. Yarkoni, A. Skolik, F. Neukart, M. Leib, "Beating classical heuristics for the binary paint shop
  problem with the quantum approximate optimization algorithm", Phys. Rev. A 104, 012403 (2021).
* S. Yarkoni, A. Alekseyenko, M. Streif, D. Von Dollen, F. Neukart, T. Baeck, "Multi-car paint shop optimization with
  quantum annealing", IEEE QCE 2021, p. 35.
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Word = List[int]


# ---- binary paint shop: instances ----------------------------------------------------------------

def random_word(n: int, seed: int) -> Word:
    """A uniformly random sequence in which each of n cars appears twice, relabeled so cars are numbered in order
    of first appearance (car 0 comes first). The usual random-instance model for BPSP."""
    rnd = random.Random(seed)
    w = [c for c in range(n) for _ in range(2)]
    rnd.shuffle(w)
    relabel: Dict[int, int] = {}
    for c in w:
        relabel.setdefault(c, len(relabel))
    return [relabel[c] for c in w]


def check_word(word: Sequence[int]) -> int:
    """n, after checking that every car 0..n-1 appears exactly twice."""
    n = len(word) // 2
    counts = np.bincount(np.asarray(word, dtype=np.int64), minlength=n)
    if len(word) % 2 or len(counts) != n or not np.all(counts == 2):
        raise ValueError("every car must appear exactly twice")
    return n


def first_flags(word: Sequence[int]) -> List[bool]:
    """True where the position is the car's first body."""
    seen = set()
    out = []
    for c in word:
        out.append(c not in seen)
        seen.add(c)
    return out


def colors(word: Sequence[int], x: Sequence[int]) -> List[int]:
    """The color at every position: x_c on a car's first body, 1 - x_c on its second."""
    return [int(x[c]) if f else 1 - int(x[c]) for c, f in zip(word, first_flags(word))]


def count_changes(seq: Sequence[int]) -> int:
    return int(sum(a != b for a, b in zip(seq, seq[1:])))


def color_changes(word: Sequence[int], x: Sequence[int]) -> int:
    """The objective, judged here from the answer: color changes along the line."""
    return count_changes(colors(word, x))


# ---- binary paint shop: models -------------------------------------------------------------------

def qubo_terms(word: Sequence[int]) -> Tuple[Dict[Tuple[int, int], int], List[int], int]:
    """(quadratic {(c, d): v} with c < d, linear [n], offset): integer coefficients, E(x) = color_changes(x)."""
    n = check_word(word)
    first = first_flags(word)
    quad: Dict[Tuple[int, int], int] = {}
    lin = [0] * n
    off = 0
    # s_i = a_i + b_i x_{c_i}: (a, b) = (0, 1) on a first body, (1, -1) on a second.
    ab = [(0, 1) if f else (1, -1) for f in first]
    for i in range(len(word) - 1):
        c, d = word[i], word[i + 1]
        (a1, b1), (a2, b2) = ab[i], ab[i + 1]
        if c == d:              # a car's two bodies side by side: always one change
            off += 1
            continue
        # s1 + s2 - 2 s1 s2, with s1 = a1 + b1 x_c and s2 = a2 + b2 x_d
        off += a1 + a2 - 2 * a1 * a2
        lin[c] += b1 - 2 * b1 * a2
        lin[d] += b2 - 2 * a1 * b2
        key = (min(c, d), max(c, d))
        quad[key] = quad.get(key, 0) - 2 * b1 * b2
    quad = {k: v for k, v in quad.items() if v}
    return quad, lin, off


def model(word: Sequence[int]):
    """The BPSP as a `fourcolors.qubo` Model (BINARY, minimize): its energy is the number of color changes."""
    import fourcolors.qubo

    quad, lin, off = qubo_terms(word)
    return fourcolors.qubo.to_model(quad, lin, n=len(lin), offset=off)


def qubo_energy(word: Sequence[int], x: Sequence[int]) -> int:
    quad, lin, off = qubo_terms(word)
    return off + sum(l * int(x[i]) for i, l in enumerate(lin)) + sum(v * int(x[a]) * int(x[b])
                                                                     for (a, b), v in quad.items())


def ising_couplings(word: Sequence[int]) -> Tuple[Dict[Tuple[int, int], float], float]:
    """({(c, d): J_cd}, constant) with H(z) = constant + sum J_cd z_c z_d = color changes, z = 1 - 2x. Pairs whose
    sign products cancel get no coupling (and no gate)."""
    n = check_word(word)
    sign = [1 if f else -1 for f in first_flags(word)]
    J: Dict[Tuple[int, int], float] = {}
    const = 0.0
    for i in range(len(word) - 1):
        c, d = word[i], word[i + 1]
        if c == d:
            const += 1.0
            continue
        const += 0.5
        key = (min(c, d), max(c, d))
        J[key] = J.get(key, 0.0) - 0.5 * sign[i] * sign[i + 1]
    del n
    return {k: v for k, v in J.items() if v != 0.0}, const


def ising_energy(J: Dict[Tuple[int, int], float], const: float, z: Sequence[int]) -> float:
    return const + sum(v * z[a] * z[b] for (a, b), v in J.items())


# ---- binary paint shop: references ---------------------------------------------------------------

def greedy(word: Sequence[int]) -> List[int]:
    """The textbook greedy heuristic (the classical baseline in Streif et al.): walk the line; a car's first body
    takes the color of the body before it (white at the start), its second body gets the other color. A labeled
    classical reference, not the optimizer."""
    n = check_word(word)
    x: List[Optional[int]] = [None] * n
    prev = 0
    for c in word:
        if x[c] is None:
            x[c] = prev
            prev = x[c]
        else:
            prev = 1 - x[c]
    return [int(v) for v in x]


def brute_force(word: Sequence[int], max_n: int = 22) -> Tuple[int, List[int]]:
    """(fewest color changes, an optimal x) by enumerating every coloring with car 0 white (the flip symmetry).
    Exact; for n <= max_n."""
    n = check_word(word)
    if n > max_n:
        raise ValueError(f"brute force is for n <= {max_n}")
    first = np.array(first_flags(word))
    w = np.asarray(word)
    best, arg = None, None
    rest = n - 1
    chunk = 1 << min(rest, 16)
    for start in range(0, 1 << rest, chunk):
        idx = np.arange(start, min(start + chunk, 1 << rest), dtype=np.int64)
        bits = ((idx[:, None] >> np.arange(rest)) & 1).astype(np.int8)
        X = np.concatenate([np.zeros((len(idx), 1), dtype=np.int8), bits], axis=1)
        S = np.where(first[None, :], X[:, w], 1 - X[:, w])
        ch = (S[:, 1:] != S[:, :-1]).sum(axis=1)
        k = int(ch.argmin())
        if best is None or int(ch[k]) < best:
            best, arg = int(ch[k]), X[k].tolist()
    return best, [int(v) for v in arg]


# ---- binary paint shop: the QAOA circuit ---------------------------------------------------------

def qaoa_circuit(word: Sequence[int], p: int = 1, gammas=None, betas=None, measure: bool = True):
    """QAOA for the Ising form: H on every car qubit, then p layers of rzz(2 gamma J_cd) on each coupled pair and
    rx(2 beta) on each qubit. One rzz per nonzero J per layer. Angles default to a linear ramp (the post compiles
    circuits; it does not optimize angles, so the compile cost is the same for any angles)."""
    from qiskit import QuantumCircuit

    n = check_word(word)
    J, _ = ising_couplings(word)
    if gammas is None or betas is None:
        gammas = [0.6 * (k + 0.5) / p for k in range(p)]
        betas = [0.6 * (1 - (k + 0.5) / p) for k in range(p)]
    qc = QuantumCircuit(n, name=f"bpsp_n{n}_p{p}")
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for (c, d), v in sorted(J.items()):
            qc.rzz(2.0 * float(g) * v, c, d)
        qc.rx(2.0 * float(b), range(n))
    if measure:
        qc.measure_all()
    return qc


# ---- multi-car paint shop ------------------------------------------------------------------------

def random_multicar(n_bodies: int, n_classes: int, seed: int, black_share: float = 0.5):
    """(classes, quotas): a random class per body (every class used) and, per class, how many of its bodies must be
    black (about black_share of them, at least 1 and at most all but one when the class has two or more bodies)."""
    rnd = random.Random(seed)
    classes = [k % n_classes for k in range(n_bodies)]
    rnd.shuffle(classes)
    quotas = []
    for k in range(n_classes):
        size = classes.count(k)
        q = round(black_share * size)
        if size >= 2:
            q = min(max(q, 1), size - 1)
        quotas.append(int(q))
    return classes, quotas


def multicar_feasible(classes: Sequence[int], quotas: Sequence[int], x: Sequence[int]) -> bool:
    got = [0] * len(quotas)
    for k, v in zip(classes, x):
        got[k] += int(v)
    return got == list(quotas)


def multicar_model(classes: Sequence[int], quotas: Sequence[int]):
    """sum_i [x_i != x_{i+1}] as a QUBO, plus one exact row per class: sum_{i in k} x_i = q_k."""
    import fourcolors.qubo

    n = len(classes)
    quad = {(i, i + 1): -2 for i in range(n - 1)}
    lin = [0] * n
    for i in range(n - 1):
        lin[i] += 1
        lin[i + 1] += 1
    m = fourcolors.qubo.to_model(quad, lin, n=n)
    for k, q in enumerate(quotas):
        m.add_constraint({i: 1 for i, c in enumerate(classes) if c == k}, lo=q, hi=q)
    return m


def multicar_sequence_heuristic(classes: Sequence[int], quotas: Sequence[int]) -> List[int]:
    """A simple sequence-order baseline: each class paints its first q_k bodies black, the rest white. Always
    feasible. A labeled classical reference [CHECK: is this the greedy baseline Yarkoni et al. report?]."""
    left = list(quotas)
    x = []
    for k in classes:
        if left[k] > 0:
            x.append(1)
            left[k] -= 1
        else:
            x.append(0)
    return x


def multicar_brute_force(classes: Sequence[int], quotas: Sequence[int], max_n: int = 18) -> Tuple[int, List[int]]:
    """(fewest switches, an optimal x) over every feasible coloring, by enumeration. Exact; for n <= max_n."""
    n = len(classes)
    if n > max_n:
        raise ValueError(f"brute force is for n <= {max_n}")
    best, arg = None, None
    for bits in itertools.product((0, 1), repeat=n):
        if not multicar_feasible(classes, quotas, bits):
            continue
        v = count_changes(bits)
        if best is None or v < best:
            best, arg = v, list(bits)
    if best is None:
        raise ValueError("no feasible coloring")
    return best, arg
