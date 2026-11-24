"""The three million-variable families of the anytime study, each as one Ising model H(s) = sum c_k s_i s_j.

Specs (the --instance argument), "kind:key=value,...":
    ea3d:L=100,seed=1        3D +-J Edwards-Anderson spin glass, L^3 spins, periodic (examplekit's generator)
    torus2d:L=1000,seed=1    2D +-J spin glass on an L x L periodic square lattice (a torus), L^2 spins
    regular:n=1000000,seed=1 random 3-regular graph, unit-weight Max-Cut (H = sum s_i s_j over the edges)

Every instance is the dict the solver harness of the comparison post uses (`qubo-solver-showdown/instances.py`):
n, i, j, c (int64 arrays, i < j), kind, sense, reference, reference_kind. `instances.energy` is the one evaluator.

References (what a gap is measured against):
- ea3d: the published ground-state energy density -1.7863(4) per spin (K. F. Pal, Physica A 223, 283-292, 1996);
  the sensitivity check uses -1.7876(3) (A. K. Hartmann, Europhys. Lett. 40, 429-434, 1997, power-law fit).
- torus2d: -1.4015(8) per spin (C. De Simone, M. Diehl, M. Junger et al., "Exact ground states of
  two-dimensional +-J Ising spin glasses", J. Stat. Phys. 84 (1996), doi 10.1007/BF02174135; exact ground states,
  extrapolated to the infinite lattice).
- regular: no published constant is used; the reference is the best cut any arm found in the same run, labeled
  "best found (may be ours)". A density reference is for a family, not a proved optimum of one instance: a finite
  instance can land slightly above or below it.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(next(iter(sorted(__import__("pathlib").Path(__file__).resolve().parents[1].glob("*-qubo-solver-showdown"))))))

import sys as _sys                           # noqa: E402
from pathlib import Path as _Path             # noqa: E402
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import examplekit as bk           # noqa: E402
import instances as I          # noqa: E402  (the comparison post's instance module: one evaluator)

E3D_PAL1996 = bk.EA3D_E0_PAL1996          # -1.7863
E3D_HARTMANN1997 = bk.EA3D_E0_LATER       # -1.7876, the sensitivity check
E2D_DESIMONE1996 = -1.4015

REFERENCES = {
    "ea3d": ("ground-state density, Pal 1996", E3D_PAL1996),
    "torus2d": ("ground-state density, De Simone et al. 1996", E2D_DESIMONE1996),
    "regular": ("best found by any arm (may be ours)", None),
}
ALT_REFERENCES = {"ea3d": ("ground-state density, Hartmann 1997", E3D_HARTMANN1997)}


def torus2d(L: int, seed: int):
    """(i, j, c) of the 2D +-J model H = -sum J s_i s_j on an L x L torus; site i = y * L + x, bonds to +x, +y."""
    if L < 3:
        raise ValueError("L must be at least 3")
    rng = np.random.default_rng([seed, 2])
    site = np.arange(L * L, dtype=np.int64)
    x, y = site % L, site // L
    right = y * L + (x + 1) % L
    down = ((y + 1) % L) * L + x
    J = rng.choice(np.array([-1, 1], dtype=np.int64), size=2 * L * L)
    u = np.concatenate([site, site])
    v = np.concatenate([right, down])
    return I._merge(L * L, u, v, -J)


def random_cubic(n: int, seed: int, max_tries: int = 200):
    """Edges (u, v) of a uniformly random simple 3-regular graph on n nodes (pairing model, numpy, retried).

    A pairing is simple with probability about exp(-2), so a handful of tries is typical at any n."""
    if n % 2 or n < 4:
        raise ValueError("a 3-regular graph needs an even n >= 4")
    rng = np.random.default_rng([seed, 3])
    stubs = np.repeat(np.arange(n, dtype=np.int64), 3)
    for _ in range(max_tries):
        p = rng.permutation(stubs).reshape(-1, 2)
        a, b = p.min(axis=1), p.max(axis=1)
        if np.any(a == b):
            continue
        key = a * n + b
        if len(np.unique(key)) == len(key):
            order = np.argsort(key)
            return a[order], b[order]
    raise RuntimeError("no simple pairing found")


def build(spec: str) -> dict:
    p = I.parse_spec(spec)
    kind = p["kind"]
    seed = int(p.get("seed", 1))
    if kind == "ea3d":
        inst = I.build(spec)
    elif kind == "torus2d":
        L = int(p["L"])
        u, v, c = torus2d(L, seed)
        inst = {"name": f"torus2d-L{L}-s{seed}", "kind": "spinglass", "n": L * L, "i": u, "j": v, "c": c,
                "sense": "min"}
    elif kind == "regular":
        n = int(p["n"])
        u, v = random_cubic(n, seed)
        w = np.ones(len(u), dtype=np.int64)
        inst = I._maxcut(f"regular-n{n}-d3-s{seed}", n, u, v, w, None, None)
    else:
        raise ValueError(f"unknown family {kind!r} (ea3d, torus2d, regular)")
    inst["family"] = kind
    label, density = REFERENCES[kind]
    inst["reference"] = None if density is None else density * inst["n"]
    inst["reference_kind"] = label
    if kind in ALT_REFERENCES:
        alt_label, alt = ALT_REFERENCES[kind]
        inst["reference_alt"], inst["reference_alt_kind"] = alt * inst["n"], alt_label
    inst["sha256"] = fingerprint(inst)
    return inst


def fingerprint(inst: dict) -> str:
    """sha256 of (n, i, j, c) as little-endian int64: the instance's identity in every row and in the run record."""
    h = hashlib.sha256()
    h.update(np.int64(inst["n"]).tobytes())
    for k in ("i", "j", "c"):
        h.update(np.ascontiguousarray(inst[k], dtype="<i8").tobytes())
    return h.hexdigest()


def energy_per_spin(inst: dict, s) -> float:
    return I.energy(inst, s) / inst["n"]


def score(inst: dict, s) -> dict:
    """Energy, the native objective and the gap(s) of one answer, re-evaluated here from the spins."""
    s = np.asarray(s, dtype=np.int64)
    if s.shape != (inst["n"],) or not np.isin(s, (-1, 1)).all():
        raise ValueError("an answer is n values in {-1, +1}")
    h = I.energy(inst, s)
    obj = I.objective(inst, s)
    out = {"energy": h, "objective": obj, "energy_per_spin": h / inst["n"], "gap": I.gap(inst, obj)}
    if inst.get("reference_alt") is not None:
        out["gap_alt"] = (obj - inst["reference_alt"]) / abs(inst["reference_alt"])
    return out
