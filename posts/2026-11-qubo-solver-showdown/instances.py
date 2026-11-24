"""Benchmark instances, all held in one form: an Ising model H(s) = Σ_k c_k s_i s_j to minimize.

Specs (the --instance argument), "kind:key=value,key=value":
    gset:path=G22.txt,ref=13359     a Gset-style Max-Cut file ("n m", then "a b w", 1-based); ref = a best
                                    known cut from the literature, if you have one (optional)
    ea3d:L=20,seed=1                3D +-J spin glass (examplekit); reference -1.7863 per spin (Pal 1996)
    regular:n=10000,d=3,seed=1      random d-regular graph, unit-weight Max-Cut
    sparse:n=100000,deg=8,seed=1    random sparse graph with about deg * n / 2 edges, unit-weight Max-Cut
    (regular and sparse take w=pm1 for random +-1 weights)

Max-Cut with weights w: cut(s) = (W - Σ w s_i s_j) / 2 with W = Σ w, so c = w and maximizing the cut is
minimizing H. Spin glass: H = -Σ J s_i s_j, so c = -J. `objective(inst, s)` gives the native number
(the cut, or the energy) and `gap` compares it with the reference in the right direction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import examplekit as bk


def _merge(n, u, v, w):
    """Edges as u < v, duplicates summed, zeros dropped, sorted."""
    u, v, w = np.asarray(u, np.int64), np.asarray(v, np.int64), np.asarray(w, np.int64)
    keep = u != v
    a, b, w = np.minimum(u, v)[keep], np.maximum(u, v)[keep], w[keep]
    key = a * n + b
    uk, inv = np.unique(key, return_inverse=True)
    ws = np.bincount(inv, weights=w).astype(np.int64)
    nz = ws != 0
    return uk[nz] // n, uk[nz] % n, ws[nz]


def read_gset(path):
    with open(path) as fh:
        n, m = map(int, fh.readline().split())
        data = np.loadtxt(fh, dtype=np.int64, ndmin=2)
    if len(data) != m:
        raise ValueError(f"{path}: header says {m} edges, the file has {len(data)}")
    return n, data[:, 0] - 1, data[:, 1] - 1, data[:, 2]


def sparse_random(n: int, deg: float, seed: int):
    """About deg * n / 2 distinct random edges (self-loops and repeats dropped, then topped up)."""
    rng = np.random.default_rng(seed)
    want = int(round(deg * n / 2))
    keys = np.zeros(0, dtype=np.int64)
    while len(keys) < want:
        a = rng.integers(0, n, 2 * (want - len(keys)) + 16)
        b = rng.integers(0, n, len(a))
        k = np.minimum(a, b) * n + np.maximum(a, b)
        keys = np.unique(np.concatenate([keys, k[a != b]]))
    keys = rng.permutation(keys)[:want]
    keys.sort()
    return keys // n, keys % n


def parse_spec(spec: str) -> dict:
    kind, _, rest = spec.partition(":")
    kv = dict(p.split("=", 1) for p in rest.split(",") if p)
    return {"kind": kind, **kv}


def build(spec: str) -> dict:
    p = parse_spec(spec)
    kind = p["kind"]
    if kind == "gset":
        n, u, v, w = read_gset(p["path"])
        u, v, w = _merge(n, u, v, w)
        name = Path(p["path"]).stem
        ref = int(p["ref"]) if "ref" in p else None
        return _maxcut(name, n, u, v, w, ref, "best known (given)" if ref is not None else None)
    if kind == "ea3d":
        L, seed = int(p["L"]), int(p.get("seed", 1))
        r, c, val = bk.ea3d_arrays(L, seed)
        n = L ** 3
        return {"name": f"ea3d-L{L}-s{seed}", "kind": "spinglass", "n": n, "i": r, "j": c, "c": val,
                "sense": "min", "reference": bk.EA3D_E0_PAL1996 * n, "reference_kind": "ground-state density"}
    if kind in ("regular", "sparse"):
        n, seed = int(p["n"]), int(p.get("seed", 1))
        if kind == "regular":
            d = int(p.get("d", 3))
            e = np.array(bk.random_regular(n, d, seed), dtype=np.int64).reshape(-1, 2)
            u, v, label = e[:, 0], e[:, 1], f"regular-n{n}-d{d}-s{seed}"
        else:
            deg = float(p.get("deg", 8))
            u, v = sparse_random(n, deg, seed)
            label = f"sparse-n{n}-deg{deg:g}-s{seed}"
        if p.get("w") == "pm1":
            w = np.random.default_rng(seed + 7).choice([-1, 1], len(u)).astype(np.int64)
            label += "-pm1"
        else:
            w = np.ones(len(u), dtype=np.int64)
        u, v, w = _merge(n, u, v, w)
        return _maxcut(label, n, u, v, w, None, None)
    raise ValueError(f"unknown instance kind {kind!r}")


def _maxcut(name, n, u, v, w, ref, ref_kind):
    return {"name": name, "kind": "maxcut", "n": n, "i": u, "j": v, "c": w, "sense": "max",
            "total_weight": int(w.sum()), "reference": ref, "reference_kind": ref_kind}


def energy(inst: dict, s) -> int:
    """H(s) = Σ c s_i s_j, the one evaluator every answer goes through."""
    return bk.ising_energy(inst["i"], inst["j"], inst["c"], s)


def objective(inst: dict, s) -> int:
    """The native number: the cut (Max-Cut) or the energy (spin glass)."""
    h = energy(inst, s)
    return (inst["total_weight"] - h) // 2 if inst["kind"] == "maxcut" else h


def gap(inst: dict, value, reference=None):
    """Relative gap to the reference, positive when worse; None without a reference."""
    ref = inst["reference"] if reference is None else reference
    if ref is None or value is None:
        return None
    return (ref - value) / abs(ref) if inst["sense"] == "max" else (value - ref) / abs(ref)


def write_maxcut_file(inst: dict, path) -> None:
    """The instance as an MQLib Max-Cut file: maximizing the cut with weights c minimizes H."""
    with open(path, "w") as fh:
        fh.write(f"{inst['n']} {len(inst['i'])}\n")
        np.savetxt(fh, np.stack([inst["i"] + 1, inst["j"] + 1, inst["c"]], 1), fmt="%d")
