"""The two dense families of the post, generated (never downloaded), plus the one evaluator every answer goes through.

- `reg`: a random d-regular graph on n nodes with +-1 couplings, as Max-Cut (H(s) = sum c_ij s_i s_j over the edges,
  minimizing H maximizes the cut with weights c). It is the union of d random perfect matchings, so every node has
  degree d before repeated pairs are merged (summed; a pair that sums to 0 is dropped). At n = 100,000 a repeat is
  rare (about d / n per edge), and the exact term count is recorded per instance.
- `full`: a fully dense binary QUBO on n variables: every pair i < j has an integer weight drawn uniformly from
  [-100, 100] without 0, and every variable a linear weight from [-100, 100]. min x^T Q x + q^T x over x in {0,1}^n.

Both are deterministic functions of (n, d or none, seed). Instances are dev-only generator output: fresh seeds,
sha256 recorded, never a public benchmark file.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
import examplekit as bk

POST_REG = {"n": 100_000, "degrees": [4, 16, 64, 256]}
POST_FULL = {"sizes": [3_000, 5_000, 7_000]}
WEIGHT = 100                      # |Q_ij|, |q_i| <= 100 in the full family


# ---- the regular family ---------------------------------------------------------------------------

def reg_instance(n: int, d: int, seed: int) -> dict:
    """Random d-regular +-1 Max-Cut on n nodes (n even), edges as i < j, sorted, repeats merged."""
    if n % 2:
        raise ValueError("n must be even (perfect matchings)")
    if not 1 <= d < n:
        raise ValueError("need 1 <= d < n")
    rng = np.random.default_rng([seed, d, n])
    keys, w = [], []
    for _ in range(d):
        p = rng.permutation(n)
        a, b = p[0::2], p[1::2]
        keys.append(np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b))
        w.append(rng.choice(np.array([-1, 1], dtype=np.int64), size=n // 2))
    key, inv = np.unique(np.concatenate(keys), return_inverse=True)
    c = np.bincount(inv, weights=np.concatenate(w)).astype(np.int64)
    keep = c != 0
    key, c = key[keep], c[keep]
    i, j = key // n, key % n
    return {"name": f"reg-d{d}-n{n}-s{seed}", "family": "reg", "kind": "maxcut", "vartype": "SPIN", "n": n,
            "degree": d, "seed": seed, "i": i, "j": j, "c": c, "linear": None, "sense": "max",
            "total_weight": int(c.sum()), "reference": None, "reference_kind": None}


# ---- the fully dense family -----------------------------------------------------------------------

def _full_block(n: int, seed: int, r0: int, r1: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows r0..r1-1 of the upper triangle: a pure function of (n, seed, r0) so blocks are independent."""
    rng = np.random.default_rng([seed, n, r0, 7])
    rows = np.repeat(np.arange(r0, r1, dtype=np.int64), n - 1 - np.arange(r0, r1))
    cols = np.concatenate([np.arange(r + 1, n, dtype=np.int64) for r in range(r0, r1)]) if r1 > r0 else rows
    v = rng.integers(1, 2 * WEIGHT + 1, size=rows.size, dtype=np.int64)
    v = np.where(v > WEIGHT, WEIGHT - v, v)                         # 1..100 and -1..-100, never 0
    return rows, cols, v


def full_linear(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng([seed, n, 11]).integers(-WEIGHT, WEIGHT + 1, size=n, dtype=np.int64)


def full_chunks(n: int, seed: int, block_rows: int = 256) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The upper triangle in (row, col) order, block_rows rows at a time. The block size changes the instance,
    so it is fixed here and recorded (FULL_BLOCK)."""
    for r0 in range(0, n, block_rows):
        yield _full_block(n, seed, r0, min(n, r0 + block_rows))


FULL_BLOCK = 256


def full_instance(n: int, seed: int) -> dict:
    """The whole fully dense instance in memory (24 bytes per term: 590 MB at n = 7,000)."""
    parts = list(full_chunks(n, seed, FULL_BLOCK))
    i, j, c = (np.concatenate([p[k] for p in parts]) for k in range(3))
    return {"name": f"full-n{n}-s{seed}", "family": "full", "kind": "qubo", "vartype": "BINARY", "n": n,
            "degree": n - 1, "seed": seed, "i": i, "j": j, "c": c, "linear": full_linear(n, seed), "sense": "min",
            "reference": None, "reference_kind": None}


def build(family: str, n: int, seed: int, d: int | None = None) -> dict:
    if family == "reg":
        return reg_instance(n, d, seed)
    if family == "full":
        return full_instance(n, seed)
    raise ValueError(family)


# ---- sizes ----------------------------------------------------------------------------------------

def expected_terms(family: str, n: int, d: int | None = None) -> int:
    """Couplings before repeats are merged (reg) or exactly (full). The symmetric matrix holds twice as many
    nonzeros: degree 256 at n = 100,000 is 12.8M couplings, 25.6M matrix entries."""
    return n * d // 2 if family == "reg" else n * (n - 1) // 2


def size_report(family: str, n: int, d: int | None = None) -> dict:
    nnz = expected_terms(family, n, d)
    raw = 40 + 8 * n + 16 * nnz                    # 4CQB v1: header, linear int64, row u32, col u32, val int64
    return {"family": family, "n": n, "degree": d if family == "reg" else n - 1, "couplings": nnz,
            "matrix_nonzeros": 2 * nnz, "file_bytes_raw": raw, "client_memory_gb_est": round(24 * nnz / 1e9, 2)}


# ---- files and fingerprints -----------------------------------------------------------------------

def sha256(inst: dict) -> str:
    """Fingerprint of the instance terms (recorded per row; the post's instances are pinned by it)."""
    h = hashlib.sha256()
    for a in (inst["i"], inst["j"], inst["c"]):
        h.update(np.ascontiguousarray(a, dtype="<i8").tobytes())
    if inst["linear"] is not None:
        h.update(np.ascontiguousarray(inst["linear"], dtype="<i8").tobytes())
    h.update(f"{inst['n']}:{inst['vartype']}".encode())
    return h.hexdigest()


def write_file(inst: dict, path, *, chunk: int = 1 << 20) -> dict:
    """The instance as one zstd-compressed 4CQB file (streamed in slices; byte-identical to the client's writer)."""
    i, j, c = inst["i"], inst["j"], inst["c"]

    def chunks():
        for k in range(0, len(i), chunk):
            yield i[k:k + chunk], j[k:k + chunk], c[k:k + chunk]

    return bk.write_4cqb_stream(path, n=inst["n"], nnz=len(i), chunks=chunks, linear=inst["linear"],
                                vartype=inst["vartype"])


# ---- the evaluator --------------------------------------------------------------------------------

def energy(inst: dict, x) -> int:
    """H(s) for `reg` (s in {-1,+1}), x^T Q x + q^T x for `full` (x in {0,1}). Exact int64."""
    x = np.asarray(x, dtype=np.int64)
    if len(x) != inst["n"]:
        raise ValueError(f"answer has {len(x)} values, the instance {inst['n']}")
    allowed = (-1, 1) if inst["vartype"] == "SPIN" else (0, 1)
    if not np.isin(x, allowed).all():
        raise ValueError(f"answer values must be in {allowed}")
    return bk.ising_energy(inst["i"], inst["j"], inst["c"], x, linear=inst["linear"])


def cut(inst: dict, s) -> int:
    """The Max-Cut value of a +-1 answer on the `reg` family: (W - H) / 2."""
    return (inst["total_weight"] - energy(inst, s)) // 2


def gap(e: float | None, best: float | None) -> float | None:
    """Relative gap of an energy to the best energy found (minimization), positive when worse."""
    if e is None or best is None or best == 0:
        return None
    return (e - best) / abs(best)


def best_found(rows) -> Dict[str, int]:
    """Per instance, the lowest energy any arm, budget or seed of the study found: the reference the gaps use.
    It is "best found in this study", not a proven optimum, and the post says so."""
    best: Dict[str, int] = {}
    for r in rows:
        e = r.get("energy")
        if r.get("event") == "solve" and e is not None:
            k = r["instance"]
            best[k] = min(best.get(k, e), e)
    return best
