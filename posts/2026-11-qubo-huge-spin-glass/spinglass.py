"""3D +-J Edwards-Anderson spin glasses at very large sizes, for the 4colors Optimizer for QUBO.

The instance itself (lattice, couplings, slabs) lives in examplekit (ea3d_*), because the solver
comparison post uses the same family. This module adds what the large-model post needs:

- `size_report(L)`: variables, terms, file size and memory estimates, printed before anything is built.
- `write_instance(path, L, seed)`: the 4CQB file, written slab by slab (one zstd frame by default),
  without ever holding the model.
- `score(L, seed, s)`: energy per spin of an answer and its gap to the published ground-state density.

Sizes used in the post (the standard plan, 8 GiB per job): L = 100 (1,000,000 spins), 126 (2,000,376),
159 (4,019,679) and 182 (6,028,568). Larger lattices work the same way but need more than the standard plan holds.
"""
from __future__ import annotations

import time

import numpy as np

import examplekit as bk

SIZES = {"1M": 100, "2M": 126, "4M": 159, "6M": 182}
SLAB_SITES = 1 << 20             # sites per generated slab; the generator's memory is set by this, not by L
BYTES_PER_SLAB_SITE = 200        # measured order of magnitude for numpy temporaries per slab site (estimate)


def size_report(L: int) -> dict:
    """Everything worth knowing before generating or uploading an L^3 instance (estimates, not measurements)."""
    n, nnz = L ** 3, bk.ea3d_nnz(L)
    raw = 40 + 8 * n + 16 * nnz                    # header + linear + row/col (u32) + val (int64)
    return {
        "L": L, "n": n, "nnz": nnz,
        "file_raw_gb": raw / 1e9,                   # upper bound for the zstd file too
        "generator_mem_gb": (BYTES_PER_SLAB_SITE * min(n, max(SLAB_SITES, L * L)) + n) / 1e9,
        "service_mem_gb": bk.service_memory_gb(n),  # about 1 GB per million variables
        "reference_energy": bk.EA3D_E0_PAL1996 * n,
    }


def print_size_report(rep: dict) -> None:
    print(f"L = {rep['L']}: {rep['n']:,} spins, {rep['nnz']:,} couplings")
    print(f"  4CQB file: at most {rep['file_raw_gb']:.2f} GB raw (zstd makes it smaller)")
    print(f"  generator memory: about {rep['generator_mem_gb']:.2f} GB (slab by slab)")
    print(f"  service memory: about {rep['service_mem_gb']:.0f} GB (rule of thumb, 1 GB per million variables;"
          " the job's memory check decides, and a refusal is billed nothing)")


def write_instance(path, L: int, seed: int, *, compress: bool = True, threads: int = 0) -> dict:
    """Write the instance to `path` as 4CQB (Ising, int64, H = sum val s_i s_j with val = -J)."""
    t0 = time.perf_counter()
    info = bk.write_4cqb_stream(path, n=L ** 3, nnz=bk.ea3d_nnz(L),
                                chunks=lambda: bk.ea3d_chunks(L, seed, max_sites=SLAB_SITES),
                                vartype="SPIN", compress=compress, threads=threads)
    info.update(L=L, seed=seed, write_s=time.perf_counter() - t0)
    return info


def score(L: int, seed: int, s) -> dict:
    """Energy of an answer (re-evaluated here, slab by slab), per spin, and the gap to the reference."""
    s = np.asarray(s, dtype=np.int8)
    if s.shape != (L ** 3,) or not np.isin(s, (-1, 1)).all():
        raise ValueError("an answer is L^3 values in {-1, +1}")
    e = bk.ea3d_energy(L, seed, s)
    per = e / L ** 3
    return {"energy": e, "energy_per_spin": per,
            # positive gap = above the reference; a finite instance can land slightly below it
            "gap_pal1996": (per - bk.EA3D_E0_PAL1996) / abs(bk.EA3D_E0_PAL1996),
            "gap_later": (per - bk.EA3D_E0_LATER) / abs(bk.EA3D_E0_LATER)}
