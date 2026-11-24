"""Small helpers shared by the examples in this repository.

Every example talks to the 4colors API through the public `fourcolors` client, as you would.

Key: the key is read from the file named by ``FOURCOLORS_KEY_FILE`` (so it never appears in a command line or a
log), else from ``FOURCOLORS_API_KEY``, as in the client itself.

Every example needs a key: it sends its model to the 4colors service and reports what the service answered.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = ["client", "load_key", "Results", "base_parser", "result_fields", "reason_code", "client_version",
           "plot_style", "load_rows", "median", "SERIES", "INK", "INK_2", "SURFACE", "GRID", "MUTED", "MARKERS",
           "style_axes", "random_regular", "gnp", "maxcut_qubo", "cut_value", "brute_force_maxcut",
           "time_to_target", "ising_energy", "ea3d_coupling", "ea3d_nnz", "ea3d_chunks", "ea3d_arrays",
           "ea3d_energy", "EA3D_E0_PAL1996", "EA3D_E0_LATER", "write_4cqb_stream", "peak_rss_gb",
           "service_memory_gb"]

TRIAL_URL = (
    "https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=examplekit&utm_content=script"
)
NO_KEY = (f"This example needs a 4colors API key. Start a free trial at {TRIAL_URL}, copy your key, then "
          "export FOURCOLORS_API_KEY=... (or FOURCOLORS_KEY_FILE=/path/to/file).")


def load_key() -> None:
    """For scripts written exactly as the docs show them, which use the client's default key.

    If FOURCOLORS_KEY_FILE names a file, its key is put in FOURCOLORS_API_KEY for this process only (the client
    reads that variable). With no key at all, exit with a short note on how to get one."""
    path = os.environ.get("FOURCOLORS_KEY_FILE")
    if path:
        os.environ["FOURCOLORS_API_KEY"] = Path(path).expanduser().read_text().strip()
    if not os.environ.get("FOURCOLORS_API_KEY"):
        raise SystemExit(NO_KEY)


def client():
    """A `fourcolors.Client`: key from FOURCOLORS_KEY_FILE, else FOURCOLORS_API_KEY."""
    import fourcolors

    path = os.environ.get("FOURCOLORS_KEY_FILE")
    key = Path(path).expanduser().read_text().strip() if path else None
    try:
        return fourcolors.Client(api_key=key)
    except fourcolors.errors.AuthenticationError as exc:
        raise SystemExit(f"{exc}\n{NO_KEY}")


class Results:
    """Append one JSON object per line; every row gets the UTC time it was written."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, **row) -> dict:
        row.setdefault("written_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return row

    def rows(self) -> List[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def base_parser(description: str, *, time_limit: float = 5.0, seeds=(1, 2, 3)):
    """The arguments every QUBO example script takes: --out DIR, --time-limit S, --seeds ..."""
    import argparse

    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="directory for results.jsonl")
    p.add_argument("--time-limit", type=float, default=time_limit, help="seconds per job (default %(default)s)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(seeds), help="solver seeds (default %(default)s)")
    return p


def reason_code(r) -> Optional[str]:
    reason = getattr(r, "reason", None)
    return reason.get("code") if isinstance(reason, dict) else reason


def client_version() -> Optional[str]:
    """The installed `fourcolors` client's version (None if it is not installed)."""
    try:
        import fourcolors
    except ImportError:
        return None
    return getattr(fourcolors, "__version__", None)


def result_fields(r) -> dict:
    """What every row records about a fourcolors result, exactly as the service reported it.

    `engine_version` is the release that answered (`r.engine["version"]`), so a row says which build it measured;
    `client_version` is the `fourcolors` client that sent it. `solver_objective` is the service's number; scripts
    store their own recomputed objective beside it. `trajectory` is [[t_search_s, objective], ...]."""
    job = getattr(r, "job", None)
    traj = list(getattr(r, "trajectory", None) or [])
    engine = getattr(r, "engine", None)
    engine = dict(engine) if isinstance(engine, dict) else {}
    return {"job_id": getattr(job, "id", None), "status": r.status, "reason": reason_code(r),
            "engine_version": engine.get("version"), "engine": engine, "client_version": client_version(),
            "solver_objective": r.objective, "billed_s": r.billed_s,
            "service_constraints": getattr(r, "constraints", None),
            "trajectory": [[p.get("t_search_s"), p.get("objective")] for p in traj]}


# ---- plots ----------------------------------------------------------------------------------------

# Categorical colors in fixed order (never cycled past 8); text stays in ink colors, never a series color.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def plot_style():
    """Matplotlib (Agg) with a clean chart look: light surface, recessive grid, 2px lines, 8px markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 8,
        "legend.frameon": False, "font.size": 10, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })
    return plt


def load_rows(paths) -> List[dict]:
    """Rows of one or more results.jsonl files (or directories holding one)."""
    rows: List[dict] = []
    for p in paths:
        p = Path(p)
        rows += Results(p / "results.jsonl" if p.is_dir() else p).rows()
    return rows


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else 0.5 * (xs[k - 1] + xs[k])


# ---- graphs and Max-Cut (from the blog posts' shared helper) ---------------------------------------

Edge = Tuple[int, int]


def random_regular(n: int, d: int, seed: int) -> List[Edge]:
    """A random d-regular simple graph on n nodes (pairing model, retried until simple).

    Raises ValueError when no such graph exists: n * d odd (the degrees cannot pair up), or d >= n."""
    if n * d % 2:
        raise ValueError(f"no {d}-regular graph on {n} nodes: n * d = {n * d} is odd (every edge uses two "
                         f"endpoints, so n * d must be even)")
    if d < 0 or (n > 0 and d >= n):
        raise ValueError(f"no simple {d}-regular graph on {n} nodes: need 0 <= d < n")
    rnd = random.Random(seed)
    while True:
        stubs = [v for v in range(n) for _ in range(d)]
        rnd.shuffle(stubs)
        edges = {tuple(sorted(stubs[k:k + 2])) for k in range(0, len(stubs), 2)}
        if len(edges) == len(stubs) // 2 and all(a != b for a, b in edges):
            return sorted(edges)


def gnp(n: int, p: float, seed: int) -> List[Edge]:
    rnd = random.Random(seed)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if rnd.random() < p]


def maxcut_qubo(edges: Iterable[Edge], weights: Optional[Dict[Edge, float]] = None):
    """Max-Cut as min E(x) = sum_w (2 x_i x_j - x_i - x_j); then cut(x) = -E(x)."""
    Q: Dict[Edge, float] = {}
    lin: Dict[int, float] = {}
    for e in edges:
        w = 1.0 if weights is None else weights[e]
        i, j = e
        Q[(i, j)] = Q.get((i, j), 0.0) + 2 * w
        lin[i] = lin.get(i, 0.0) - w
        lin[j] = lin.get(j, 0.0) - w
    n = 1 + max(max(e) for e in Q) if Q else 0
    return Q, [lin.get(i, 0.0) for i in range(n)]


def cut_value(edges: Iterable[Edge], x, weights: Optional[Dict[Edge, float]] = None) -> float:
    return sum((1.0 if weights is None else weights[e]) for e in edges if x[e[0]] != x[e[1]])


def brute_force_maxcut(n: int, edges: List[Edge], weights=None) -> Tuple[float, List[int]]:
    """Exact maximum cut by enumeration (node 0 fixed to 0); fine up to about n = 24. A reference for checking
    small answers and for labeled reference optima in posts, never a substitute for the service."""
    import numpy as np

    if n > 26:
        raise ValueError("brute force is for n <= 26")
    best, arg = -1.0, None
    idx = np.arange(1 << (n - 1), dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(n - 1)) & 1).astype(np.int8)
    bits = np.concatenate([np.zeros((len(idx), 1), dtype=np.int8), bits], axis=1)
    w = np.array([1.0 if weights is None else weights[e] for e in edges])
    a = np.array([e[0] for e in edges]); b = np.array([e[1] for e in edges])
    cuts = ((bits[:, a] != bits[:, b]) * w).sum(axis=1)
    k = int(cuts.argmax())
    best, arg = float(cuts[k]), bits[k].tolist()
    return best, arg



def time_to_target(trajectory, target: float, *, sense: str = "min") -> Optional[float]:
    """Search seconds of the first trajectory point at least as good as `target`; None if never reached.

    `trajectory` is a result's trajectory (dicts with t_search_s and objective) or [[t, objective], ...] pairs."""
    for p in trajectory or []:
        t, obj = (p["t_search_s"], p["objective"]) if isinstance(p, dict) else (p[0], p[1])
        if obj is None or t is None:
            continue
        if (obj <= target) if sense == "min" else (obj >= target):
            return float(t)
    return None



# ---- Ising energy, one evaluator for every answer -------------------------------------------------

def ising_energy(row, col, val, s, linear=None, offset=0) -> int:
    """H(s) = offset + sum_i linear[i] s_i + sum_k val[k] s[row[k]] s[col[k]] for s in {-1,+1}, exactly
    (int64 sums; fine while sum |val| < 2**63). Works for 0/1 x as well (then it is the QUBO energy)."""
    import numpy as np

    s = np.asarray(s, dtype=np.int64)
    e = int(offset) + int(np.dot(np.asarray(val, dtype=np.int64), s[np.asarray(row)] * s[np.asarray(col)]))
    if linear is not None:
        e += int(np.dot(np.asarray(linear, dtype=np.int64), s))
    return e


# ---- 3D +-J Edwards-Anderson spin glass, generated in slabs ---------------------------------------
#
# Sites of an L x L x L periodic lattice are numbered i = (z * L + y) * L + x. Site i owns three bonds,
# to its +x, +y and +z neighbors (with wrap-around). The coupling J of the bond owned by (i, d) is a
# pure function of (seed, 3 i + d): a splitmix64 hash, so any slab can be generated on its own and the
# instance does not depend on the slab size. The model is H(s) = -sum_<ij> J_ij s_i s_j, written as
# Ising terms val = -J (the 4CQB file then holds exactly H). Ground-state energy density of this
# family in the large-L limit: about -1.7863 per spin (K. F. Pal, Physica A 223 (1996) 283-292).

EA3D_E0_PAL1996 = -1.7863
EA3D_E0_LATER = -1.7876      # a later estimate; used only as a sensitivity check


def _splitmix64(k):
    import numpy as np

    with np.errstate(over="ignore"):
        z = k + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def ea3d_coupling(seed: int, owner, d: int):
    """J (+1 or -1, int64) of the bonds owned by sites `owner` in direction d (0 = x, 1 = y, 2 = z)."""
    import numpy as np

    with np.errstate(over="ignore"):
        key = np.asarray(owner, dtype=np.uint64) * np.uint64(3) + np.uint64(d)
        key = key ^ _splitmix64(np.uint64(seed) + np.uint64(0x5EED))
    h = _splitmix64(key)
    return np.where((h >> np.uint64(63)) == 1, 1, -1).astype(np.int64)


def ea3d_nnz(L: int) -> int:
    return 3 * L ** 3


def ea3d_chunks(L: int, seed: int, max_sites: int = 1 << 20):
    """Yield (row, col, val) int64 arrays of the Ising terms val = -J, in 4CQB order (row < col, sorted
    by (row, col)), one slab of whole z-planes (about max_sites sites) at a time. Memory per slab is
    roughly 200 bytes per site, so the default slab needs about 200 MB whatever L is."""
    import numpy as np

    if L < 3:
        raise ValueError("L must be at least 3 (at L = 2 the +x and -x neighbors coincide)")
    plane = L * L
    step = max(1, max_sites // plane)
    for z0 in range(0, L, step):
        z1 = min(L, z0 + step)
        i = np.arange(z0 * plane, z1 * plane, dtype=np.int64)
        coord = (i % L, (i // L) % L, i // plane)
        stride = (1, L, plane)
        rows, cols, vals = [], [], []
        for d in range(3):
            c, st = coord[d], stride[d]
            plus = i + (((c + 1) % L) - c) * st          # the +d neighbor (owned bond)
            minus = i + (((c - 1) % L) - c) * st         # the -d neighbor (owns the bond to i)
            keep = plus > i
            rows.append(i[keep]); cols.append(plus[keep]); vals.append(-ea3d_coupling(seed, i[keep], d))
            keep = minus > i                              # only where the lattice wraps
            rows.append(i[keep]); cols.append(minus[keep]); vals.append(-ea3d_coupling(seed, minus[keep], d))
        r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        order = np.lexsort((c, r))
        yield r[order], c[order], v[order]


def ea3d_arrays(L: int, seed: int):
    """The whole instance as (row, col, val) arrays: for small L only (24 bytes x 3 L^3)."""
    import numpy as np

    parts = list(ea3d_chunks(L, seed))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def ea3d_energy(L: int, seed: int, s) -> int:
    """H(s) of the instance, slab by slab (no full coupling arrays in memory)."""
    import numpy as np

    s = np.asarray(s, dtype=np.int8)
    return sum(ising_energy(r, c, v, s) for r, c, v in ea3d_chunks(L, seed))


# ---- a streaming 4CQB writer ----------------------------------------------------------------------
#
# fourcolors.qubo.write_4cqb builds the whole file in memory from a Model, which itself holds every term.
# For tens of millions of terms we write the documented 4CQB version 1 layout (the module docstring of
# fourcolors' optimizer format: 40-byte header, linear, row, col, val; little-endian) straight to disk,
# optionally as one zstd frame. `chunks` is called three times (once per array) and must yield the same
# terms each time, in (row, col) order. The output is byte-identical to write_4cqb for the same model
# (tests check this), so it has the same input id.

def write_4cqb_stream(path, *, n: int, nnz: int, chunks, linear=None, offset: int = 0, vartype: str = "BINARY",
                      sense: str = "min", compress: bool = True, level: int = 3, threads: int = 0) -> dict:
    """Write an int64-valued 4CQB file without holding the model; return sizes and the input id.

    linear: None (all zero), an int array of length n, or a callable yielding int chunks in index order.
    compress: one zstd frame (as the API takes it; threads > 0 still writes one frame)."""
    import struct
    import numpy as np

    FLAG_INT64, FLAG_MAX, FLAG_ISING = 1, 2, 4
    LIMIT = 2 ** 53
    flags = FLAG_INT64 | (FLAG_MAX if sense == "max" else 0) | (FLAG_ISING if vartype == "SPIN" else 0)
    raw_size = 40 + 8 * n + (4 + 4 + 8) * nnz
    try:
        import blake3
        hasher = blake3.blake3()
    except ImportError:
        hasher = None
    fh = open(path, "wb")
    sink = fh
    if compress:
        import zstandard
        sink = zstandard.ZstdCompressor(level=level, threads=threads).stream_writer(fh, size=raw_size,
                                                                                      closefd=False)
    written = [0]

    def put(b: bytes):
        sink.write(b)
        if hasher is not None:
            hasher.update(b)
        written[0] += len(b)

    put(struct.pack("<4sIQQII", b"4CQB", 1, n, nnz, flags, 0) + struct.pack("<q", int(offset)))
    if linear is None:
        zero = np.zeros(min(n, 1 << 20), dtype="<i8").tobytes()
        left = n
        while left:
            k = min(left, 1 << 20)
            put(zero[:8 * k]); left -= k
    else:
        parts = linear() if callable(linear) else [np.asarray(linear)]
        count = 0
        for p in parts:
            p = np.asarray(p, dtype=np.int64)
            if p.size and np.abs(p).max() > LIMIT:
                raise ValueError("a linear value is beyond 2**53")
            put(p.astype("<i8").tobytes()); count += p.size
        if count != n:
            raise ValueError(f"linear has {count} values, not n = {n}")
    for k, dt in ((0, "<u4"), (1, "<u4"), (2, "<i8")):
        count, last = 0, (-1, -1)
        for r, c, v in chunks():
            r = np.asarray(r, dtype=np.int64); c = np.asarray(c, dtype=np.int64); v = np.asarray(v, dtype=np.int64)
            if k == 0 and r.size:                      # check the order once, on the first pass
                if not (np.all(r < c) and c.max() < n and r.min() >= 0):
                    raise ValueError("terms need 0 <= row < col < n")
                key_prev = np.concatenate([[last[0]], r[:-1]]), np.concatenate([[last[1]], c[:-1]])
                if not np.all((r > key_prev[0]) | ((r == key_prev[0]) & (c > key_prev[1]))):
                    raise ValueError("terms must be sorted by (row, col) without duplicates")
                if np.any(v == 0) or np.abs(v).max() > LIMIT:
                    raise ValueError("values must be nonzero and within 2**53")
                last = (int(r[-1]), int(c[-1]))
            put((r, c, v)[k].astype(dt).tobytes()); count += r.size
        if count != nnz:
            raise ValueError(f"the chunks gave {count} terms, not nnz = {nnz}")
    if compress:
        sink.close()
    fh.close()
    import os
    return {"path": str(path), "n": n, "nnz": nnz, "bytes_raw": written[0], "bytes_on_disk": os.path.getsize(path),
            "input_id": hasher.hexdigest() if hasher is not None else None}


def peak_rss_gb() -> float:
    """This process's peak resident memory so far (Linux ru_maxrss is in KiB)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def service_memory_gb(n: int) -> float:
    """Rule of thumb for the service's memory for a sparse model: about 1 GB per million variables
    (the scale runs behind the 1M-10M figures). The job's own memory check is what decides."""
    return n / 1e6


# ---- plots ----------------------------------------------------------------------------------------



# ---- plot helpers used by the posts' charts --------------------------------------------------------

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3de"


def style_axes(ax):
    """Recessive grid and axes; the data carries the ink."""
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)

