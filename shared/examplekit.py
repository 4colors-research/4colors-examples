"""Small helpers shared by the examples in this repository.

Every example talks to the 4colors API through the public `fourcolors` client, as you would.

Key: the key is read from the file named by ``FOURCOLORS_KEY_FILE`` (so it never appears in a command line or a
log), else from ``FOURCOLORS_API_KEY``, as in the client itself.

Offline mode: set ``FOURCOLORS_OFFLINE=1`` and every example runs with no key and no network.
``fourcolors.qubo.solve`` and ``fourcolors.qubo.submit`` are then answered by `OfflineQubo`, a small local stand-in
(exact enumeration for tiny models, a short simulated annealing above that). Its answers and times are made up for
testing: they are not 4colors results and say nothing about the service.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional

__all__ = ["client", "offline_enabled", "Results", "base_parser", "result_fields", "reason_code", "client_version",
           "OfflineQubo", "plot_style", "load_rows", "median", "SERIES", "INK", "INK_2", "SURFACE", "GRID"]


def offline_enabled() -> bool:
    return os.environ.get("FOURCOLORS_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def client():
    """A `fourcolors.Client`: key from FOURCOLORS_KEY_FILE, else FOURCOLORS_API_KEY.

    With FOURCOLORS_OFFLINE=1 it installs the offline stand-in and returns None (the client functions then use the
    stand-in, so no key is needed)."""
    if offline_enabled():
        OfflineQubo().install()
        return None
    import fourcolors

    path = os.environ.get("FOURCOLORS_KEY_FILE")
    key = Path(path).expanduser().read_text().strip() if path else None
    return fourcolors.Client(api_key=key)


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


class _Patcher:
    """Plain setattr, for use outside pytest (pytest's monkeypatch has the same method)."""

    @staticmethod
    def setattr(obj, name, value):
        setattr(obj, name, value)


class OfflineQubo:
    """A stand-in for `fourcolors.qubo.solve` and `fourcolors.qubo.submit`, for tests and offline runs.

    It takes the same model types (through `fourcolors.qubo.to_model`), honors constraint rows, `sense`, `seed`
    and `warm_start`, and returns real `fourcolors.qubo.Result` objects with a trajectory, `constraints`, a fake job
    id and `billed_s`. Up to `exact_max_n` variables it enumerates every assignment (so it returns the true
    optimum); above that it runs a short seeded annealing with the rows as penalties. It is a test double, not a
    solver to measure: its times are made up.

        fake = OfflineQubo(); fake.install(monkeypatch)    # in a pytest test
        OfflineQubo().install()                            # in a script (what FOURCOLORS_OFFLINE=1 does)
    """

    def __init__(self, exact_max_n: int = 18, sweeps: int = 300):
        self.exact_max_n = exact_max_n
        self.sweeps = sweeps
        self.calls: List[dict] = []

    def install(self, monkeypatch=None) -> "OfflineQubo":
        import fourcolors.qubo

        mp = monkeypatch or _Patcher()
        mp.setattr(fourcolors.qubo, "solve", self.solve)
        mp.setattr(fourcolors.qubo, "submit", self.submit)
        return self

    def solve(self, model=None, time_limit_s: float = 10, *, on_incumbent=None, **kw):
        job = self.submit(model, time_limit_s, **kw)
        if on_incumbent is not None:
            for best in job.incumbents():
                on_incumbent(best)
        return job.result()

    def submit(self, model=None, time_limit_s: float = 10, *, linear=None, vartype=None, sense=None, offset=0,
               seed=None, warm_start=None, client=None, **kw):
        import fourcolors.qubo as fq

        if isinstance(model, fq.Model):
            m = model
        else:
            m = fq.to_model(model, linear, vartype=vartype, sense=sense or "min", offset=offset)
        self.calls.append({"n": m.n, "m": m.m, "time_limit_s": time_limit_s, "seed": seed,
                           "warm_start": None if warm_start is None else list(warm_start)})
        steps = self._search(m, time_limit_s, seed, warm_start)
        return _OfflineJob(m, steps, time_limit_s, f"offline-{len(self.calls)}")

    def _rows(self, m):
        import numpy as np

        A = np.zeros((m.m, m.n))
        lo = np.full(m.m, -np.inf)
        hi = np.full(m.m, np.inf)
        for r, c in enumerate(m.constraints):
            A[r, c.var.astype(np.int64)] = c.mant.astype(float) * 10.0 ** c.exp.astype(float)
            if c.lo is not None:
                lo[r] = c.lo[0] * 10.0 ** c.lo[1]
            if c.hi is not None:
                hi[r] = c.hi[0] * 10.0 ** c.hi[1]
        return A, lo, hi

    def _search(self, m, time_limit_s, seed, warm_start):
        """[(t, x), ...]: each strict improvement of the best feasible assignment, in model (min) terms."""
        import numpy as np

        sign = -1.0 if m.sense == "max" else 1.0
        A, lo, hi = self._rows(m)
        n = m.n
        lin = m.linear.astype(float)
        if n <= self.exact_max_n:
            idx = np.arange(1 << n, dtype=np.int64)
            X = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)
            E = float(m.offset) + X @ lin
            E = E + (m.val.astype(float) * X[:, m.row.astype(np.int64)] * X[:, m.col.astype(np.int64)]).sum(axis=1)
            ok = np.ones(len(X), dtype=bool)
            if m.m:
                R = X @ A.T
                ok = ((R >= lo - 1e-9) & (R <= hi + 1e-9)).all(axis=1)
            cand = np.flatnonzero(ok)
            for k in cand[np.argsort(sign * E[cand], kind="stable")]:
                if m.feasible(X[k]):            # the exact check decides
                    return [(0.25 * time_limit_s, X[k].tolist())]
            return []
        if m.vartype == "SPIN":
            raise NotImplementedError("OfflineQubo handles BINARY models")
        rnd = np.random.default_rng(seed)
        W = np.zeros((n, n))
        np.add.at(W, (m.row.astype(np.int64), m.col.astype(np.int64)), m.val.astype(float))
        diag = np.diag(W).copy()
        W = W + W.T
        np.fill_diagonal(W, 0.0)
        lin = (lin + diag) * sign
        W = W * sign
        scale = 1.0 + np.abs(lin).sum() / max(n, 1) + np.abs(W).sum() / max(n, 1)
        P = 4.0 * scale
        x = (np.asarray(warm_start, dtype=float) if warm_start is not None
             else rnd.integers(0, 2, n).astype(float))
        field_ = W @ x
        r = A @ x
        energy = lambda: float(lin @ x + 0.5 * x @ W @ x)  # noqa: E731
        viol = lambda rr: float((np.maximum(0, lo - rr) ** 2).sum() + (np.maximum(0, rr - hi) ** 2).sum())  # noqa: E731
        best, steps = np.inf, []
        total = self.sweeps * n
        temps = np.geomspace(2.0 * scale, 0.01, total)
        cur = energy() + P * viol(r)
        for it in range(total):
            i = int(rnd.integers(n))
            d = 1.0 - 2.0 * x[i]
            r_new = r + d * A[:, i] if m.m else r
            dv = P * (viol(r_new) - viol(r)) if m.m else 0.0
            dE = d * (lin[i] + field_[i]) + dv
            if dE <= 0 or rnd.random() < np.exp(-dE / temps[it]):
                x[i] += d
                field_ += d * W[:, i]
                r = r_new
                cur += dE
                if (not m.m or viol(r) < 1e-12) and cur < best - 1e-9:
                    xi = x.astype(np.int8)
                    if m.feasible(xi):
                        best = cur
                        steps.append((time_limit_s * (it + 1) / total, xi.tolist()))
        return steps


class _OfflineJob:
    """The `fourcolors.Job` shapes the examples use: id, best(), incumbents(), stop(), result(), billed_s."""

    def __init__(self, m, steps, time_limit_s, job_id):
        self.m, self.steps, self.time_limit_s, self.id = m, steps, time_limit_s, job_id
        self._seen = 0
        self._final = None
        self.status = "running"

    def _result(self, upto: int, status: str):
        from types import SimpleNamespace
        import fourcolors.qubo as fq

        steps = self.steps[:upto]
        traj = [{"seq": k, "t_search_s": t, "objective": self.m.energy(x), "objective_exact": True}
                for k, (t, x) in enumerate(steps)]
        x = steps[-1][1] if steps else None
        running = status == "running"
        billed = None if running else (steps[-1][0] if status == "stopped" and steps else float(self.time_limit_s))
        reason = None
        if status == "stopped":
            reason = {"code": "stopped"}
        elif x is None and not running:
            reason = {"code": "no_feasible_found"}
        cons = None
        if self.m.m and not running:
            cons = {"m": self.m.m, "feasible": x is not None and self.m.feasible(x), "rows_checked": self.m.m,
                    "rows_violated": 0 if x is not None and self.m.feasible(x) else None}
        return fq.Result(status=status, objective=None if x is None else self.m.energy(x), solution=x,
                         labels=list(self.m.labels or range(self.m.n)), vartype=self.m.vartype, reason=reason,
                         engine={"version": "offline"},
                         job=SimpleNamespace(id=self.id, billed_s=billed), trajectory=[] if running else traj,
                         constraints=cons)

    @property
    def billed_s(self):
        return None if self._final is None else self._final.billed_s

    def best(self):
        if self._final is not None:
            return self._final
        if self._seen < len(self.steps):
            self._seen += 1
        return self._result(self._seen, "running") if self._seen else None

    def incumbents(self, **kw):
        for k in range(self._seen + 1, len(self.steps) + 1):
            self._seen = k
            yield self._result(k, "running")
        yield self.result()

    def result(self, *a, **kw):
        if self._final is None:
            self._final = self._result(len(self.steps), "finished")
            self.status = "done"
        return self._final

    wait = result

    def stop(self, **kw):
        if self._final is None:
            self._final = self._result(self._seen, "stopped")
            self.status = "done"
        return self._final


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
