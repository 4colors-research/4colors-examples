"""Small helpers shared by the examples in this repository.

Every example talks to the 4colors API through the public `fourcolors` client, as you would.

Key: the key is read from the file named by ``FOURCOLORS_KEY_FILE`` (so it never appears in a command line or a
log), else from ``FOURCOLORS_API_KEY``, as in the client itself.

Every example needs a key: it sends its model to the 4colors service and reports what the service answered.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional

__all__ = ["client", "load_key", "Results", "base_parser", "result_fields", "reason_code", "client_version",
           "plot_style", "load_rows", "median", "SERIES", "INK", "INK_2", "SURFACE", "GRID"]

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
