"""How often does a job's trajectory get a new point? Read it from the record before trusting a stop rule.

    python cadence.py RUN/out/results.jsonl [more.jsonl ...]

For each `final` row it prints the job id, the number of trajectory points, the search time of the first point, and
the median and largest gap between consecutive points (the service's `t_search_s`). If the gaps cluster at one value
(a reporting step rather than real search progress), a patience shorter than about twice that step stops jobs that
are still improving: pick the patience from this output, not from a guess. Needs no key; reads files only.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import List, Optional


def gaps(trajectory) -> List[float]:
    ts = [p[0] if not isinstance(p, dict) else p.get("t_search_s") for p in trajectory or []]
    ts = [float(t) for t in ts if t is not None]
    return [b - a for a, b in zip(ts, ts[1:])]


def summary(row: dict) -> dict:
    traj = row.get("trajectory") or []
    g = gaps(traj)
    first: Optional[float] = None
    if traj:
        p = traj[0]
        first = p.get("t_search_s") if isinstance(p, dict) else p[0]
    return {"job_id": row.get("job_id"), "arm": row.get("arm"), "seed": row.get("seed"), "points": len(traj),
            "first_s": first, "median_gap_s": statistics.median(g) if g else None, "max_gap_s": max(g) if g else None}


def main(argv=None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print(__doc__)
        return 2
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("kind") == "final":
                    print(json.dumps(summary(r)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
