"""Does the service list the warm start itself as the first trajectory point? Read it from the record.

    python first_points.py RUN/out/results.jsonl

For each warm `solve` row it compares the job's first trajectory point with that day's `carry_over` row (yesterday's
answer scored on today's weights, the warm start the job was given). It prints one line per job and a verdict:
`warm start listed at t=0` when every warm job's first point has the carry-over energy at (near) zero search time;
`first point is an improvement` when none does; `mixed` otherwise. The post states which, because it decides whether a
warm job's time to quality can be 0. Needs no key; reads files only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List


def check(rows: List[dict], *, t0: float = 0.05) -> dict:
    carry = {r["day"]: r["energy_python"] for r in rows if r.get("kind") == "carry_over"}
    jobs = []
    for r in rows:
        if r.get("kind") != "solve" or r.get("arm") != "warm":
            continue
        traj = r.get("trajectory") or []
        first = traj[0] if traj else None
        t, obj = (first[0], first[1]) if first else (None, None)
        listed = (first is not None and t is not None and t <= t0 and carry.get(r["day"]) is not None
                  and obj == carry[r["day"]])
        jobs.append({"day": r["day"], "seed": r.get("seed"), "job_id": r.get("job_id"), "first": first,
                     "carry_over_energy": carry.get(r["day"]), "warm_start_listed": listed})
    n = sum(j["warm_start_listed"] for j in jobs)
    verdict = ("no warm jobs" if not jobs else "warm start listed at t=0" if n == len(jobs)
               else "first point is an improvement" if n == 0 else "mixed")
    return {"verdict": verdict, "listed": n, "warm_jobs": len(jobs), "jobs": jobs}


def main(argv=None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print(__doc__)
        return 2
    rows = [json.loads(line) for p in paths for line in Path(p).read_text().splitlines() if line.strip()]
    rep = check(rows)
    for j in rep["jobs"]:
        print(json.dumps(j))
    print(f"verdict: {rep['verdict']} ({rep['listed']} of {rep['warm_jobs']} warm jobs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
