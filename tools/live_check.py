#!/usr/bin/env python3
"""Run a product's examples live against the 4colors service, one after another, and record what happened.

    python tools/live_check.py --products qubo --out /tmp/live-qubo            # QUBO quickstart + tutorials
    python tools/live_check.py --products qubo,compiler,platform --out DIR
    python tools/live_check.py --products compiler --dry-run                   # list what would run

This is the release step "run every example of that product live" (RELEASING.md). It needs a key, read the same
way the examples read it: FOURCOLORS_KEY_FILE (a file holding the key) or FOURCOLORS_API_KEY. The key is never
printed or written. Every job bills its run time: --dry-run lists each example's command and its "billed_s_max"
(the most execution time it can bill, from example.json) before anything runs.

Each example's "run" field in example.json is the command, run in the example's folder. Output per example goes to
DIR/<slug>.out and DIR/<slug>.err, and one JSON line per example to DIR/results.jsonl (slug, path, products,
command, exit code, wall seconds, start time, client version). Exit 0 only if every example exited 0.
Blog-post folders are skipped: their scripts take arguments and run longer; check them when a post publishes.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import products as P  # noqa: E402

KINDS = ("quickstart", "tutorial")


def selected(slugs: List[str], repo: Path = P.REPO) -> List[dict]:
    out = [m for m in P.examples(repo)
           if m.get("kind") in KINDS and m.get("run") and set(P.products_of(m)) & set(slugs)]
    order = {"quickstart": 0, "tutorial": 1}
    return sorted(out, key=lambda m: (order[m["kind"]], m["path"]))


def key_env() -> Optional[dict]:
    """The environment for the examples, with the key in FOURCOLORS_API_KEY, or None when there is no key."""
    env = dict(os.environ)
    path = env.get("FOURCOLORS_KEY_FILE")
    if path:
        env["FOURCOLORS_API_KEY"] = Path(path).expanduser().read_text().strip()
        env.pop("FOURCOLORS_KEY_FILE")
    return env if env.get("FOURCOLORS_API_KEY") else None


def client_version(python: str) -> Optional[str]:
    r = subprocess.run([python, "-c", "import fourcolors; print(fourcolors.__version__)"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--products", required=True, help="comma-separated product slugs, e.g. qubo or qubo,compiler")
    p.add_argument("--out", help="directory for the outputs and results.jsonl (required unless --dry-run)")
    p.add_argument("--dry-run", action="store_true", help="list the examples and their billed-time ceilings")
    p.add_argument("--timeout", type=float, default=1200, help="seconds per example (default %(default)s)")
    p.add_argument("--repo", default=str(P.REPO))
    a = p.parse_args(argv)
    repo = Path(a.repo)
    slugs = [s.strip() for s in a.products.split(",") if s.strip()]
    unknown = sorted(set(slugs) - set(P.by_slug(repo)))
    if unknown:
        print(f"live_check: unknown product(s) {unknown}; see products.json", file=sys.stderr)
        return 2
    todo = selected(slugs, repo)
    total = sum(m.get("billed_s_max") or 0 for m in todo)
    for m in todo:
        print(f"{m['path']}: {m['run']}  (bills at most {m.get('billed_s_max', '?')} s)")
    print(f"live_check: {len(todo)} example(s); at most {total:g} s of billed execution time in all")
    if a.dry_run:
        return 0
    if not a.out:
        print("live_check: --out is required", file=sys.stderr)
        return 2
    env = key_env()
    if env is None:
        print("live_check: no key (FOURCOLORS_KEY_FILE or FOURCOLORS_API_KEY)", file=sys.stderr)
        return 2
    env["PYTHONUNBUFFERED"] = "1"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    version = client_version(sys.executable)
    failed = []
    with (out / "results.jsonl").open("a") as rows:
        for m in todo:
            cmd = [sys.executable if c == "python" else c for c in shlex.split(m["run"])]
            started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            t0 = time.monotonic()
            with (out / f"{m['slug']}.out").open("w") as so, (out / f"{m['slug']}.err").open("w") as se:
                try:
                    rc = subprocess.run(cmd, cwd=repo / m["path"], env=env, stdout=so, stderr=se,
                                        timeout=a.timeout).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            wall = round(time.monotonic() - t0, 2)
            rows.write(json.dumps({"slug": m["slug"], "path": m["path"], "products": P.products_of(m),
                                   "command": m["run"], "exit": rc, "wall_s": wall, "started_utc": started,
                                   "client_version": version}) + "\n")
            rows.flush()
            print(f"{'ok  ' if rc == 0 else 'FAIL'} {m['path']} ({wall} s, exit {rc})", flush=True)
            if rc != 0:
                failed.append(m["path"])
    print(f"live_check: {len(todo) - len(failed)} of {len(todo)} passed" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
