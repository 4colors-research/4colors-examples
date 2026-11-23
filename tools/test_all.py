#!/usr/bin/env python3
"""Run every example's tests, each example in its own pytest process.

    python tools/test_all.py            # tools/tests, then every example directory with a tests/ folder
    python tools/test_all.py posts/     # only the examples under posts/

Examples are self-contained folders whose modules have plain names (plot.py, for instance), so one pytest
process per example keeps them from importing each other's modules. No test needs a key or the network.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def example_dirs(roots):
    for root in roots:
        for tests in sorted(Path(root).glob("**/tests")):
            rel = tests.relative_to(REPO) if tests.is_absolute() else tests
            if any(part.startswith(".") for part in rel.parts) or not tests.is_dir():
                continue
            yield tests.parent


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    roots = [Path(a).resolve() for a in args] or [REPO]
    env = dict(os.environ, MPLBACKEND="Agg")
    env.pop("FOURCOLORS_API_KEY", None)
    env.pop("FOURCOLORS_KEY_FILE", None)
    failed = []
    for d in example_dirs(roots):
        print(f"== {d.relative_to(REPO)}", flush=True)
        rc = subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=d, env=env)
        if rc != 0:
            failed.append(str(d.relative_to(REPO)))
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("all example tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
