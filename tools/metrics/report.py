#!/usr/bin/env python3
"""Summarize the collected metrics as a Markdown report (weekly by default).

    python tools/metrics/report.py                       # reads $FOURCOLORS_METRICS_DIR
    python tools/metrics/report.py --dir /path --weeks 8

Rules, so the numbers can be rebuilt from the raw rows at any time:
- Daily views and clones: for each (repo, kind, day) the row from the LATEST collection wins (GitHub revises the
  current day while it is still running). Uniques do not add up across days (one person on two days counts
  twice), so weekly uniques are reported as "sum of daily uniques", labelled as such.
- Stars, forks, watchers: the last snapshot of each week. New stars per week from github-stars.jsonl.
- Release downloads: the latest download_count per asset (a running total).
- PyPI: the daily series "without_mirrors" (pypistats' own category), summed per week.
- Binder: launches per week from binder.jsonl.
- Referrers and paths are GitHub's 14-day top 10 at the latest collection; they are not summed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def read(d: Path, name: str) -> List[dict]:
    p = d / name
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def week_of(day: str) -> str:
    d = dt.date.fromisoformat(day[:10])
    monday = d - dt.timedelta(days=d.weekday())
    return monday.isoformat()


def daily_traffic(rows: List[dict]) -> Dict[tuple, dict]:
    """(repo, kind, day) -> the latest collected row for that day."""
    best: Dict[tuple, dict] = {}
    for r in rows:
        if r.get("kind") not in ("views", "clones"):
            continue
        k = (r["repo"], r["kind"], r["day"])
        if k not in best or r["collected_utc"] >= best[k]["collected_utc"]:
            best[k] = r
    return best


def weekly(d: Path) -> Dict[str, Dict[str, dict]]:
    """{repo or package: {week: {metric: value}}}"""
    out: Dict[str, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for (repo, kind, day), r in daily_traffic(read(d, "github-traffic.jsonl")).items():
        w = out[repo][week_of(day)]
        w[kind] = w.get(kind, 0) + r["count"]
        w[f"{kind}_daily_uniques_sum"] = w.get(f"{kind}_daily_uniques_sum", 0) + r["uniques"]
    for r in sorted(read(d, "github-repo.jsonl"), key=lambda r: r["collected_utc"]):
        w = out[r["repo"]][week_of(r["collected_utc"])]
        w.update(stars=r["stars"], forks=r["forks"], watchers=r["watchers"])
    for r in read(d, "github-stars.jsonl"):
        if r.get("starred_at"):
            w = out[r["repo"]][week_of(r["starred_at"])]
            w["new_stars"] = w.get("new_stars", 0) + 1
    pypi_days: Dict[tuple, dict] = {}
    for r in read(d, "pypi.jsonl"):
        if r.get("kind") == "without_mirrors" and r.get("day"):
            k = (r["package"], r["day"])
            if k not in pypi_days or r["collected_utc"] >= pypi_days[k]["collected_utc"]:
                pypi_days[k] = r
    for (pkg, day), r in pypi_days.items():
        w = out[f"pypi:{pkg}"][week_of(day)]
        w["downloads"] = w.get("downloads", 0) + (r.get("downloads") or 0)
    binder_rows: Dict[tuple, dict] = {}
    for r in read(d, "binder.jsonl"):
        binder_rows[(r["day"], r["spec"], r.get("provider"), r.get("status"))] = r
    for (day, spec, _p, status), r in binder_rows.items():
        if status == "success":
            w = out[f"binder:{spec}"][week_of(day)]
            w["launches"] = w.get("launches", 0) + r["launches"]
    return out


def latest_assets(d: Path) -> List[dict]:
    last: Dict[tuple, dict] = {}
    for r in read(d, "github-releases.jsonl"):
        k = (r["repo"], r["tag"], r["asset"])
        if k not in last or r["collected_utc"] >= last[k]["collected_utc"]:
            last[k] = r
    return sorted(last.values(), key=lambda r: (r["repo"], r["tag"] or "", r["asset"] or ""))


def latest_top(d: Path, kind: str) -> List[dict]:
    rows = [r for r in read(d, "github-traffic.jsonl") if r.get("kind") == kind]
    if not rows:
        return []
    stamp = max(r["collected_utc"] for r in rows)
    return sorted((r for r in rows if r["collected_utc"] == stamp), key=lambda r: (r["repo"], r["rank"]))


COLUMNS = ["views", "views_daily_uniques_sum", "clones", "clones_daily_uniques_sum", "new_stars", "stars", "forks",
           "watchers", "downloads", "launches"]


def render(d: Path, weeks: int = 8) -> str:
    data = weekly(d)
    lines = [f"# 4colors examples: traction report ({dt.date.today().isoformat()})", "",
             "Weekly, Monday to Sunday (UTC). Uniques are sums of daily uniques, not distinct people.", ""]
    for name in sorted(data):
        ws = sorted(data[name])[-weeks:]
        cols = [c for c in COLUMNS if any(c in data[name][w] for w in ws)]
        lines += [f"## {name}", "", "| week | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for w in ws:
            lines.append(f"| {w} | " + " | ".join(str(data[name][w].get(c, "")) for c in cols) + " |")
        lines.append("")
    for kind, title in (("referrers", "Top referrers (GitHub, last 14 days)"),
                        ("paths", "Top pages (GitHub, last 14 days)")):
        top = latest_top(d, kind)
        if top:
            lines += [f"## {title}", "", "| repo | rank | referrer or path | views | uniques |", "|---|---|---|---|---|"]
            lines += [f"| {r['repo']} | {r['rank']} | {r['key']} | {r['count']} | {r['uniques']} |" for r in top]
            lines.append("")
    assets = latest_assets(d)
    if assets:
        lines += ["## Release asset downloads (running totals)", "", "| repo | tag | asset | downloads |",
                  "|---|---|---|---|"]
        lines += [f"| {r['repo']} | {r['tag']} | {r['asset']} | {r['downloads']} |" for r in assets]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=os.environ.get("FOURCOLORS_METRICS_DIR",
                                                   str(Path.home() / ".local/state/fourcolors-metrics")))
    p.add_argument("--weeks", type=int, default=8)
    a = p.parse_args(argv)
    print(render(Path(a.dir).expanduser(), a.weeks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
