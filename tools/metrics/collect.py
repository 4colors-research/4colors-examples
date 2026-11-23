#!/usr/bin/env python3
"""Collect traction metrics for the public 4colors repositories and the `fourcolors` package, once a day.

    FOURCOLORS_METRICS_TOKEN_FILE=/path/to/token python tools/metrics/collect.py --config config.json
    python tools/metrics/collect.py --config config.json --only pypi binder      # sources needing no token

GitHub keeps traffic (views, clones, referrers, popular paths) for 14 days only, so this must run at least
every few days; daily is the plan. Every run appends what it read, as read, to JSONL files in the output
directory (append-only: nothing is rewritten, so a later report can always be rebuilt from the raw rows):

    github-traffic.jsonl    views and clones per day (count and uniques), referrers and popular paths (14-day top 10)
    github-repo.jsonl       stars, forks, watchers, open issues, size (a snapshot per run)
    github-stars.jsonl      one row per star: starred_at and a salted hash of the account (never the name)
    github-releases.jsonl   download_count of every release asset
    pypi.jsonl              pypistats: recent (day/week/month) and the daily series with and without mirrors
    binder.jsonl            mybinder.org launches of the configured repositories (public launch archive)
    collect.log             one line per source per run (ok or the error), when run by the systemd unit

Token: a GitHub token read from the file named by FOURCOLORS_METRICS_TOKEN_FILE (never from the command line or
the config). Traffic needs push access: a fine-grained token with read-only "Administration" and "Metadata" on the
repositories, or a GitHub App installation token with the same permissions. The token is never logged.

Config (JSON): {"repos": ["4colors-research/4colors-examples"], "pypi": ["fourcolors"],
                "binder_specs": ["4colors-research/4colors-examples"], "star_salt_file": "/path/to/salt"}
Standard library only, so the systemd unit needs nothing installed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

GITHUB = "https://api.github.com"
PYPISTATS = "https://pypistats.org/api/packages"
BINDER_ARCHIVE = "https://archive.analytics.mybinder.org"
USER_AGENT = "fourcolors-examples-metrics/1.0"
SOURCES = ("traffic", "repo", "stars", "releases", "pypi", "binder")

Fetch = Callable[[str, Dict[str, str]], bytes]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http_fetch(url: str, headers: Dict[str, str], *, retries: int = 3, timeout: float = 60.0) -> bytes:
    """GET with a generic user agent and a few retries on 5xx/429. Raises on 4xx."""
    req_headers = {"User-Agent": USER_AGENT, **headers}
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=req_headers), timeout=timeout) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as exc:
            last = exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {retries} tries: {last}")


class Sink:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    def append(self, name: str, rows: Iterable[dict]) -> int:
        n = 0
        with (self.out_dir / name).open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                n += 1
        return n

    def read(self, name: str) -> List[dict]:
        p = self.out_dir / name
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


class GitHub:
    def __init__(self, token: Optional[str], fetch: Fetch):
        self.token, self.fetch = token, fetch

    def get(self, path: str, accept: str = "application/vnd.github+json"):
        h = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return json.loads(self.fetch(GITHUB + path, h))

    def pages(self, path: str, accept: str = "application/vnd.github+json", per_page: int = 100):
        page = 1
        while True:
            sep = "&" if "?" in path else "?"
            items = self.get(f"{path}{sep}per_page={per_page}&page={page}", accept)
            if not items:
                return
            yield from items
            if len(items) < per_page:
                return
            page += 1


def collect_traffic(gh: GitHub, repo: str, stamp: str) -> List[dict]:
    rows = []
    for kind in ("views", "clones"):
        data = gh.get(f"/repos/{repo}/traffic/{kind}?per=day")
        for d in data.get(kind, []):
            rows.append({"collected_utc": stamp, "repo": repo, "kind": kind, "day": d["timestamp"][:10],
                         "count": d["count"], "uniques": d["uniques"]})
        rows.append({"collected_utc": stamp, "repo": repo, "kind": f"{kind}_14d", "count": data.get("count"),
                     "uniques": data.get("uniques")})
    for kind, path in (("referrers", "popular/referrers"), ("paths", "popular/paths")):
        for k, d in enumerate(gh.get(f"/repos/{repo}/traffic/{path}")):
            rows.append({"collected_utc": stamp, "repo": repo, "kind": kind, "rank": k + 1,
                         "key": d.get("referrer") or d.get("path"), "title": d.get("title"),
                         "count": d["count"], "uniques": d["uniques"]})
    return rows


def collect_repo(gh: GitHub, repo: str, stamp: str) -> List[dict]:
    d = gh.get(f"/repos/{repo}")
    return [{"collected_utc": stamp, "repo": repo, "stars": d.get("stargazers_count"), "forks": d.get("forks_count"),
             "watchers": d.get("subscribers_count"), "open_issues": d.get("open_issues_count"),
             "size_kb": d.get("size"), "pushed_at": d.get("pushed_at")}]


def account_hash(login: str, salt: bytes) -> str:
    return hashlib.sha256(salt + login.encode("utf-8")).hexdigest()[:16]


def collect_stars(gh: GitHub, repo: str, stamp: str, salt: bytes, seen: set) -> List[dict]:
    """New stars since the last run. Only the time and a salted hash of the account are kept (for de-duplication);
    an unstar is not visible here and shows only as a lower total in github-repo.jsonl."""
    rows = []
    for s in gh.pages(f"/repos/{repo}/stargazers", accept="application/vnd.github.star+json"):
        h = account_hash((s.get("user") or {}).get("login", ""), salt)
        if (repo, h) in seen:
            continue
        seen.add((repo, h))
        rows.append({"collected_utc": stamp, "repo": repo, "starred_at": s.get("starred_at"), "account": h})
    return rows


def collect_releases(gh: GitHub, repo: str, stamp: str) -> List[dict]:
    rows = []
    for rel in gh.pages(f"/repos/{repo}/releases"):
        for a in rel.get("assets", []):
            rows.append({"collected_utc": stamp, "repo": repo, "tag": rel.get("tag_name"), "asset": a.get("name"),
                         "downloads": a.get("download_count")})
    return rows


def collect_pypi(fetch: Fetch, package: str, stamp: str) -> List[dict]:
    rows = []
    recent = json.loads(fetch(f"{PYPISTATS}/{package}/recent", {"Accept": "application/json"}))
    rows.append({"collected_utc": stamp, "package": package, "kind": "recent", **recent.get("data", {})})
    overall = json.loads(fetch(f"{PYPISTATS}/{package}/overall", {"Accept": "application/json"}))
    for d in overall.get("data", []):
        rows.append({"collected_utc": stamp, "package": package, "kind": d.get("category"), "day": d.get("date"),
                     "downloads": d.get("downloads")})
    return rows


def collect_binder(fetch: Fetch, specs: List[str], day: str, stamp: str) -> List[dict]:
    """Launch counts for one day from the public mybinder.org archive (no IPs there: spec, ref, time, status)."""
    raw = fetch(f"{BINDER_ARCHIVE}/events-{day}.jsonl", {})
    counts: Dict[tuple, int] = {}
    for line in io.BytesIO(raw):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        spec = e.get("spec", "")
        for want in specs:
            if spec.lower().startswith(want.lower() + "/"):
                key = (want, e.get("provider"), e.get("status"))
                counts[key] = counts.get(key, 0) + 1
    return [{"collected_utc": stamp, "day": day, "spec": k[0], "provider": k[1], "status": k[2], "launches": v}
            for k, v in sorted(counts.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]


def read_token(env: Dict[str, str]) -> Optional[str]:
    path = env.get("FOURCOLORS_METRICS_TOKEN_FILE")
    if not path:
        return None
    return Path(path).expanduser().read_text(encoding="utf-8").strip() or None


def read_salt(cfg: dict, out_dir: Path) -> bytes:
    """A per-installation random salt so star hashes cannot be matched across datasets; created once."""
    p = Path(cfg.get("star_salt_file") or out_dir / "star-salt")
    p = p.expanduser()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(os.urandom(32))
        os.chmod(p, 0o600)
    return p.read_bytes()


def run(cfg: dict, out_dir: Path, *, only: Iterable[str] = SOURCES, fetch: Fetch = http_fetch,
        env: Optional[Dict[str, str]] = None, today: Optional[dt.date] = None) -> Dict[str, str]:
    env = dict(os.environ if env is None else env)
    only = set(only)
    sink = Sink(out_dir)
    stamp = now_utc()
    status: Dict[str, str] = {}
    token = read_token(env) if only & {"traffic", "repo", "stars", "releases"} else None
    gh = GitHub(token, fetch)
    repos = cfg.get("repos", [])

    def step(name: str, fn):
        try:
            n = fn()
            status[name] = f"ok {n} rows"
        except Exception as exc:  # one source failing must not stop the others
            status[name] = f"error {type(exc).__name__}: {str(exc)[:200]}"

    if "traffic" in only:
        if not token:
            status["traffic"] = "skipped: no token (FOURCOLORS_METRICS_TOKEN_FILE)"
        else:
            step("traffic", lambda: sum(sink.append("github-traffic.jsonl", collect_traffic(gh, r, stamp))
                                        for r in repos))
    if "repo" in only:
        step("repo", lambda: sum(sink.append("github-repo.jsonl", collect_repo(gh, r, stamp)) for r in repos))
    if "stars" in only:
        salt = read_salt(cfg, out_dir)
        seen = {(r["repo"], r["account"]) for r in sink.read("github-stars.jsonl")}
        step("stars", lambda: sum(sink.append("github-stars.jsonl", collect_stars(gh, r, stamp, salt, seen))
                                  for r in repos))
    if "releases" in only:
        step("releases", lambda: sum(sink.append("github-releases.jsonl", collect_releases(gh, r, stamp))
                                     for r in repos))
    if "pypi" in only:
        step("pypi", lambda: sum(sink.append("pypi.jsonl", collect_pypi(fetch, p, stamp))
                                 for p in cfg.get("pypi", [])))
    if "binder" in only and cfg.get("binder_specs"):
        # yesterday and up to 6 days before it that a missed run left out (the archive keeps old days)
        base = today or dt.datetime.now(dt.timezone.utc).date()
        done = {r["day"] for r in sink.read("binder-days.jsonl")}
        days = [d for d in ((base - dt.timedelta(days=k)).isoformat() for k in range(7, 0, -1)) if d not in done]

        def binder():
            n = 0
            for day in days:
                rows = collect_binder(fetch, cfg["binder_specs"], day, stamp)
                n += sink.append("binder.jsonl", rows)
                sink.append("binder-days.jsonl", [{"collected_utc": stamp, "day": day, "rows": len(rows)}])
            return n
        if days:
            step("binder", binder)
        else:
            status["binder"] = "skipped: every day of the last week already collected"
    return status


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=os.environ.get("FOURCOLORS_METRICS_DIR",
                                                   str(Path.home() / ".local/state/fourcolors-metrics")))
    p.add_argument("--only", nargs="+", choices=SOURCES, default=list(SOURCES))
    a = p.parse_args(argv)
    cfg = json.loads(Path(a.config).expanduser().read_text(encoding="utf-8"))
    status = run(cfg, Path(a.out).expanduser(), only=a.only)
    for k, v in status.items():
        print(f"{now_utc()} {k}: {v}")
    return 1 if any(v.startswith("error") for v in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
