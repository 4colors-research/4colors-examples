"""collect.py and report.py against a fake HTTP layer."""
import datetime as dt
import json
from urllib.parse import urlsplit

import collect
import report

DAYS = ["2026-11-16", "2026-11-17"]


class FakeWeb:
    def __init__(self, views=(5, 7), stars=("alice", "bob")):
        self.calls, self.views, self.stars = [], views, stars

    def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        u = urlsplit(url)
        p = u.path
        if p.endswith("/traffic/views") or p.endswith("/traffic/clones"):
            kind = p.rsplit("/", 1)[1]
            items = [{"timestamp": f"{d}T00:00:00Z", "count": c, "uniques": 1 + i} for i, (d, c) in
                     enumerate(zip(DAYS, self.views))]
            return json.dumps({"count": sum(self.views), "uniques": 2, kind: items}).encode()
        if p.endswith("/popular/referrers"):
            return json.dumps([{"referrer": "news.example.org", "count": 4, "uniques": 3}]).encode()
        if p.endswith("/popular/paths"):
            return json.dumps([{"path": "/o/r/tree/main/posts", "title": "posts", "count": 9, "uniques": 4}]).encode()
        if p.endswith("/stargazers"):
            assert headers["Accept"] == "application/vnd.github.star+json"
            page = int(dict(x.split("=") for x in u.query.split("&"))["page"])
            if page > 1:
                return b"[]"
            return json.dumps([{"starred_at": "2026-11-17T10:00:00Z", "user": {"login": s}}
                               for s in self.stars]).encode()
        if p.endswith("/releases"):
            page = int(dict(x.split("=") for x in u.query.split("&"))["page"])
            return b"[]" if page > 1 else json.dumps(
                [{"tag_name": "v1", "assets": [{"name": "examples.zip", "download_count": 12}]}]).encode()
        if p.startswith("/repos/"):
            return json.dumps({"stargazers_count": len(self.stars), "forks_count": 1, "subscribers_count": 2,
                               "open_issues_count": 0, "size": 100, "pushed_at": "2026-11-17T00:00:00Z"}).encode()
        if p.endswith("/recent"):
            return json.dumps({"data": {"last_day": 4, "last_week": 8, "last_month": 147}}).encode()
        if p.endswith("/overall"):
            return json.dumps({"data": [{"category": "without_mirrors", "date": d, "downloads": 3} for d in DAYS]
                               + [{"category": "with_mirrors", "date": DAYS[0], "downloads": 9}]}).encode()
        if "events-" in p:
            ev = [{"spec": "o/r/main", "provider": "GitHub", "status": "success"},
                  {"spec": "o/r/HEAD", "provider": "GitHub", "status": "success"},
                  {"spec": "o/rother/HEAD", "provider": "GitHub", "status": "success"},
                  {"spec": "x/y/HEAD", "provider": "GitHub", "status": "success"}]
            return ("\n".join(json.dumps(e) for e in ev) + "\nnot json\n").encode()
        raise AssertionError(f"unexpected url {url}")


CFG = {"repos": ["o/r"], "pypi": ["fourcolors"], "binder_specs": ["o/r"]}


def test_collect_all_sources(tmp_path):
    tok = tmp_path / "token"
    tok.write_text("secret-token-value\n")
    web = FakeWeb()
    out = tmp_path / "out"
    st = collect.run(CFG, out, fetch=web, env={"FOURCOLORS_METRICS_TOKEN_FILE": str(tok)},
                     today=dt.date(2026, 11, 18))
    assert all(v.startswith("ok") for v in st.values()), st
    gh = [h for u, h in web.calls if "api.github.com" in u]
    assert gh and all(h["Authorization"] == "Bearer secret-token-value" for h in gh)
    assert all("Authorization" not in h for u, h in web.calls if "api.github.com" not in u)
    for f in out.iterdir():                                   # the token never lands in an output
        assert "secret-token-value" not in f.read_text(errors="ignore")
    stars = collect.Sink(out).read("github-stars.jsonl")
    assert len(stars) == 2 and all("alice" not in json.dumps(s) and len(s["account"]) == 16 for s in stars)
    binder = collect.Sink(out).read("binder.jsonl")
    assert {r["launches"] for r in binder} == {2} and len({r["day"] for r in binder}) == 7


def test_second_run_adds_only_new_stars_and_skips_done_binder_days(tmp_path):
    tok = tmp_path / "token"
    tok.write_text("t")
    env = {"FOURCOLORS_METRICS_TOKEN_FILE": str(tok)}
    out = tmp_path / "out"
    collect.run(CFG, out, fetch=FakeWeb(stars=("alice",)), env=env, today=dt.date(2026, 11, 18))
    st = collect.run(CFG, out, fetch=FakeWeb(stars=("alice", "bob")), env=env, today=dt.date(2026, 11, 18))
    assert len(collect.Sink(out).read("github-stars.jsonl")) == 2
    assert st["binder"].startswith("skipped")


def test_no_token_skips_traffic_but_runs_public_sources(tmp_path):
    st = collect.run(CFG, tmp_path, fetch=FakeWeb(), env={}, only=["traffic", "pypi"])
    assert st["traffic"].startswith("skipped") and st["pypi"].startswith("ok")


def test_one_failing_source_does_not_stop_the_others(tmp_path):
    def broken(url, headers):
        if "pypistats" in url:
            raise RuntimeError("down")
        return FakeWeb()(url, headers)
    st = collect.run(CFG, tmp_path, fetch=broken, env={}, only=["pypi", "repo"])
    assert st["pypi"].startswith("error") and st["repo"].startswith("ok")


def test_report_keeps_the_latest_value_per_day(tmp_path):
    tok = tmp_path / "token"
    tok.write_text("t")
    env = {"FOURCOLORS_METRICS_TOKEN_FILE": str(tok)}
    collect.run(CFG, tmp_path, fetch=FakeWeb(views=(5, 7)), env=env, only=["traffic", "repo", "pypi"])
    collect.run(CFG, tmp_path, fetch=FakeWeb(views=(5, 9)), env=env, only=["traffic", "repo", "pypi"])
    w = report.weekly(tmp_path)
    week = report.week_of(DAYS[0])
    assert w["o/r"][week]["views"] == 14                   # 5 + 9: the later read of the 17th wins
    assert w["pypi:fourcolors"][week]["downloads"] == 6    # without mirrors only
    text = report.render(tmp_path)
    assert "news.example.org" in text and "| week |" in text
