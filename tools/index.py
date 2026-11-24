#!/usr/bin/env python3
"""Write examples-index.json: where every example lives on GitHub, and which docs page or post shows it.

    python tools/index.py            # rewrite examples-index.json
    python tools/index.py --check    # exit 1 if it is out of date

The docs site and the 4colors website read this file at build time. A page whose URL is an example's docs_url or
post_url gets a "Code for this page" box: View on GitHub (the example's folder), Clone (the clone and cd
commands), and Open in Colab when the example has a notebook. Pages are matched by URL path, so no page carries a
hand-written link that can go stale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import products as P  # noqa: E402

REPO_SLUG = "4colors-research/4colors-examples"
GITHUB = f"https://github.com/{REPO_SLUG}"
BRANCH = "main"


def entry(m: dict) -> dict:
    path = m["path"]
    e = {
        "slug": m["slug"], "kind": m["kind"], "title": m["title"], "products": P.products_of(m), "path": path,
        "github_url": f"{GITHUB}/tree/{BRANCH}/{path}",
        "clone": f"git clone {GITHUB}\ncd 4colors-examples/{path}",
        "docs_url": m.get("docs_url"), "post_url": m.get("post_url"),
        "docs_path": urlsplit(m["docs_url"]).path if m.get("docs_url") else None,
        "post_path": urlsplit(m["post_url"]).path if m.get("post_url") else None,
    }
    if m.get("notebook"):
        e["colab_url"] = f"https://colab.research.google.com/github/{REPO_SLUG}/blob/{BRANCH}/{path}/{m['notebook']}"
    return e


def build(repo: Path = P.REPO) -> str:
    items = sorted((entry(m) for m in P.examples(repo)), key=lambda e: e["path"])
    # One example per page: when two share a docs page, the tutorial (the fuller example) is that page's code.
    rank = {"tutorial": 0, "quickstart": 1, "post": 2}
    for key in ("docs_path", "post_path"):
        best = {}
        for e in items:
            if e[key] and (e[key] not in best or rank[e["kind"]] < rank[best[e[key]]["kind"]]):
                best[e[key]] = e
        for e in items:
            if e[key] and best[e[key]] is not e:
                e[key] = None
    doc = {"repository": GITHUB, "branch": BRANCH, "examples": items}
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--repo", default=str(P.REPO))
    a = p.parse_args(argv)
    repo = Path(a.repo)
    out = repo / "examples-index.json"
    text = build(repo)
    if a.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            print("examples-index.json is out of date (run python tools/index.py)")
            return 1
        return 0
    out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
