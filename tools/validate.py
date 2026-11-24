#!/usr/bin/env python3
"""Check the repository's layout rules: products.json, every example.json, and where each example lives.

    python tools/validate.py          # exit 1 and list the problems, if any

Rules:
- products.json: each product has a unique lowercase slug, a name, a short name, a status (live or preview),
  a docs URL and what it requires.
- example.json: kind, slug, title, summary, products, domains, level and requires are present; every product is
  in products.json; the slug is lowercase letters, digits and dashes, and unique across the repository.
- Where an example lives, by kind:
    quickstart  quickstart/<product>/                 slug quickstart-<product>, one product
    tutorial    tutorials/<product>/<name>/           <product> is one of its products (the first one);
                                                      <name> is the slug without a leading "<product>-"
                tutorials/joint/<name>/               for a tutorial that needs two or more products equally
    post        posts/<yyyy-mm>-<slug>/               "month" is <yyyy-mm>
- A tutorial or quickstart names its docs page (docs_url on the docs site); a tutorial's page is
  start/tutorials/<slug>/. A post with a post_url links to the 4colors site.
- Every example folder has a README.md and a tests/ folder with at least one test file.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import products as P  # noqa: E402

SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DOCS = "https://docs.4colors-research.com/"
SITE = "https://4colors-research.com/"
KINDS = ("quickstart", "tutorial", "post")
LEVELS = ("beginner", "intermediate", "advanced")
REQUIRED = ("kind", "slug", "title", "summary", "products", "domains", "level", "requires")


def check_products(repo: Path) -> List[str]:
    errs, seen = [], set()
    try:
        rows = P.load(repo)
    except (OSError, ValueError, KeyError) as exc:
        return [f"products.json: cannot read it ({exc})"]
    for k, p in enumerate(rows):
        where = f"products.json row {k}"
        for f in ("slug", "name", "short", "status", "docs_url", "requires"):
            if not p.get(f):
                errs.append(f"{where}: missing {f}")
        s = p.get("slug", "")
        if s and not SLUG.match(s):
            errs.append(f"{where}: slug {s!r} is not lowercase letters, digits and dashes")
        if s == P.JOINT:
            errs.append(f"{where}: {P.JOINT!r} is reserved for tutorials/joint/")
        if s in seen:
            errs.append(f"{where}: slug {s!r} listed twice")
        seen.add(s)
        if p.get("status") and p["status"] not in P.STATUSES:
            errs.append(f"{where}: status must be one of {P.STATUSES}")
        if p.get("docs_url") and not p["docs_url"].startswith(DOCS):
            errs.append(f"{where}: docs_url must be on {DOCS}")
    return errs


def check_example(m: dict, known: set, repo: Path) -> List[str]:
    path = m["path"]
    where = f"{path}/example.json"
    errs = [f"{where}: missing {f}" for f in REQUIRED if f not in m]
    if errs:
        return errs
    kind, slug, prods = m["kind"], m["slug"], P.products_of(m)
    if kind not in KINDS:
        errs.append(f"{where}: kind must be one of {KINDS}")
    if not SLUG.match(slug):
        errs.append(f"{where}: slug {slug!r} is not lowercase letters, digits and dashes")
    if not prods:
        errs.append(f"{where}: products must be a non-empty list")
    for p in prods:
        if p not in known:
            errs.append(f"{where}: product {p!r} is not in products.json")
    if m["level"] not in LEVELS:
        errs.append(f"{where}: level must be one of {LEVELS}")
    if not isinstance(m["domains"], list):
        errs.append(f"{where}: domains must be a list")
    parts = path.split("/")
    if kind == "quickstart":
        if len(parts) != 2 or parts[0] != "quickstart":
            errs.append(f"{where}: a quickstart lives in quickstart/<product>/")
        elif prods != [parts[1]] or slug != f"quickstart-{parts[1]}":
            errs.append(f"{where}: quickstart/{parts[1]}/ has products [{parts[1]!r}] and slug quickstart-{parts[1]}")
    elif kind == "tutorial":
        if len(parts) != 3 or parts[0] != "tutorials":
            errs.append(f"{where}: a tutorial lives in tutorials/<product>/<name>/")
        else:
            group, name = parts[1], parts[2]
            if group == P.JOINT:
                if len(prods) < 2:
                    errs.append(f"{where}: tutorials/joint/ is for tutorials with two or more products")
                if name != slug:
                    errs.append(f"{where}: the folder name must be the slug ({slug})")
            else:
                if not prods or prods[0] != group:
                    errs.append(f"{where}: tutorials/{group}/ holds tutorials whose first product is {group!r}")
                expected = slug[len(group) + 1:] if slug.startswith(group + "-") else slug
                if name != expected:
                    errs.append(f"{where}: the folder name must be {expected!r} (the slug without '{group}-')")
        if m.get("docs_url") and m["docs_url"] != f"{DOCS}start/tutorials/{slug}/":
            errs.append(f"{where}: docs_url must be {DOCS}start/tutorials/{slug}/")
    elif kind == "post":
        month = m.get("month", "")
        if not re.fullmatch(r"20\d\d-[01]\d", month):
            errs.append(f"{where}: a post needs month yyyy-mm")
        if len(parts) != 2 or parts[0] != "posts" or parts[1] != f"{month}-{slug}":
            errs.append(f"{where}: a post lives in posts/<month>-<slug>/ (posts/{month}-{slug}/)")
        if m.get("post_url") and not m["post_url"].startswith(SITE):
            errs.append(f"{where}: post_url must be on {SITE}")
    if kind in ("quickstart", "tutorial") and not str(m.get("docs_url") or "").startswith(DOCS):
        errs.append(f"{where}: a {kind} names its docs page (docs_url on {DOCS})")
    folder = repo / path
    if not (folder / "README.md").is_file():
        errs.append(f"{path}/: no README.md")
    if not any((folder / "tests").glob("test_*.py")):
        errs.append(f"{path}/: no tests/test_*.py")
    if m.get("notebook") and not (folder / m["notebook"]).is_file():
        errs.append(f"{where}: notebook {m['notebook']!r} is not in the folder")
    return errs


def validate(repo: Path = P.REPO) -> List[str]:
    errs = check_products(repo)
    known = {p["slug"] for p in P.load(repo)} if not errs else set()
    seen = {}
    for m in P.examples(repo):
        errs += check_example(m, known, repo)
        s = m.get("slug")
        if s in seen:
            errs.append(f"{m['path']}/example.json: slug {s!r} is also used by {seen[s]}/")
        seen.setdefault(s, m["path"])
    return errs


def main(argv=None) -> int:
    repo = Path(argv[0]) if argv else P.REPO
    errs = validate(repo)
    for e in errs:
        print(e)
    print(f"validate: {len(errs)} problem(s)", file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
