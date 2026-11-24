#!/usr/bin/env python3
"""Build the gallery tables in README.md and domains/README.md from products.json and every example.json.

    python tools/gallery.py            # rewrite the tables
    python tools/gallery.py --check    # exit 1 if a table is out of date

README.md gets one section per product, in the order products.json lists them; an example with several products
appears under each of them. domains/README.md gets one table per domain. The tables are written between
<!-- gallery:<name>:start --> and <!-- gallery:<name>:end --> markers.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import products as P  # noqa: E402
from links import utm  # noqa: E402

REPO = P.REPO
KIND_ORDER = {"quickstart": 0, "tutorial": 1, "post": 2}


def load(repo: Path = REPO) -> List[dict]:
    out = P.examples(repo)
    out.sort(key=lambda m: (KIND_ORDER.get(m.get("kind"), 9), m.get("month", ""), m["path"]))
    return out


def row(m: dict, names: Dict[str, str], link_prefix: str = "") -> str:
    title = f"[{m['title']}]({link_prefix}{m['path']}/)"
    links = []
    if m.get("docs_url"):
        links.append(f"[docs]({utm(m['docs_url'], m['slug'], 'gallery')})")
    if m.get("post_url"):
        links.append(f"[post]({utm(m['post_url'], m['slug'], 'gallery')})")
    if m.get("notebook"):
        links.append(f"[notebook]({link_prefix}{m['path']}/{m['notebook']})")
    prods = ", ".join(names.get(p, p) for p in P.products_of(m))
    return (f"| {title} | {m.get('kind', '')} | {prods} | {m.get('level', '')} | "
            f"{', '.join(m.get('domains', []))} | {' · '.join(links)} |")


HEADER = "| Example | Kind | Products | Level | Domains | Links |\n|---|---|---|---|---|---|"


def table(items: List[dict], names: Dict[str, str], link_prefix: str = "") -> str:
    if not items:
        return "_Nothing here yet._"
    return HEADER + "\n" + "\n".join(row(m, names, link_prefix) for m in items)


def product_sections(items: List[dict], prods: List[dict]) -> str:
    names = {p["slug"]: p["short"] for p in prods}
    parts = []
    for p in prods:
        mine = [m for m in items if p["slug"] in P.products_of(m)]
        status = " (preview)" if p.get("status") == "preview" else ""
        docs = utm(p["docs_url"], p["slug"], "gallery-section")
        parts.append(f"### {p['name']}{status}\n\n[Docs]({docs}) · Needs: `{p['requires']}`\n\n"
                     + table(mine, names))
    return "\n\n".join(parts)


def domain_tables(items: List[dict], prods: List[dict]) -> str:
    names = {p["slug"]: p["short"] for p in prods}
    by: Dict[str, List[dict]] = {}
    for m in items:
        for d in m.get("domains") or ["general"]:
            by.setdefault(d, []).append(m)
    parts = [f"### {d}\n\n" + table(by[d], names, link_prefix="../") for d in sorted(by)]
    return "\n\n".join(parts) if parts else "_Nothing here yet._"


def replace_block(text: str, name: str, body: str) -> str:
    rx = re.compile(rf"(<!-- gallery:{name}:start -->\n)(?:.*?\n)?(<!-- gallery:{name}:end -->)", re.S)
    if not rx.search(text):
        raise SystemExit(f"gallery: no <!-- gallery:{name}:start/end --> markers")
    return rx.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text, count=1)


def build(repo: Path = REPO) -> Dict[Path, str]:
    items, prods = load(repo), P.load(repo)
    readme = repo / "README.md"
    domains = repo / "domains" / "README.md"
    return {
        readme: replace_block(readme.read_text(encoding="utf-8"), "products", product_sections(items, prods)),
        domains: replace_block(domains.read_text(encoding="utf-8"), "domains", domain_tables(items, prods)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--repo", default=str(REPO))
    a = p.parse_args(argv)
    stale = []
    for path, text in build(Path(a.repo)).items():
        if path.read_text(encoding="utf-8") != text:
            stale.append(path)
            if not a.check:
                path.write_text(text, encoding="utf-8")
    if a.check and stale:
        print("gallery out of date: " + ", ".join(str(s) for s in stale) + " (run python tools/gallery.py)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
