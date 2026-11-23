#!/usr/bin/env python3
"""Build the gallery tables in README.md and domains/README.md from every example.json.

    python tools/gallery.py            # rewrite the tables
    python tools/gallery.py --check    # exit 1 if a table is out of date (for CI)

Each example directory (quickstart, tutorials, posts) holds an example.json:
    {"kind": "post" | "tutorial" | "quickstart", "slug", "title", "summary", "product": "qubo" | "compiler" | "both",
     "domains": [...], "level": "beginner" | "intermediate" | "advanced", "post_url": url or null, ...}
The tables are written between <!-- gallery:<name>:start --> and <!-- gallery:<name>:end --> markers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from links import utm  # noqa: E402

PRODUCT_LABEL = {"qubo": "Optimizer for QUBO", "compiler": "Quantum Compiler", "both": "both"}
KIND_ORDER = {"quickstart": 0, "tutorial": 1, "post": 2}


def load(repo: Path = REPO) -> List[dict]:
    out = []
    for p in sorted(repo.glob("**/example.json")):
        if any(part.startswith(".") for part in p.relative_to(repo).parts):
            continue
        meta = json.loads(p.read_text(encoding="utf-8"))
        meta["path"] = p.parent.relative_to(repo).as_posix()
        out.append(meta)
    out.sort(key=lambda m: (KIND_ORDER.get(m.get("kind"), 9), m.get("month", ""), m["path"]))
    return out


def row(m: dict, link_prefix: str = "") -> str:
    title = f"[{m['title']}]({link_prefix}{m['path']}/)"
    post = f"[post]({utm(m['post_url'], m['slug'], 'gallery')})" if m.get("post_url") else ""
    domains = ", ".join(m.get("domains", []))
    return (f"| {title} | {m.get('kind', '')} | {PRODUCT_LABEL.get(m.get('product'), m.get('product', ''))} | "
            f"{m.get('level', '')} | {domains} | {post} |")


HEADER = "| Example | Kind | Product | Level | Domains | Post |\n|---|---|---|---|---|---|"


def table(items: List[dict], link_prefix: str = "") -> str:
    if not items:
        return "_Nothing here yet._"
    return HEADER + "\n" + "\n".join(row(m, link_prefix) for m in items)


def domain_tables(items: List[dict]) -> str:
    by: Dict[str, List[dict]] = {}
    for m in items:
        for d in m.get("domains") or ["general"]:
            by.setdefault(d, []).append(m)
    parts = []
    for d in sorted(by):
        parts.append(f"### {d}\n\n" + table(by[d], link_prefix="../"))
    return "\n\n".join(parts) if parts else "_Nothing here yet._"


def replace_block(text: str, name: str, body: str) -> str:
    rx = re.compile(rf"(<!-- gallery:{name}:start -->\n)(?:.*?\n)?(<!-- gallery:{name}:end -->)", re.S)
    if not rx.search(text):
        raise SystemExit(f"gallery: no <!-- gallery:{name}:start/end --> markers")
    return rx.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text, count=1)


def build(repo: Path = REPO) -> Dict[Path, str]:
    items = load(repo)
    readme = repo / "README.md"
    domains = repo / "domains" / "README.md"
    out = {}
    out[readme] = replace_block(readme.read_text(encoding="utf-8"), "all", table(items))
    out[domains] = replace_block(domains.read_text(encoding="utf-8"), "domains", domain_tables(items))
    return out


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
