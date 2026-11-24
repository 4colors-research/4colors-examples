"""The product registry (products.json at the repository root) and the example metadata (example.json files).

Every tool reads products from here, so adding a product is one row in products.json plus its folders.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
STATUSES = ("live", "preview")
JOINT = "joint"          # tutorials/joint/<name>/: tutorials that use more than one product


def load(repo: Path = REPO) -> List[dict]:
    """The products, in the order products.json lists them (the order of the gallery sections)."""
    return json.loads((repo / "products.json").read_text(encoding="utf-8"))["products"]


def by_slug(repo: Path = REPO) -> Dict[str, dict]:
    return {p["slug"]: p for p in load(repo)}


def products_of(meta: dict) -> List[str]:
    """The product slugs an example.json names."""
    ps = meta.get("products")
    return list(ps) if isinstance(ps, list) else []


def examples(repo: Path = REPO) -> List[dict]:
    """Every example.json under the repository (hidden folders skipped), with its folder as "path"."""
    out = []
    for p in sorted(repo.glob("**/example.json")):
        rel = p.relative_to(repo)
        if any(part.startswith(".") for part in rel.parts):
            continue
        meta = json.loads(p.read_text(encoding="utf-8"))
        meta["path"] = p.parent.relative_to(repo).as_posix()
        out.append(meta)
    return out
