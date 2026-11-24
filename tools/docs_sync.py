#!/usr/bin/env python3
"""Keep the code in the docs and in notebooks identical to the example scripts here: one copy of each snippet.

    python tools/docs_sync.py list                        # every region in the repository
    python tools/docs_sync.py check                       # notebooks only
    python tools/docs_sync.py check --docs <docs checkout>    # notebooks and docs pages; exit 1 on any drift
    python tools/docs_sync.py write --docs <docs checkout>    # rewrite drifted docs blocks and notebook cells
    python tools/docs_sync.py mark  --docs <docs checkout>    # add markers to docs blocks that equal a region

Regions. A script marks a snippet with comment lines (the same lines editors use for folding):

    # region docs:build
    ...the code the docs show...
    # endregion

A region is named `<path from the repository root>#<name>`, e.g. `tutorials/qubo/maxcut/maxcut.py#build`.
The scripts run top to bottom, so a tutorial's regions are its docs blocks in reading order, and the docs blocks
run as one script, as the docs' own example runner expects.

Docs pages. A fenced code block is tied to a region by an HTML comment in the comment lines just above its fence
(other comments, such as an UNVERIFIED-EXAMPLE flag, may sit beside it):

    <!-- example: tutorials/qubo/maxcut/maxcut.py#build -->
    ```python
    ...
    ```

Notebooks. A code cell is tied to a region by its metadata: {"example": {"region": "<path>#<name>"}}.

This repository's own pages (README.md files that show a script's code) use the same markers as docs pages, and
`check` and `write` always cover them and the notebooks; `--docs` adds a docs checkout. `mark` without `--docs`
marks this repository's pages.

Direction: the scripts here are the source. `write` changes only docs blocks and notebook cells, never a script.
Blocks without a marker are left alone (illustrations that are not part of a runnable script).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
SCRIPT_SUFFIXES = {".py", ".sh"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "build", ".docusaurus"}
REGION_START = re.compile(r"^#\s*region docs:([A-Za-z0-9_-]+)\s*$")
REGION_END = re.compile(r"^#\s*endregion\b")
MARKER = re.compile(r"^<!--\s*example:\s*(\S+#[A-Za-z0-9_-]+)\s*-->\s*$")
COMMENT = re.compile(r"^<!--.*-->\s*$")
FENCE_OPEN = re.compile(r"^(`{3,})([A-Za-z0-9_+-]*)\s*$")


class SyncError(ValueError):
    pass


def _trim(lines: List[str]) -> str:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return "\n".join(lines) + "\n" if lines else ""


def regions_of(text: str, name: str) -> Dict[str, str]:
    """The `# region docs:<id>` ... `# endregion` blocks of one script, by id. Regions do not nest."""
    out: Dict[str, str] = {}
    cur: Optional[str] = None
    body: List[str] = []
    for k, line in enumerate(text.splitlines(), start=1):
        m = REGION_START.match(line)
        if m:
            if cur is not None:
                raise SyncError(f"{name}:{k}: region {m.group(1)!r} starts inside region {cur!r}")
            if m.group(1) in out:
                raise SyncError(f"{name}:{k}: region {m.group(1)!r} is defined twice")
            cur, body = m.group(1), []
        elif REGION_END.match(line):
            if cur is None:
                raise SyncError(f"{name}:{k}: endregion without a region")
            out[cur] = _trim(body)
            cur = None
        elif cur is not None:
            body.append(line)
    if cur is not None:
        raise SyncError(f"{name}: region {cur!r} is never closed")
    return out


def _walk(root: Path, suffixes) -> List[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if p.is_file() and p.suffix in suffixes:
            out.append(p)
    return out


def all_regions(repo: Path = REPO) -> Dict[str, str]:
    """Regions of the example scripts; the repository tools and every tests/ folder are not examples."""
    out: Dict[str, str] = {}
    for p in _walk(repo, SCRIPT_SUFFIXES):
        parts = p.relative_to(repo).parts
        if parts[0] == "tools" or "tests" in parts[:-1]:
            continue
        rel = p.relative_to(repo).as_posix()
        for rid, body in regions_of(p.read_text(encoding="utf-8"), rel).items():
            out[f"{rel}#{rid}"] = body
    return out


# ---- docs pages ---------------------------------------------------------------------------------

@dataclass
class Block:
    start: int              # index of the fence-open line
    end: int                # index of the fence-close line
    lang: str
    body: str
    ref: Optional[str]      # the region named by a marker above the fence
    marker_at: Optional[int]


def blocks_of(lines: List[str]) -> List[Block]:
    out, k = [], 0
    while k < len(lines):
        m = FENCE_OPEN.match(lines[k])
        if not m:
            k += 1
            continue
        ticks, lang = m.group(1), m.group(2)
        j = k + 1
        while j < len(lines) and lines[j].rstrip() != ticks:
            j += 1
        ref = marker_at = None
        c = k - 1
        while c >= 0 and COMMENT.match(lines[c]):
            mm = MARKER.match(lines[c])
            if mm:
                ref, marker_at = mm.group(1), c
            c -= 1
        out.append(Block(k, j, lang, _trim(lines[k + 1:j]), ref, marker_at))
        k = j + 1
    return out


@dataclass
class Drift:
    where: str
    ref: str
    problem: str

    def __str__(self) -> str:
        return f"{self.where}: {self.ref}: {self.problem}"


def sync_page(path: Path, regions: Dict[str, str], *, write: bool, label: str) -> Tuple[List[Drift], int]:
    """Check (and with write=True, fix) one docs page. Returns (drifts, number of marked blocks)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    drifts: List[Drift] = []
    marked = 0
    for b in reversed(blocks_of(lines)):            # bottom up, so line numbers above stay valid while writing
        if b.ref is None:
            continue
        marked += 1
        where = f"{label}:{b.start + 1}"
        if b.ref not in regions:
            drifts.append(Drift(where, b.ref, "no such region in the repository"))
            continue
        if b.body != regions[b.ref]:
            drifts.append(Drift(where, b.ref, "docs block differs from the region"))
            if write:
                lines[b.start + 1:b.end] = regions[b.ref].rstrip("\n").split("\n")
    if write and drifts:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list(reversed(drifts)), marked


def mark_page(path: Path, regions: Dict[str, str], *, label: str) -> List[str]:
    """Add a marker above every unmarked block whose code equals exactly one region. Returns what it marked."""
    by_body: Dict[str, List[str]] = {}
    for ref, body in regions.items():
        by_body.setdefault(body, []).append(ref)
    lines = path.read_text(encoding="utf-8").splitlines()
    done = []
    for b in reversed(blocks_of(lines)):
        refs = by_body.get(b.body, [])
        if b.ref is None and len(refs) == 1:
            lines.insert(b.start, f"<!-- example: {refs[0]} -->")
            done.append(f"{label}:{b.start + 1}: {refs[0]}")
    if done:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list(reversed(done))


def docs_pages(docs: Path) -> List[Path]:
    return _walk(docs, {".md", ".mdx"})


# ---- notebooks ----------------------------------------------------------------------------------

def _cell_source(cell) -> str:
    src = cell.get("source", "")
    return _trim(("".join(src) if isinstance(src, list) else src).splitlines())


def sync_notebook(path: Path, regions: Dict[str, str], *, write: bool, label: str) -> Tuple[List[Drift], int]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    drifts, marked = [], 0
    for k, cell in enumerate(nb.get("cells", [])):
        ref = (cell.get("metadata", {}).get("example") or {}).get("region")
        if cell.get("cell_type") != "code" or not ref:
            continue
        marked += 1
        if ref not in regions:
            drifts.append(Drift(f"{label} cell {k}", ref, "no such region in the repository"))
            continue
        if _cell_source(cell) != regions[ref]:
            drifts.append(Drift(f"{label} cell {k}", ref, "notebook cell differs from the region"))
            if write:
                body = regions[ref].rstrip("\n").split("\n")
                cell["source"] = [ln + "\n" for ln in body[:-1]] + [body[-1]]
                cell["outputs"], cell["execution_count"] = [], None
    if write and drifts:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return drifts, marked


# ---- command line -------------------------------------------------------------------------------

def run(cmd: str, repo: Path = REPO, docs: Optional[Path] = None, out=sys.stdout) -> int:
    try:
        regions = all_regions(repo)
    except SyncError as exc:
        print(f"docs_sync: {exc}", file=out)
        return 2
    if cmd == "list":
        for ref in sorted(regions):
            print(ref, file=out)
        return 0
    if cmd == "mark":
        root = docs if docs is not None else repo        # without --docs: this repository's own pages
        n = 0
        for page in docs_pages(root):
            for line in mark_page(page, regions, label=page.relative_to(root).as_posix()):
                print(f"marked {line}", file=out)
                n += 1
        print(f"docs_sync: {n} marker(s) added", file=out)
        return 0
    write = cmd == "write"
    drifts: List[Drift] = []
    used = set()
    n_nb = n_docs = n_own = 0
    for nb in _walk(repo, {".ipynb"}):
        d, m = sync_notebook(nb, regions, write=write, label=nb.relative_to(repo).as_posix())
        drifts += d
        n_nb += m
    for page in docs_pages(repo):                       # READMEs here that show a script's code
        d, m = sync_page(page, regions, write=write, label=page.relative_to(repo).as_posix())
        drifts += d
        n_own += m
    if docs is not None:
        for page in docs_pages(docs):
            d, m = sync_page(page, regions, write=write, label=page.relative_to(docs).as_posix())
            drifts += d
            n_docs += m
            used.update(b.ref for b in blocks_of(page.read_text(encoding="utf-8").splitlines()) if b.ref)
    for d in drifts:
        print(("fixed " if write else "") + str(d), file=out)
    summary = (f"docs_sync: {n_docs} docs block(s), {n_own} block(s) in this repository's pages and {n_nb} "
               f"notebook cell(s) tied to {len(regions)} region(s)")
    if docs is not None:
        unused = sorted(set(regions) - used)
        if unused:
            summary += f"; {len(unused)} region(s) not shown in these docs"
    print(summary + f"; {len(drifts)} drift(s)" + (" fixed" if write and drifts else ""), file=out)
    return 1 if drifts and not write else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["list", "check", "write", "mark"])
    p.add_argument("--docs", help="a checkout of the docs source (its pages are searched recursively)")
    p.add_argument("--repo", default=str(REPO))
    a = p.parse_args(argv)
    return run(a.cmd, Path(a.repo), Path(a.docs) if a.docs else None)


if __name__ == "__main__":
    sys.exit(main())
