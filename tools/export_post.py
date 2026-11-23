#!/usr/bin/env python3
"""Export one blog post's companion code from the internal blog branch into posts/<yyyy-mm>-<slug>/.

    python tools/export_post.py <internal checkout>/.../blog/qubo-puzzles \\
        --slug qubo-puzzles --month 2026-11 --product qubo --domains puzzles \\
        --title "Sudoku, N-queens and map coloring as constrained QUBOs" \\
        --summary "One-hot models with exact rows." --level beginner \\
        --denylist /path/outside/the/repo/terms.txt

What it does, in order, in a staging directory next to the destination:

1. Copies only reader files: *.py, *.md, *.ipynb, *.json, *.csv, *.txt, *.toml and tests/. It never copies the
   internal run notes (the EXPERIMENTS, FAIRNESS, POST, NOTES* and RUN files), results (*.jsonl), figures,
   caches or anything over 256 KB, and it lists what it left out.
2. Rewrites the code for this repository's layout: the shared helper module is `examplekit` in shared/; its path
   insertion is rewritten, and internal helper module names are renamed by the export config's "kit_map". A post
   that imports a module on the config's "refuse_imports" list (a helper not yet ported here) is refused.
3. Rewrites README.md for readers: drops the internal "How we measured" block and every line naming the internal
   run notes, removes the release placeholder after product names, and adds a link to the published post and the
   docs (with UTM tags), a "Results" section pointing at the post, and a short "Run it yourself" section. The free
   trial is mentioned once, at the key step.
4. Writes example.json (the gallery metadata).
Internal names (helper modules, denylist paths) come from an export config kept OUTSIDE this repository:
    --config /path/outside/export.json   {"kit_map": {"<internal module>": "examplekit"},
                                          "refuse_imports": ["<module>"], "denylists": ["/path/terms.txt"]}

5. Runs tools/leak_check.py over the staged files with the organization denylist (required unless
   --no-denylist, which is for tests only). Unfilled draft placeholders fail the export unless --draft. The
   check also refuses any stand-in solver for the service (rule `stand-in-solver`): examples here need a key,
   and only the service answers them.
   Only a clean result is moved into place; otherwise nothing is written.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import leak_check  # noqa: E402
from links import utm  # noqa: E402

COPY_SUFFIXES = {".py", ".md", ".ipynb", ".json", ".csv", ".txt", ".toml"}
NEVER_COPY = re.compile(r"^(?:EXPERIMENTS|FAIRNESS|POST|NOTES[\w-]*|RUN)\.(?:md|ya?ml)$|\.jsonl$|^RUN\.yaml$", re.I)
SKIP_DIRS = {"__pycache__", ".pytest_cache", "runs", "figures", "out", ".ipynb_checkpoints"}
MAX_FILE = 256_000

PATH_INSERT = re.compile(r"sys\.path\.insert\(0, str\(Path\(__file__\)\.resolve\(\)\.parents\[1\]\)\)[^\n]*")
PATH_INSERT_NEW = 'sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))  # examplekit'
CONFTEST_INSERT = re.compile(r"sys\.path\.insert\(0, str\(HERE\.parent\)\)")
CONFTEST_INSERT_NEW = 'sys.path.insert(0, str(HERE.parents[1] / "shared"))  # examplekit'
PRODUCT_VERSION = re.compile(r"((?:4colors Optimizer for QUBO|4colors Quantum Compiler|fourcolors client)) \[VERSION\]")

PRODUCTS = {"qubo": ("4colors Optimizer for QUBO", "https://docs.4colors-research.com/qubo/"),
            "compiler": ("4colors Quantum Compiler", "https://docs.4colors-research.com/compiler/"),
            "both": ("4colors Optimizer for QUBO and 4colors Quantum Compiler", "https://docs.4colors-research.com/")}
# The trial sign-up link is not confirmed yet: the placeholder stays until the owner gives the URL. Once it is a
# real URL, exported READMEs carry it with UTM tags (tools/links.py), so trial sign-ups can be traced to the repo.
TRIAL_URL = "[TRIAL SIGNUP URL]"


def trial_link(campaign: str, content: str) -> str:
    return TRIAL_URL if TRIAL_URL.startswith("[") else utm(TRIAL_URL, campaign, content)


class ExportError(RuntimeError):
    pass


def transform_python(text: str, name: str, kit_map: Dict[str, str], refuse: List[str]) -> str:
    for kit in refuse:
        if re.search(rf"^\s*(?:import|from)\s+{re.escape(kit)}\b", text, re.M):
            raise ExportError(f"{name} imports a helper module that is not ported to shared/ yet (refuse list)")
    text = PATH_INSERT.sub(PATH_INSERT_NEW, text)
    text = CONFTEST_INSERT.sub(CONFTEST_INSERT_NEW, text)
    for old, new in kit_map.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def _drop_section(text: str, heading: str) -> Tuple[str, bool]:
    """Remove a '## heading' section up to the next heading of level 1 or 2."""
    lines = text.splitlines(keepends=True)
    out, skipping, dropped = [], False, False
    for ln in lines:
        if re.match(r"^#{1,2} ", ln):
            skipping = ln.strip().lower() == f"## {heading}".lower()
            dropped = dropped or skipping
        if not skipping:
            out.append(ln)
    return "".join(out), dropped


def transform_readme(text: str, meta: dict) -> str:
    text, _ = _drop_section(text, "How we measured")
    text = "".join(ln for ln in text.splitlines(keepends=True)
                   if not re.search(r"EXPERIMENTS\.md|FAIRNESS\.md|run record|recorded run", ln, re.I))
    text = PRODUCT_VERSION.sub(r"\1", text)
    campaign = meta["slug"]
    # One trial mention, at the key step (## Setup); without a Setup section it goes in "Run it yourself".
    trial = f"Start a free trial at {trial_link(campaign, 'readme-setup')} and copy your key, then:\n\n"
    has_setup = re.search(r"^## Setup\n", text, re.M) is not None
    text = re.sub(r"^## Setup\n\n(?!Start a free trial)", lambda m: m.group(0) + trial, text, count=1, flags=re.M)
    product_name, docs = PRODUCTS[meta["product"]]
    post = meta.get("post_url")
    links = []
    if post:
        links.append(f"**Read the post:** [{meta['title']}]({utm(post, campaign, 'readme-top')})")
    links.append(f"**Docs:** [{product_name}]({utm(docs, campaign, 'readme-top')})")
    banner = " · ".join(links) + "\n"
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith("# "):
        lines.insert(1, "\n" + banner)
    else:
        lines.insert(0, banner + "\n")
    text = "".join(lines).rstrip() + "\n"
    results = ("\n## Results\n\n"
               + (f"The numbers and charts are in [the post]({utm(post, campaign, 'readme-results')}). "
                  if post else "The numbers and charts are in the post (link added on publication). ")
               + "Every number there names the 4colors release and the `fourcolors` client version it was measured "
               "with; each row this code writes records both (`engine_version`, `client_version`).\n")
    footer = ("\n## Run it yourself\n\n"
              + ("" if has_setup else f"Start a free trial at {trial_link(campaign, 'readme-trial')} for a key. ")
              + "Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your "
              "shell history) and run the commands above. Every answer comes from the 4colors service.\n")
    return text + results + footer


def collect(src: Path) -> Tuple[List[Path], List[str]]:
    keep, left = [], []
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.is_dir():
            continue
        if NEVER_COPY.search(p.name):
            left.append(f"{rel} (internal)")
        elif p.suffix.lower() not in COPY_SUFFIXES:
            left.append(f"{rel} (type not exported)")
        elif p.stat().st_size > MAX_FILE:
            left.append(f"{rel} (over {MAX_FILE} bytes)")
        else:
            keep.append(rel)
    return keep, left


def export(src: Path, dest_root: Path, meta: dict, *, denylists: List[str], draft: bool = False,
           force: bool = False, require_denylist: bool = True, allowlist: Optional[Path] = None,
           kit_map: Optional[Dict[str, str]] = None, refuse_imports: Optional[List[str]] = None) -> dict:
    kit_map = dict(kit_map or {})
    refuse_imports = list(refuse_imports or [])
    if not (src / "README.md").exists():
        raise ExportError(f"{src} has no README.md")
    if require_denylist and not denylists:
        raise ExportError("no denylist: pass --denylist or set FOURCOLORS_LEAK_DENYLIST (organization terms are "
                          "kept outside the repository)")
    if not re.fullmatch(r"20\d\d-[01]\d", meta["month"]) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", meta["slug"]):
        raise ExportError("month must be yyyy-mm and slug lowercase letters, digits and dashes")
    if meta["product"] not in PRODUCTS:
        raise ExportError(f"product must be one of {sorted(PRODUCTS)}")
    name = f"{meta['month']}-{meta['slug']}"
    dest = dest_root / name
    if dest.exists() and not force:
        raise ExportError(f"{dest} exists (use --force to replace it)")
    dest_root.mkdir(parents=True, exist_ok=True)
    keep, left = collect(src)
    stage = Path(tempfile.mkdtemp(prefix=f".export-{name}-", dir=dest_root))
    try:
        for rel in keep:
            text = (src / rel).read_text(encoding="utf-8")
            if rel.suffix == ".py":
                text = transform_python(text, str(rel), kit_map, refuse_imports)
            elif rel == Path("README.md"):
                text = transform_readme(text, meta)
            out = stage / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        example = {"kind": "post", "slug": meta["slug"], "month": meta["month"], "title": meta["title"],
                   "summary": meta.get("summary", ""), "product": meta["product"],
                   "domains": meta.get("domains", []), "level": meta.get("level", "intermediate"),
                   "post_url": meta.get("post_url") or None, "requires": "fourcolors[optimizer]>=1.0.0"
                   if meta["product"] == "qubo" else "fourcolors>=1.0.0",
                   "tests": (stage / "tests").is_dir(),
                   "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (stage / "example.json").write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")
        deny = leak_check.load_denylists(denylists)
        allow = leak_check.load_allowlist(allowlist or leak_check.DEFAULT_ALLOWLIST)
        findings = leak_check.check_paths([str(stage)], deny=deny, allow=allow, placeholders=not draft)
        findings = [leak_check.Finding(f.path.replace(str(stage), name), f.line, f.rule, f.detail) for f in findings]
        if findings:
            raise ExportError("leak check failed; nothing written:\n" + "\n".join(f"  {f}" for f in findings))
        if dest.exists():
            shutil.rmtree(dest)
        stage.rename(dest)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"dest": str(dest), "copied": [str(r) for r in keep], "left_out": left, "denylist_terms": len(deny)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="the post directory in an internal checkout of the blog branch")
    p.add_argument("--slug", required=True)
    p.add_argument("--month", required=True, help="publication month, yyyy-mm")
    p.add_argument("--title", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--product", required=True, choices=sorted(PRODUCTS))
    p.add_argument("--domains", default="", help="comma-separated, e.g. finance,puzzles")
    p.add_argument("--level", default="intermediate", choices=["beginner", "intermediate", "advanced"])
    p.add_argument("--post-url", default="")
    p.add_argument("--dest", default=str(REPO / "posts"))
    p.add_argument("--denylist", action="append", default=[])
    p.add_argument("--config", help="export config JSON kept outside the repository (kit_map, refuse_imports, "
                                    "denylists)")
    p.add_argument("--draft", action="store_true", help="allow unfilled placeholders (fix them before merging)")
    p.add_argument("--force", action="store_true", help="replace an existing export")
    p.add_argument("--no-denylist", action="store_true", help="tests only: run without organization terms")
    a = p.parse_args(argv)
    meta = {"slug": a.slug, "month": a.month, "title": a.title, "summary": a.summary, "product": a.product,
            "domains": [d.strip() for d in a.domains.split(",") if d.strip()], "level": a.level,
            "post_url": a.post_url}
    cfg = json.loads(Path(a.config).expanduser().read_text(encoding="utf-8")) if a.config else {}
    try:
        res = export(Path(a.src), Path(a.dest), meta,
                     denylists=leak_check.denylist_paths(a.denylist + cfg.get("denylists", [])),
                     draft=a.draft, force=a.force, require_denylist=not a.no_denylist,
                     kit_map=cfg.get("kit_map", {}), refuse_imports=cfg.get("refuse_imports", []))
    except ExportError as exc:
        print(f"export_post: {exc}", file=sys.stderr)
        return 1
    print(f"exported to {res['dest']} ({len(res['copied'])} files, {res['denylist_terms']} denylist terms checked)")
    for x in res["left_out"]:
        print(f"  left out: {x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
