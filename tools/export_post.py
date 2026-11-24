#!/usr/bin/env python3
"""Export one blog post's companion code from the internal blog branch into posts/<yyyy-mm>-<slug>/.

    python tools/export_post.py <internal checkout>/.../blog/qubo-puzzles \\
        --slug qubo-puzzles --month 2026-11 --products qubo --domains puzzles \\
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

   With --code-only (code before the post's numbers exist), README.md keeps only its how-to sections, other
   Markdown files (post text) are left out, and example.json says "code_only": true.
5. Runs tools/leak_check.py over the staged files with the organization denylist (required unless
   --no-denylist, which is for tests only). Unfilled draft placeholders fail the export unless --draft. The
   check also refuses any stand-in solver for the service (rule `stand-in-solver`): examples here need a key,
   and only the service answers them.
   Only a clean result is moved into place; otherwise nothing is written.
"""
from __future__ import annotations

import argparse
import ast
import os
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
import products as P  # noqa: E402
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

# The trial sign-up page. Exported READMEs carry it with UTM tags (tools/links.py), so trial sign-ups can be traced
# to this repository.
TRIAL_URL = "https://account.4colors-research.com/account/"


def trial_link(campaign: str, content: str) -> str:
    return f"[account.4colors-research.com]({utm(TRIAL_URL, campaign, content)})"


class ExportError(RuntimeError):
    pass


# ---- import paths: where each sys.path entry of an internal post really points -----------------------
#
# Internal posts live in <blog>/<post>/ and reach helpers through sys.path entries written in many forms
# (parents[1], HERE.parent, os.path.join(dirname(__file__), ".."), sys.path[:0] = [...], ...). Each entry is
# evaluated as if the file still sat at its internal place, and the directory it names is mapped by
# `path_roots` (relative to the post, from the export config; default below) to its public place:
#   "shared"      -> <repo>/shared (examplekit, circuitkit, qaoakit)
#   "post-tests"  -> this post's own tests/ (the shared test helpers are copied there)
# A directory inside the post keeps its expression; a sibling post maps to posts/<month>-<sibling>/; anything
# else stops the export.
DEFAULT_PATH_ROOTS = {"..": "shared", "../tests": "post-tests"}
_PL = '__import__("pathlib").Path(__file__).resolve()'


def _safe_import(name, *a, **kw):
    if name in ("pathlib", "os", "os.path"):
        return __import__(name, *a, **kw)
    raise ImportError(name)


def _path_nodes(tree: ast.AST):
    """Every expression that sys.path receives: insert(i, X), append(X), and sys.path[:k] = [X, Y, ...]."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("insert", "append") and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "path" and node.args):
            yield node.args[-1]
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript)
              and isinstance(node.targets[0].value, ast.Attribute) and node.targets[0].value.attr == "path"
              and isinstance(node.value, (ast.List, ast.Tuple))):
            yield from node.value.elts


def _namespace(tree: ast.AST, text: str, internal_file: Path) -> dict:
    """Module-level names the path expressions use (HERE = Path(__file__).resolve().parent, DIR, ...)."""
    import pathlib
    ns = {"__builtins__": {"__import__": _safe_import, "str": str}, "__file__": str(internal_file),
          "Path": pathlib.Path, "_Path": pathlib.Path, "os": os, "pathlib": pathlib}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and all(isinstance(x, ast.Name) for x in node.targets):
            try:
                val = eval(ast.get_source_segment(text, node.value), ns)    # noqa: S307 - our own blog source
            except Exception:
                continue
            for x in node.targets:
                ns[x.id] = val
    return ns


def rewrite_paths(text: str, rel: Path, post_src: Path, path_roots: Dict[str, str], month: str) -> str:
    """Rewrite the sys.path entries of one post file (internal place post_src/rel) for posts/<month>-<post>/rel."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ExportError(f"{rel}: not valid Python ({exc})")
    internal_file = (post_src / rel).resolve()
    ns = _namespace(tree, text, internal_file)
    roots = {os.path.normpath(post_src.resolve() / k): v for k, v in path_roots.items()}
    post_dir = os.path.normpath(post_src.resolve())
    blog_dir = os.path.dirname(post_dir)
    k = len(rel.parts) + 1                    # parents[k] of a public file is the repository root
    edits = []
    for node in _path_nodes(tree):
        src = ast.get_source_segment(text, node)
        try:
            target = os.path.normpath(os.path.abspath(str(eval(src, ns))))   # noqa: S307 - our own blog source
        except Exception as exc:
            raise ExportError(f"{rel}:{node.lineno}: cannot work out where `{src}` points ({exc})")
        if target == post_dir or target.startswith(post_dir + os.sep):
            continue                          # inside the post: same layout in public
        kind = roots.get(target)
        if kind == "shared":
            new = f'str({_PL}.parents[{k}] / "shared")'
        elif kind == "post-tests":
            new = f'str({_PL}.parents[{k - 2}] / "tests")'
        elif os.path.dirname(target) == blog_dir and os.path.isdir(target):
            sib = os.path.basename(target)
            new = f'str(next(iter(sorted({_PL}.parents[{k - 1}].glob("*-{sib}")))))'
        else:
            raise ExportError(f"{rel}:{node.lineno}: `{src}` points outside the post and its helpers")
        edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, new))
    if not edits:
        return text
    lines = text.splitlines(keepends=True)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))
    for l0, c0, l1, c1, new in sorted(edits, reverse=True):
        a, b = offs[l0 - 1] + c0, offs[l1 - 1] + c1
        text = text[:a] + new + text[b:]
    return text


QUBO_USE = re.compile(r"fourcolors\.qubo\b|fourcolors\.solve\b|FourColorsSampler|\bqubo\.(?:solve|submit|upload)\(|"
                      r"optimizer\.qubo")
COMPILER_USE = re.compile(r"fourcolors\.quantum\b|\bcompiler\.(?:compile|submit)\(|quantum\.compiler|circuitkit|"
                          r"qaoakit|compile_4colors")


def detect_products(src: Path) -> List[str]:
    """The products a post's code calls: the QUBO optimizer, the compiler, or both (from its .py files)."""
    code = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted(src.rglob("*.py"))
                     if "tests" not in p.relative_to(src).parts)
    out = [s for s, rx in (("qubo", QUBO_USE), ("compiler", COMPILER_USE)) if rx.search(code)]
    return out or ["qubo"]


def readme_title(src: Path) -> str:
    for line in (src / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise ExportError(f"{src}/README.md has no '# ' title; pass --title")


def imported_names(text: str) -> set:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.add(node.module.split(".")[0])
    return out


REQ = re.compile(r"^fourcolors(?:\[([\w,]+)\])?>=([\d.]+)$")


def requires(slugs: List[str], known: Dict[str, dict]) -> str:
    """What a post needs: the products' `fourcolors` requirements merged (all extras, the highest minimum
    version); a post with no Python product gets its first product's requirement as written."""
    reqs = [known[s]["requires"] for s in slugs]
    extras, versions = set(), []
    for r in reqs:
        m = REQ.match(r)
        if m:
            extras.update(x for x in (m.group(1) or "").split(",") if x)
            versions.append(tuple(int(v) for v in m.group(2).split(".")))
    if not versions:
        return reqs[0]
    ex = f"[{','.join(sorted(extras))}]" if extras else ""
    return f"fourcolors{ex}>={'.'.join(map(str, max(versions)))}"


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


PLACEHOLDER_ANY = re.compile(r"\[(?:N|VERSION|CHECK[^\]]*|OD-[^\]]*|run id|source|dates|True/False)\]")
KEY_PATH = re.compile(r"/path/to/[\w.-]*\.(?:key|pem)\b")
#: Sections a code-only README keeps (matched case-insensitively on the "## " heading, by prefix); every other
#: section, and any "###" under a dropped one, is left out: the post itself carries the story and the numbers.
CODE_ONLY_KEEP = ("what you will learn", "what you get", "the model", "setup", "step by step", "the process",
                  "step ", "data", "run the unit tests", "try it on your own", "what each row in results.jsonl")


def fix_text(text: str, kit_map: Dict[str, str]) -> str:
    """Reader-facing text in Markdown files: internal helper names become their public names (with paths into
    shared/), and a key-file path in a setup line becomes a neutral one."""
    for old, new in kit_map.items():
        text = re.sub(rf"(?:\.\./)?\b{re.escape(old)}\.py\b", f"../../shared/{new}.py", text)
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return KEY_PATH.sub("/path/to/key-file", text)


def _strip_comment_placeholder(line: str) -> str:
    """A code line whose trailing comment holds a placeholder keeps its code and loses the comment."""
    m = re.match(r"^(.*?)\s+#.*$", line)
    if m and PLACEHOLDER_ANY.search(line) and not PLACEHOLDER_ANY.search(m.group(1)):
        return m.group(1).rstrip() + "\n"
    return line


def code_only(text: str) -> Tuple[str, List[str]]:
    """Keep the title and the how-to sections (CODE_ONLY_KEEP); inside them drop any paragraph that still holds a
    result placeholder, strip placeholder comments from code, and drop a code block whose code holds one.
    Returns the text and the headings it left out."""
    out: List[str] = []
    dropped: List[str] = []
    keep = False
    lines = text.splitlines(keepends=True)
    k = 0
    para: List[str] = []

    def flush():
        if para and not any(PLACEHOLDER_ANY.search(x) for x in para):
            out.extend(para)
        para.clear()

    while k < len(lines):
        ln = lines[k]
        if ln.startswith("# "):
            flush()
            out.append(ln)
            keep = False                      # the lede (before the first section) is post text: left out
        elif ln.startswith("## "):
            flush()
            head = ln[3:].strip().lower()
            keep = head.startswith(CODE_ONLY_KEEP)
            (out.append("\n" + ln) if keep else dropped.append(ln[3:].strip()))
        elif ln.lstrip().startswith("```"):
            flush()
            j = k + 1
            while j < len(lines) and not lines[j].lstrip().startswith("```"):
                j += 1
            block = [ln] + [_strip_comment_placeholder(x) for x in lines[k + 1:j]] + lines[j:j + 1]
            if keep and not any(PLACEHOLDER_ANY.search(x) for x in block):
                out.extend(block)
            elif keep:
                # The block goes; so does the sentence that introduced it ("In code:"), never left dangling.
                while out and not out[-1].strip():
                    out.pop()
                if out and out[-1].rstrip().endswith(":") and not out[-1].startswith("#"):
                    out.pop()
                out.append("\n")
            k = j + 1
            continue
        elif keep:
            if ln.strip():
                para.append(ln)
            else:
                flush()
                out.append(ln)
        k += 1
    flush()
    text = re.sub(r"\n{3,}", "\n\n", "".join(out))
    return text, dropped


def tag_trial_links(text: str, campaign: str, content: str) -> str:
    """Bare trial links in post text get the repository's UTM tags, so sign-ups from each post can be counted."""
    return re.sub(re.escape(TRIAL_URL) + r"(?![?\w])", lambda m: utm(TRIAL_URL, campaign, content), text)


def transform_readme(text: str, meta: dict, *, kit_map: Optional[Dict[str, str]] = None,
                     code_only_mode: bool = False) -> str:
    text = tag_trial_links(fix_text(text, kit_map or {}), meta["slug"], "readme-text")
    if code_only_mode:
        text, _ = code_only(text)
    text, _ = _drop_section(text, "How we measured")
    text = "".join(ln for ln in text.splitlines(keepends=True)
                   if not re.search(r"EXPERIMENTS\.md|FAIRNESS\.md|run record|recorded run", ln, re.I))
    text = PRODUCT_VERSION.sub(r"\1", text)
    campaign = meta["slug"]
    # One trial mention, at the key step (## Setup); without a Setup section it goes in "Run it yourself".
    trial = f"Start a free trial at {trial_link(campaign, 'readme-setup')} and copy your key, then:\n\n"
    has_setup = re.search(r"^## Setup\n", text, re.M) is not None
    text = re.sub(r"^## Setup\n\n(?!Start a free trial)", lambda m: m.group(0) + trial, text, count=1, flags=re.M)
    known = P.by_slug()
    post = meta.get("post_url")
    links = []
    if post:
        links.append(f"**Read the post:** [{meta['title']}]({utm(post, campaign, 'readme-top')})")
    docs = ", ".join(f"[{known[s]['name']}]({utm(known[s]['docs_url'], campaign, 'readme-top')})"
                     for s in meta["products"])
    links.append(f"**Docs:** {docs}")
    banner = " · ".join(links) + "\n"
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith("# "):
        lines.insert(1, "\n" + banner)
    else:
        lines.insert(0, banner + "\n")
    text = "".join(lines).rstrip() + "\n"
    # Without a published post there is nothing to point to yet: the Results section comes with --post-url.
    results = ("\n## Results\n\n"
               f"The numbers and charts are in [the post]({utm(post, campaign, 'readme-results')}). "
               "Every number there names the 4colors release and the `fourcolors` client version it was measured "
               "with; each row this code writes records both (`engine_version`, `client_version`).\n") if post else ""
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
           kit_map: Optional[Dict[str, str]] = None, refuse_imports: Optional[List[str]] = None,
           code_only_mode: bool = False, path_roots: Optional[Dict[str, str]] = None) -> dict:
    path_roots = dict(path_roots or DEFAULT_PATH_ROOTS)
    kit_map = dict(kit_map or {})
    refuse_imports = list(refuse_imports or [])
    if not (src / "README.md").exists():
        raise ExportError(f"{src} has no README.md")
    if require_denylist and not denylists:
        raise ExportError("no denylist: pass --denylist or set FOURCOLORS_LEAK_DENYLIST (organization terms are "
                          "kept outside the repository)")
    if not re.fullmatch(r"20\d\d-[01]\d", meta["month"]) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", meta["slug"]):
        raise ExportError("month must be yyyy-mm and slug lowercase letters, digits and dashes")
    known = P.by_slug()
    if not meta.get("products") or any(s not in known for s in meta["products"]):
        raise ExportError(f"products must be one or more of {sorted(known)} (products.json)")
    name = f"{meta['month']}-{meta['slug']}"
    dest = dest_root / name
    if dest.exists() and not force:
        raise ExportError(f"{dest} exists (use --force to replace it)")
    dest_root.mkdir(parents=True, exist_ok=True)
    keep, left = collect(src)
    if code_only_mode:                        # other Markdown files are post text: they go out with the post
        left += [f"{r} (post text; code-only export)" for r in keep if r.suffix == ".md" and r != Path("README.md")]
        keep = [r for r in keep if r.suffix != ".md" or r == Path("README.md")]
    # Shared test helpers (a module the post imports that sits in <blog>/tests/) are copied into its tests/.
    helpers_dir = [os.path.normpath(src.resolve() / k) for k, v in path_roots.items() if v == "post-tests"]
    wanted = set()
    for rel in keep:
        if rel.suffix == ".py":
            wanted |= imported_names((src / rel).read_text(encoding="utf-8"))
    extra = {}
    for d in helpers_dir:
        for mod in sorted(wanted):
            h = Path(d) / f"{mod}.py"
            if h.is_file() and Path("tests") / h.name not in keep:
                extra[Path("tests") / h.name] = h
    stage = Path(tempfile.mkdtemp(prefix=f".export-{name}-", dir=dest_root))
    try:
        for rel in list(keep) + sorted(extra):
            origin = extra.get(rel, src / rel)
            text = origin.read_text(encoding="utf-8")
            if rel.suffix == ".py":
                if rel not in extra:
                    text = rewrite_paths(text, rel, src, path_roots, meta["month"])
                text = transform_python(text, str(rel), kit_map, refuse_imports)
            elif rel == Path("README.md"):
                text = transform_readme(text, meta, kit_map=kit_map, code_only_mode=code_only_mode)
            elif rel.suffix == ".md":
                text = tag_trial_links(fix_text(text, kit_map), meta["slug"], "text")
            if rel.suffix == ".py":
                text = tag_trial_links(text, meta["slug"], "script")
            out = stage / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        example = {"kind": "post", "slug": meta["slug"], "month": meta["month"], "title": meta["title"],
                   "summary": meta.get("summary", ""), "products": list(meta["products"]),
                   "domains": meta.get("domains", []), "level": meta.get("level", "intermediate"),
                   "post_url": meta.get("post_url") or None, "requires": requires(meta["products"], known),
                   "tests": (stage / "tests").is_dir(), "code_only": code_only_mode,
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
    return {"dest": str(dest), "copied": [str(r) for r in list(keep) + sorted(extra)], "left_out": left,
            "denylist_terms": len(deny)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="the post directory in an internal checkout of the blog branch")
    p.add_argument("--slug", required=True)
    p.add_argument("--month", required=True, help="publication month, yyyy-mm")
    p.add_argument("--title", help="default: the README's first '# ' heading")
    p.add_argument("--summary", default="")
    p.add_argument("--products", help="comma-separated product slugs from products.json, e.g. qubo or "
                                       "qubo,compiler (default: detected from the post's code)")
    p.add_argument("--domains", default="", help="comma-separated, e.g. finance,puzzles")
    p.add_argument("--level", default="intermediate", choices=["beginner", "intermediate", "advanced"])
    p.add_argument("--post-url", default="")
    p.add_argument("--dest", default=str(REPO / "posts"))
    p.add_argument("--denylist", action="append", default=[])
    p.add_argument("--config", help="export config JSON kept outside the repository (kit_map, refuse_imports, "
                                    "denylists)")
    p.add_argument("--draft", action="store_true", help="allow unfilled placeholders (fix them before merging)")
    p.add_argument("--force", action="store_true", help="replace an existing export")
    p.add_argument("--code-only", action="store_true",
                   help="before the post's numbers exist: keep only the README's how-to sections (setup, steps, "
                        "model, tests), drop the results and the story, and point to the post")
    p.add_argument("--no-denylist", action="store_true", help="tests only: run without organization terms")
    a = p.parse_args(argv)
    src = Path(a.src)
    try:
        title = a.title or readme_title(src)
    except ExportError as exc:
        print(f"export_post: {exc}", file=sys.stderr)
        return 1
    products = [s.strip() for s in a.products.split(",") if s.strip()] if a.products else detect_products(src)
    meta = {"slug": a.slug, "month": a.month, "title": title, "summary": a.summary, "products": products,
            "domains": [d.strip() for d in a.domains.split(",") if d.strip()], "level": a.level,
            "post_url": a.post_url}
    cfg = json.loads(Path(a.config).expanduser().read_text(encoding="utf-8")) if a.config else {}
    try:
        res = export(Path(a.src), Path(a.dest), meta,
                     denylists=leak_check.denylist_paths(a.denylist + cfg.get("denylists", [])),
                     draft=a.draft, force=a.force, require_denylist=not a.no_denylist,
                     kit_map=cfg.get("kit_map", {}), refuse_imports=cfg.get("refuse_imports", []),
                     code_only_mode=a.code_only, path_roots=cfg.get("path_roots"))
    except ExportError as exc:
        print(f"export_post: {exc}", file=sys.stderr)
        return 1
    print(f"exported to {res['dest']} ({len(res['copied'])} files, {res['denylist_terms']} denylist terms checked)")
    for x in res["left_out"]:
        print(f"  left out: {x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
