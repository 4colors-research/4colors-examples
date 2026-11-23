#!/usr/bin/env python3
"""Check files for things that must never reach a public repository.

    python tools/leak_check.py .                                  # generic rules only
    python tools/leak_check.py posts/ --denylist /secure/terms.txt --require-denylist

Two kinds of rules:

1. Generic rules, kept here: home-directory paths, private IP ranges, local host names, key and token shapes,
   key-file names, dated run identifiers, bare commit hashes, e-mail addresses, links the product rules forbid,
   internal record files, British spellings (the repository uses US spelling) and unfilled draft placeholders.
2. Organization terms, kept OUTSIDE the repository in one or more denylist files (internal host names, internal
   tool and team names, excluded data sets, excluded products, held-out instance names). Name them with
   --denylist or the FOURCOLORS_LEAK_DENYLIST variable (paths separated by the OS path separator). A line is a
   literal (matched case-insensitively) or `re:<regex>`; `#` starts a comment. A denylist hit is reported by
   denylist file and line number only, never by the matched text, so running the check prints no listed term.

Exact tokens that the hash and e-mail rules may accept are listed in tools/leak_allowlist.txt.
Exit status: 0 clean, 1 findings, 2 usage error (for example --require-denylist with no denylist).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
DEFAULT_ALLOWLIST = HERE / "leak_allowlist.txt"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "node_modules", ".venv"}
MAX_BYTES = 2_000_000

GENERIC = [
    ("home-path", re.compile(r"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|~/(?:work|code)\b")),
    ("private-ip", re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))"
                              r"\.\d{1,3}\.\d{1,3}\b")),
    ("local-host", re.compile(r"\b[a-z0-9-]+\.(?:local|lan|internal|corp|home\.arpa)\b(?!\()", re.I)),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}"
                         r"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,})")),
    ("secret-literal", re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password)\s*[=:]\s*[\"'][^\"'\s]{12,}[\"']")),
    ("key-file", re.compile(r"[\w~./-]*[\w-]\.(?:key|pem)\b")),
    ("run-id", re.compile(r"\b20\d\d-[01]\d-[0-3]\d-[A-Za-z0-9]+-[A-Za-z0-9][\w.-]*")),
    ("commit-hash", re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b")),
    ("forbidden-link", re.compile(r"api\.4colors-research\.com/(?:account|developers|docs)\b")),
    ("internal-record", re.compile(r"\b(?:RUN\.yaml|ledger\.jsonl|EXPERIMENTS\.md|FAIRNESS\.md)\b")),
    ("british-spelling", re.compile(r"\b(?:licen[c]e[sd]?|col[o]urs?|col[o]ured|col[o]uring|optimi[s](?:e|ed|es|ing|ation)"
                                    r"|behavio[u]r|normali[s](?:e|ed|ation)|catalog[u]e|cent[r]e|model[l]ing|analy[s](?:e|ed|ing))\b",
                                    re.I)),
]
PLACEHOLDER = ("placeholder", re.compile(r"\[(?:N|VERSION|run id|source|dates|True/False)\]"))
EMAIL_OK = re.compile(r"@(?:users\.noreply\.github\.com|example\.(?:com|org))$", re.I)


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}" + (f" ({self.detail})" if self.detail else "")


@dataclass
class DenyTerm:
    source: str        # denylist file name
    lineno: int
    pattern: re.Pattern


def load_denylists(paths: Iterable[str]) -> List[DenyTerm]:
    terms: List[DenyTerm] = []
    for p in paths:
        path = Path(p).expanduser()
        for k, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            rx = re.compile(s[3:], re.I) if s.startswith("re:") else re.compile(re.escape(s), re.I)
            terms.append(DenyTerm(path.name, k, rx))
    return terms


def load_allowlist(path: Optional[Path]) -> set:
    if path is None or not path.exists():
        return set()
    out = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.split("#", 1)[0].strip()
        if s:
            out.add(s)
    return out


def iter_files(roots: Sequence[str]):
    """(path, name relative to its root): only the relative name would be public, so only it is checked."""
    for root in roots:
        p = Path(root)
        if p.is_file():
            yield p, p.name
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for f in sorted(filenames):
                fp = Path(dirpath) / f
                yield fp, str(fp.relative_to(p))


def read_text(path: Path) -> Optional[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > MAX_BYTES or b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def check_text(text: str, name: str, *, deny: Sequence[DenyTerm] = (), allow: set = frozenset(),
               placeholders: bool = True) -> List[Finding]:
    rules = list(GENERIC) + ([PLACEHOLDER] if placeholders else [])
    found: List[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for rule, rx in rules:
            for m in rx.finditer(line):
                tok = m.group(0)
                if rule in ("commit-hash", "email") and tok in allow:
                    continue
                if rule == "email" and EMAIL_OK.search(tok):
                    continue
                # A hash rule hit shows only its length: the text may itself be a private identifier.
                detail = f"{len(tok)} hex chars" if rule == "commit-hash" else tok
                if rule in ("token", "secret-literal", "private-key"):
                    detail = "value withheld"
                found.append(Finding(name, lineno, rule, detail))
        for t in deny:
            if t.pattern.search(line):
                found.append(Finding(name, lineno, "denylist", f"{t.source} entry at line {t.lineno}"))
    return found


def check_paths(roots: Sequence[str], *, deny: Sequence[DenyTerm] = (), allow: set = frozenset(),
                placeholders: bool = True, skip: Sequence[Path] = ()) -> List[Finding]:
    skip_set = {p.resolve() for p in skip}
    found: List[Finding] = []
    for path, rel in iter_files(roots):
        if path.resolve() in skip_set:
            continue
        # The file name itself is public too.
        found += check_text(rel, f"{path} [name]", deny=deny, allow=allow, placeholders=False)
        text = read_text(path)
        if text is not None:
            found += check_text(text, str(path), deny=deny, allow=allow, placeholders=placeholders)
    # a name finding is reported once even if both the name and the path text matched
    seen, out = set(), []
    for f in found:
        key = (f.path, f.line, f.rule, f.detail)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def denylist_paths(cli: Sequence[str]) -> List[str]:
    env = os.environ.get("FOURCOLORS_LEAK_DENYLIST", "")
    return list(cli) + [p for p in env.split(os.pathsep) if p.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--denylist", action="append", default=[], help="file of organization terms (outside the repo)")
    p.add_argument("--require-denylist", action="store_true", help="fail (exit 2) when no denylist is given")
    p.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST))
    p.add_argument("--allow-placeholders", action="store_true", help="drafts: do not flag unfilled placeholders")
    a = p.parse_args(argv)
    lists = denylist_paths(a.denylist)
    if a.require_denylist and not lists:
        print("leak_check: --require-denylist but no denylist (--denylist or FOURCOLORS_LEAK_DENYLIST)",
              file=sys.stderr)
        return 2
    deny = load_denylists(lists)
    allow_path = Path(a.allowlist)
    found = check_paths(a.paths, deny=deny, allow=load_allowlist(allow_path),
                        placeholders=not a.allow_placeholders, skip=[allow_path])
    for f in found:
        print(f)
    print(f"leak_check: {len(found)} finding(s); {len(deny)} denylist term(s) loaded", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
