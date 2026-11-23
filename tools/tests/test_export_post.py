"""export_post on a synthetic post laid out like the internal blog branch."""
import json
from pathlib import Path

import pytest

import export_post as ep

NOTES = "EXPERIMENTS" + ".md"
VER = "[" + "VERSION]"

META = {"slug": "toy", "month": "2026-11", "title": "A toy post", "summary": "s", "product": "qubo",
        "domains": ["puzzles"], "level": "beginner", "post_url": "https://4colors-research.com/blog/toy/"}

README = f"""# A toy post

Hands-on companion code for the post, against the 4colors Optimizer for QUBO {VER}.

## Setup

pip install "fourcolors[optimizer]"

Rerun the jobs as listed in {NOTES} before citing.

## How we measured

> Seeds 1 2 3. Run record: the internal id.

## Tests

python -m pytest -q tests
"""

SCRIPT = '''import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # internalkit
sys.path.insert(0, str(Path(__file__).resolve().parent))
import internalkit

print(internalkit.client)
'''

CONFTEST = '''import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import internalkit
'''


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "internal" / "toy"
    (d / "tests").mkdir(parents=True)
    (d / "__pycache__").mkdir()
    (d / "README.md").write_text(README)
    (d / NOTES).write_text("internal commands\n")
    (d / "solve.py").write_text(SCRIPT)
    (d / "tests" / "conftest.py").write_text(CONFTEST)
    (d / "results.jsonl").write_text("{}\n")
    (d / "__pycache__" / "solve.cpython-312.pyc").write_bytes(b"\0")
    (d / "chart.png").write_bytes(b"\x89PNG")
    return d


def run(src, tmp_path, **kw):
    kw.setdefault("kit_map", {"internalkit": "examplekit"})
    kw.setdefault("refuse_imports", ["otherkit"])
    kw.setdefault("denylists", [])
    kw.setdefault("require_denylist", False)
    return ep.export(src, tmp_path / "posts", dict(META), **kw)


def test_export_copies_reader_files_and_rewrites_them(src, tmp_path):
    res = run(src, tmp_path)
    out = Path(res["dest"])
    assert out.name == "2026-11-toy"
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == ["README.md", "example.json", "solve.py", "tests/conftest.py"]
    assert any(NOTES in x for x in res["left_out"]) and any("results.jsonl" in x for x in res["left_out"])
    py = (out / "solve.py").read_text()
    assert "internalkit" not in py and "import examplekit" in py and 'parents[2] / "shared"' in py
    assert 'HERE.parents[1] / "shared"' in (out / "tests" / "conftest.py").read_text()
    md = (out / "README.md").read_text()
    assert "How we measured" not in md and "EXPERIMENTS" not in md and VER not in md
    assert "utm_source=github" in md and "## Results" in md and "account.4colors-research.com" in md
    meta = json.loads((out / "example.json").read_text())
    assert meta["kind"] == "post" and meta["slug"] == "toy" and meta["offline_tests"] is True
    assert not list((tmp_path / "posts").glob(".export-*"))


def test_a_leak_stops_the_export_and_writes_nothing(src, tmp_path):
    (src / "solve.py").write_text(SCRIPT + "KEY = '/ho" + "me/alice/team" + ".key'\n")
    with pytest.raises(ep.ExportError, match="leak check failed"):
        run(src, tmp_path)
    assert not (tmp_path / "posts" / "2026-11-toy").exists()
    assert not list((tmp_path / "posts").glob(".export-*"))


def test_denylist_terms_stop_the_export(src, tmp_path):
    dl = tmp_path / "terms.txt"
    dl.write_text("hostal" + "pha\n")
    (src / "solve.py").write_text(SCRIPT + "# ran on HOSTAL" + "PHA\n")
    with pytest.raises(ep.ExportError) as e:
        run(src, tmp_path, denylists=[str(dl)], require_denylist=True)
    assert "denylist" in str(e.value) and "hostal" + "pha" not in str(e.value).lower()


def test_denylist_is_required_by_default(src, tmp_path):
    with pytest.raises(ep.ExportError, match="no denylist"):
        ep.export(src, tmp_path / "posts", dict(META), denylists=[])


def test_placeholders_need_draft(src, tmp_path):
    (src / "README.md").write_text(README.replace("## How we measured", "Matches on [" + "N] of them.\n\n## How we measured"))
    with pytest.raises(ep.ExportError, match="placeholder"):
        run(src, tmp_path)
    assert Path(run(src, tmp_path, draft=True)["dest"]).exists()


def test_unported_helper_is_refused(src, tmp_path):
    (src / "solve.py").write_text("import otherkit\n")
    with pytest.raises(ep.ExportError, match="not ported"):
        run(src, tmp_path)


def test_existing_export_needs_force(src, tmp_path):
    run(src, tmp_path)
    with pytest.raises(ep.ExportError, match="exists"):
        run(src, tmp_path)
    assert Path(run(src, tmp_path, force=True)["dest"]).exists()
