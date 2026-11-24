"""export_post on a synthetic post laid out like the internal blog branch."""
import json
from pathlib import Path

import pytest

import export_post as ep

NOTES = "EXPERIMENTS" + ".md"
VER = "[" + "VERSION]"

META = {"slug": "toy", "month": "2026-11", "title": "A toy post", "summary": "s", "products": ["qubo"],
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
    assert 'resolve().parents[3] / "shared"' in (out / "tests" / "conftest.py").read_text()
    md = (out / "README.md").read_text()
    assert "How we measured" not in md and "EXPERIMENTS" not in md and VER not in md
    assert "utm_source=github" in md and "## Results" in md and md.count("free trial") == 1
    assert md.index("## Setup\n\nStart a free trial") < md.index('pip install')
    meta = json.loads((out / "example.json").read_text())
    assert meta["kind"] == "post" and meta["slug"] == "toy" and meta["tests"] is True
    assert not list((tmp_path / "posts").glob(".export-*"))


def test_a_leak_stops_the_export_and_writes_nothing(src, tmp_path):
    (src / "solve.py").write_text(SCRIPT + "KEY = '/ho" + "me/alice/team" + ".key'\n")
    with pytest.raises(ep.ExportError, match="leak check failed"):
        run(src, tmp_path)
    assert not (tmp_path / "posts" / "2026-11-toy").exists()
    assert not list((tmp_path / "posts").glob(".export-*"))



def test_a_stand_in_solver_stops_the_export(src, tmp_path):
    line = "\nWithout a " + "key, put `FOURCOLORS_OFF" + "LINE=1` in front: a local stand" + "-in answers.\n"
    (src / "README.md").write_text((src / "README.md").read_text() + line)
    (src / "solve.py").write_text(SCRIPT + "from internalkit import Off" + "lineQubo\n")
    with pytest.raises(ep.ExportError, match="stand-in-solver"):
        run(src, tmp_path)
    assert not (tmp_path / "posts" / "2026-11-toy").exists()

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


def test_requires_merges_the_products_needs():
    import products
    known = products.by_slug()
    assert ep.requires(["qubo"], known) == "fourcolors[optimizer]>=1.0.0"
    assert ep.requires(["qubo", "compiler"], known) == "fourcolors[optimizer]>=1.0.2"
    assert ep.requires(["platform"], known) == "curl and jq"


def test_unknown_product_is_refused(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "README.md").write_text("# x\n")
    with pytest.raises(ep.ExportError, match="products.json"):
        ep.export(src, tmp_path / "posts", dict(META, products=["nope"]), denylists=[],
                  require_denylist=False)



def _layout(tmp_path, files):
    blog = tmp_path / "internal"
    for rel, text in files.items():
        f = blog / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return blog


def test_every_path_form_is_resolved_and_helpers_are_copied(tmp_path):
    hdr = "import os, sys\nfrom pathlib import Path\nHERE = Path(__file__).resolve().parent\n"
    blog = _layout(tmp_path, {
        "toy/README.md": README,
        "toy/run.py": hdr + "sys.path[:0] = [str(HERE.parent), str(HERE)]\n"
                      "sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))\n"
                      "sys.path.insert(1, str(HERE.parent / 'other'))\nimport internalkit\n",
        "toy/tests/test_run.py": "import sys\nsys.path.insert(0, str(__import__('pathlib').Path(__file__)"
                                 ".resolve().parents[2] / 'tests'))\nimport canned\n\ndef test_x():\n    pass\n",
        "tests/canned.py": "# fixed answers for tests\nimport internalkit\n",
        "other/README.md": "# other\n",
    })
    res = ep.export(blog / "toy", tmp_path / "posts", dict(META), denylists=[], require_denylist=False,
                    kit_map={"internalkit": "examplekit"})
    out = Path(res["dest"])
    run_py = (out / "run.py").read_text()
    assert run_py.count('parents[2] / "shared"') == 2 and "str(HERE)]" in run_py      # list form + os.path form
    assert 'parents[1].glob("*-other")' in run_py                                    # a sibling post
    assert 'parents[1] / "tests"' in (out / "tests" / "test_run.py").read_text()     # blog tests -> post tests
    assert (out / "tests" / "canned.py").read_text() == "# fixed answers for tests\nimport examplekit\n"


def test_a_path_outside_the_post_and_its_helpers_is_refused(tmp_path):
    blog = _layout(tmp_path, {"toy/README.md": README,
                              "toy/run.py": "import sys\nsys.path.insert(0, '/opt/somewhere')\n"})
    with pytest.raises(ep.ExportError, match="points outside"):
        ep.export(blog / "toy", tmp_path / "posts", dict(META), denylists=[], require_denylist=False)


def test_code_only_keeps_the_how_to_and_drops_placeholders():
    N, KEY = "[" + "N]", "." + "key"
    text = ("# Title\n\nLede with {N} gates.\n\n## Why it matters\n\nStory.\n\n## Setup\n\nInstall it.\n\n"
            "```bash\npip install x\n```\n\n## Step by step\n\nGood step.\n\nWe saw {N} here.\n\n"
            "```python\nprint(1)   # {N}: the count\n```\n\n```python\nx = {N}\n```\n\n### Sub\n\nKept.\n\n"
            "## Results\n\n| {N} |\n\n## Run the unit tests\n\n```bash\npytest\n```\n").replace("{N}", N)
    out, dropped = ep.code_only(text)
    assert dropped == ["Why it matters", "Results"]
    assert "Lede" not in out and "Story" not in out and N not in out
    assert "Install it." in out and "Good step." in out and "print(1)\n" in out and "Kept." in out
    assert "x = " not in out and "pytest" in out


def test_fix_text_renames_helpers_and_neutralizes_key_paths():
    KEY = "." + "key"
    t = ("They import ../internalkit.py; call `internalkit.f`.\n"
         "export FOURCOLORS_KEY_FILE=/path/to/trial{KEY}\n").replace("{KEY}", KEY)
    out = ep.fix_text(t, {"internalkit": "examplekit"})
    assert "../../shared/examplekit.py" in out and "`examplekit.f`" in out and "/path/to/key-file" in out


def test_products_and_title_are_detected(tmp_path):
    blog = _layout(tmp_path, {"q/README.md": "# Q title\n",
                              "q/run.py": "import fourcolors\nfourcolors.qubo.solve(1)\n",
                              "c/README.md": "# C\n", "c/run.py": "from fourcolors.quantum import compiler\n",
                              "b/README.md": "# B\n", "b/run.py": "import circuitkit\nfourcolors.qubo.submit(1)\n"})
    assert ep.detect_products(blog / "q") == ["qubo"] and ep.readme_title(blog / "q") == "Q title"
    assert ep.detect_products(blog / "c") == ["compiler"]
    assert ep.detect_products(blog / "b") == ["qubo", "compiler"]
