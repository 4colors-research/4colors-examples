import io
import json

import pytest

import docs_sync as ds

SCRIPT = '''import x

# region docs:build
n = 3
edges = [(0, 1)]
# endregion

assert n == 3

# region docs:solve
print(n)
# endregion
'''

PAGE = '''# A page

<!-- example: ex/s.py#build -->
```python
n = 3
edges = [(0, 1)]
```

Some prose.

<!-- UNVERIFIED-EXAMPLE -->
<!-- example: ex/s.py#solve -->
```python
print(n)
```

```python
an_illustration_with_no_marker()
```
'''


def make(tmp_path, script=SCRIPT, page=PAGE):
    repo, docs = tmp_path / "repo", tmp_path / "docs"
    (repo / "ex").mkdir(parents=True)
    (repo / "ex" / "s.py").write_text(script)
    docs.mkdir()
    (docs / "page.md").write_text(page)
    return repo, docs


def run(cmd, repo, docs=None):
    out = io.StringIO()
    rc = ds.run(cmd, repo, docs, out=out)
    return rc, out.getvalue()


def test_regions_are_read():
    r = ds.regions_of(SCRIPT, "s.py")
    assert r == {"build": "n = 3\nedges = [(0, 1)]\n", "solve": "print(n)\n"}


@pytest.mark.parametrize("bad, msg", [
    ("# region docs:a\n# region docs:b\n# endregion\n", "starts inside"),
    ("# region docs:a\nx\n", "never closed"),
    ("# endregion\n", "without a region"),
    ("# region docs:a\n# endregion\n# region docs:a\n# endregion\n", "twice"),
])
def test_bad_regions(bad, msg):
    with pytest.raises(ds.SyncError, match=msg):
        ds.regions_of(bad, "s.py")


def test_in_sync(tmp_path):
    repo, docs = make(tmp_path)
    rc, out = run("check", repo, docs)
    assert rc == 0 and "2 docs block(s)" in out and "0 drift(s)" in out


def test_drift_is_found_and_written_back(tmp_path):
    repo, docs = make(tmp_path, script=SCRIPT.replace("n = 3", "n = 4"))
    rc, out = run("check", repo, docs)
    assert rc == 1 and "page.md:4: ex/s.py#build: docs block differs" in out
    rc, out = run("write", repo, docs)
    assert rc == 0 and "fixed" in out
    text = (docs / "page.md").read_text()
    assert "n = 4\nedges" in text and "an_illustration_with_no_marker()" in text
    assert run("check", repo, docs)[0] == 0


def test_missing_region(tmp_path):
    repo, docs = make(tmp_path, page=PAGE.replace("#solve", "#gone"))
    rc, out = run("check", repo, docs)
    assert rc == 1 and "ex/s.py#gone: no such region" in out


def test_mark_adds_markers_only_for_exact_matches(tmp_path):
    bare = PAGE.replace("<!-- example: ex/s.py#build -->\n", "").replace("<!-- example: ex/s.py#solve -->\n", "")
    repo, docs = make(tmp_path, page=bare)
    rc, out = run("mark", repo, docs)
    assert rc == 0 and "2 marker(s) added" in out
    assert (docs / "page.md").read_text() == PAGE          # each marker lands just above its fence
    assert run("check", repo, docs)[0] == 0


def test_notebook_cells_follow_their_region(tmp_path):
    repo, _ = make(tmp_path)
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
        {"cell_type": "code", "metadata": {"example": {"region": "ex/s.py#build"}}, "source": "n = 1\n",
         "outputs": [], "execution_count": None},
        {"cell_type": "code", "metadata": {}, "source": "free_cell()", "outputs": [], "execution_count": None}]}
    (repo / "ex" / "n.ipynb").write_text(json.dumps(nb))
    assert run("check", repo)[0] == 1
    assert run("write", repo)[0] == 0
    cells = json.loads((repo / "ex" / "n.ipynb").read_text())["cells"]
    assert "".join(cells[0]["source"]) == "n = 3\nedges = [(0, 1)]" and cells[1]["source"] == "free_cell()"
    assert run("check", repo)[0] == 0
