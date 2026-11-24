"""Every notebook is valid JSON and its code cells compile. Notebooks need a key, so nothing is executed here."""
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(HERE.glob("*.ipynb"))


def code_of(nb_path):
    nb = json.loads(nb_path.read_text())
    assert nb["nbformat"] == 4
    lines = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] or cell.get("execution_count"), "outputs only from a real run"
            src = "".join(cell["source"])
            lines.append("\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!"))))
    return "\n\n".join(lines)


def test_there_are_notebooks():
    assert NOTEBOOKS


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_compiles(nb):
    compile(code_of(nb), str(nb), "exec")
