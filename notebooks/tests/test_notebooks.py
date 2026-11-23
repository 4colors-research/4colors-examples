"""Every notebook is valid JSON, its code cells compile, and it runs top to bottom offline (no key, no network)."""
import json
import os
import subprocess
import sys
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


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_compiles(nb):
    compile(code_of(nb), str(nb), "exec")


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs_offline(nb, tmp_path):
    script = tmp_path / "nb.py"
    script.write_text(code_of(nb))
    env = {k: v for k, v in os.environ.items() if not k.startswith("FOURCOLORS_")}
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "offline" in r.stdout
