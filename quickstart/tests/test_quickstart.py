"""The quickstarts need a key and the service, so their tests check only that each file is valid."""
import py_compile
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["qubo_hello.py", "compiler_hello.py"])
def test_python_quickstarts_compile(script, tmp_path):
    py_compile.compile(str(HERE / script), cfile=str(tmp_path / "out.pyc"), doraise=True)


def test_http_hello_is_valid_bash():
    assert subprocess.run(["bash", "-n", str(HERE / "http_hello.sh")]).returncode == 0
