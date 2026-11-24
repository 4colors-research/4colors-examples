"""Test setup: the post's modules and the shared kits on sys.path, and a `canned` fixture with fixed canned
results in place of the API (no solving or compiling; see blog/tests/canned.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tests"))

DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DIR), str(__import__("pathlib").Path(__file__).resolve().parents[3] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[3] / "shared")]

@pytest.fixture
def canned(monkeypatch):
    """Canned results only (tests/canned.py in the blog folder): the compiler returns the input circuit
    unchanged and the QUBO call returns all zeros. Nothing is compiled or solved."""
    import examplekit
    import canned as _canned

    c = _canned.CannedCompiler().install(monkeypatch, examplekit)
    q = _canned.CannedQubo().install(monkeypatch)
    monkeypatch.setenv("FOURCOLORS_API_KEY", "")
    yield type("Canned", (), {"compiler": c, "qubo": q})
