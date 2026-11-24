"""Test setup: the post's modules and the shared kits on sys.path, and a `canned` fixture with fixed canned
results in place of the API (no solving or compiling; see blog/tests/canned.py)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tests"))

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "shared"))
sys.path.insert(0, str(HERE))

import examplekit

@pytest.fixture
def canned(monkeypatch):
    """fourcolors.qubo.solve/submit return a FIXED canned result (tests/canned.py in the blog folder): all zeros,
    the warm start sent back, or the assignment a test sets on `canned.solution`. Nothing is solved."""
    import canned as _canned

    monkeypatch.setattr(examplekit, "client", lambda: None)
    monkeypatch.delenv("FOURCOLORS_API_KEY", raising=False)
    monkeypatch.delenv("FOURCOLORS_KEY_FILE", raising=False)
    return _canned.CannedQubo().install(monkeypatch)
