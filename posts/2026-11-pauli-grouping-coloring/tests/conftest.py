"""Test setup: the post's modules and the shared kits on sys.path, and fixtures with fixed canned results in place
of the API (no solving or compiling; see blog/tests/canned.py)."""
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(__import__("pathlib").Path(__file__).resolve().parents[3] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tests")]
os.environ.pop("FOURCOLORS_API_KEY", None)          # a test must never reach the API
os.environ.pop("FOURCOLORS_KEY_FILE", None)


@pytest.fixture
def canned(monkeypatch):
    """fourcolors.qubo.solve returns a FIXED assignment the test sets on `canned.solution` (default all zeros), and
    fourcolors.quantum.compiler.compile returns the input circuit unchanged. Nothing is solved or compiled."""
    import examplekit
    import canned as _canned

    monkeypatch.setattr(examplekit, "client", lambda: None)
    _canned.CannedCompiler().install(monkeypatch, examplekit)
    return _canned.CannedQubo().install(monkeypatch)
