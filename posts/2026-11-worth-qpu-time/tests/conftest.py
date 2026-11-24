"""Test setup: the post's modules and the shared kits on sys.path, and `canned` / `canned_compiler` fixtures with
fixed canned results in place of the API (no solving or compiling; see blog/tests/canned.py)."""
import os
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tests"))

POST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [POST_DIR, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "shared")]
os.environ.pop("FOURCOLORS_API_KEY", None)          # a test must never reach the API
os.environ.pop("FOURCOLORS_KEY_FILE", None)

import examplekit  # noqa: E402



@pytest.fixture
def canned(monkeypatch):
    """fourcolors.qubo.solve/submit return a FIXED canned result: all zeros, or the assignment a test sets on
    `canned.solution`. Nothing is solved."""
    import canned as _canned

    monkeypatch.setattr(examplekit, "client", lambda: None)
    return _canned.CannedQubo().install(monkeypatch)


@pytest.fixture
def canned_compiler(monkeypatch):
    """fourcolors.quantum.compiler.compile/submit return the input circuit unchanged. Nothing is compiled."""
    import canned as _canned

    return _canned.CannedCompiler().install(monkeypatch, examplekit)
