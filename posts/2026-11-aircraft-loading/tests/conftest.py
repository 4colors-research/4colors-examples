"""Test setup: the post's modules and the shared kits on sys.path, and a `canned` fixture with fixed canned
results in place of the API (no solving or compiling; see blog/tests/canned.py)."""
import os
import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tests"))

POST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [POST_DIR, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "shared")]
os.environ.pop("FOURCOLORS_API_KEY", None)          # a test must never reach the API
os.environ.pop("FOURCOLORS_KEY_FILE", None)


import postload  # noqa: E402  (blog/tests/postload.py; blog/tests is on sys.path from conftest)


def load(name):
    """Import <post dir>/<name>.py under a name unique to this post (blog/tests/postload.py), so posts with a
    run.py or plot.py of their own never collide when their tests run in one session."""
    return postload.load(POST_DIR, name)

@pytest.fixture
def canned(monkeypatch):
    """fourcolors.quantum.compiler.compile/submit return the input circuit unchanged as a canned result
    (tests/canned.py in the blog folder). Nothing is compiled."""
    import examplekit
    import canned as _canned

    return _canned.CannedCompiler().install(monkeypatch, examplekit)


@pytest.fixture
def canned_qubo(monkeypatch):
    """fourcolors.qubo.solve/submit return a FIXED canned assignment (all zeros unless a test sets
    `canned_qubo.solution`); examplekit.client is stubbed. Nothing is solved."""
    import examplekit
    import canned as _canned

    monkeypatch.setattr(examplekit, "client", lambda: None)
    return _canned.CannedQubo().install(monkeypatch)
