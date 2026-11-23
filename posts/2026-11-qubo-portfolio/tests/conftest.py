"""Test setup: the post's modules and examplekit on sys.path, and fixed canned client responses (tests/canned.py)."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1] / "shared"))  # examplekit
sys.path.insert(0, str(HERE))

import examplekit
from canned import CannedQubo


@pytest.fixture
def canned(monkeypatch):
    """canned(r1, r2, ...): fourcolors.qubo.solve returns these fixed results in call order; no key, no network."""
    monkeypatch.setattr(examplekit, "client", lambda: None)
    monkeypatch.delenv("FOURCOLORS_API_KEY", raising=False)
    monkeypatch.delenv("FOURCOLORS_KEY_FILE", raising=False)
    return lambda *responses: CannedQubo(responses).install(monkeypatch)
