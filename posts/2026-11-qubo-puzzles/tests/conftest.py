"""Offline test setup: the post's modules and examplekit on sys.path, and a fake API in place of the real one."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1] / "shared"))  # examplekit
sys.path.insert(0, str(HERE))

import examplekit


@pytest.fixture
def fake(monkeypatch):
    """fourcolors.qubo.solve/submit replaced by examplekit.OfflineQubo; no key, no network."""
    monkeypatch.setattr(examplekit, "client", lambda: None)
    monkeypatch.delenv("FOURCOLORS_API_KEY", raising=False)
    monkeypatch.delenv("FOURCOLORS_KEY_FILE", raising=False)
    return examplekit.OfflineQubo().install(monkeypatch)
