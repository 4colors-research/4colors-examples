import itertools
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[2] / "shared"))
sys.path.insert(0, str(HERE))

import examplekit  # noqa: E402
import maxcut  # noqa: E402


def test_qubo_energy_is_minus_the_cut():
    n, edges = 6, maxcut.random_graph(6, 9, seed=1)
    Q, linear = maxcut.maxcut_qubo(n, edges)
    for x in itertools.product((0, 1), repeat=n):
        e = sum(v * x[i] * x[j] for (i, j), v in Q.items()) + sum(l * xi for l, xi in zip(linear, x))
        assert e == -maxcut.cut_size(edges, x)


def test_random_graph_is_simple_and_seeded():
    g = maxcut.random_graph(50, 120, seed=3)
    assert len(g) == len(set(g)) == 120 and all(i < j for i, j in g)
    assert g == maxcut.random_graph(50, 120, seed=3)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(examplekit, "client", lambda: None)
    return examplekit.OfflineQubo().install(monkeypatch)


def test_script_end_to_end_offline(fake, capsys):
    result, edges = maxcut.main(["--n", "12", "--m", "24", "--time-limit", "1"])
    best = max(maxcut.cut_size(edges, x) for x in itertools.product((0, 1), repeat=12))
    assert maxcut.cut_size(edges, result.solution) == best          # the stand-in enumerates 12 nodes exactly
    assert "edges cross between groups" in capsys.readouterr().out
