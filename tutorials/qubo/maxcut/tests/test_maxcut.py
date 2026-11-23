import itertools
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[2] / "shared"))
sys.path.insert(0, str(HERE))

import examplekit  # noqa: E402
import maxcut  # noqa: E402
from canned import CannedJob, CannedQubo, canned_result  # noqa: E402


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
def canned(monkeypatch):
    """canned(job): fourcolors.qubo.submit returns this fixed job (tests/canned.py); no key, no network."""
    monkeypatch.setattr(examplekit, "client", lambda: None)
    return lambda *responses: CannedQubo(responses).install(monkeypatch)


# Two fixed answers for the 12-node, 24-edge graph (seed 7), written by hand: cuts of 9 and then 10 edges.
FIRST, FINAL = [0, 1] * 6, [0, 0, 1] * 4


def test_script_streams_checks_and_prints_the_answer(canned, capsys):
    final = canned_result(FINAL, -10, trajectory=[{"t_search_s": 0.2, "objective": -9},
                                                  {"t_search_s": 0.6, "objective": -10}])
    canned(CannedJob(final, incumbents=[canned_result(FIRST, -9, status="running")]))
    result, edges = maxcut.main(["--n", "12", "--m", "24", "--time-limit", "1"])
    assert maxcut.cut_size(edges, FIRST) == 9 and maxcut.cut_size(edges, result.solution) == 10
    out = capsys.readouterr().out
    assert "cut 9 of 24 edges" in out and "cut 10 of 24 edges" in out
    assert "finished: 10 edges cross between groups of 8 and 4 nodes" in out


def test_script_rejects_an_objective_that_disagrees_with_its_own_count(canned):
    canned(canned_result(FINAL, -11))
    with pytest.raises(AssertionError, match="agree"):
        maxcut.main(["--n", "12", "--m", "24", "--time-limit", "1"])
