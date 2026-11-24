import random
from pathlib import Path

import pytest

from canned import CannedJob, CannedQubo, canned_result, run_script

SCRIPT = Path(__file__).resolve().parents[1] / "maxcut.py"
N = 2000


@pytest.fixture
def canned(monkeypatch):
    """canned(job): fourcolors.qubo.submit returns this fixed job (tests/canned.py); no key, no network."""
    return lambda *responses: CannedQubo(responses).install(monkeypatch)


# Hand-written answers: everything in one group (0 edges cut), and nodes split by parity.
ALL_ZERO = [0] * N
PARITY = [i % 2 for i in range(N)]


def test_script_streams_checks_and_prints_the_answer(canned, monkeypatch, capsys):
    final = canned_result(ALL_ZERO, 0, trajectory=[{"t_search_s": 0.4, "objective": 0}])
    canned(CannedJob(final, incumbents=[canned_result(ALL_ZERO, 0, status="running")]))
    g = run_script(SCRIPT, monkeypatch)
    out = capsys.readouterr().out
    assert len(g["edges"]) == 6000 and all(i < j for i, j in g["edges"])
    assert "cut 0 of 6000 edges" in out and "finished 0" in out
    assert "0 edges cross between groups of 2000 and 0 nodes" in out
    assert "t=0.40 s  objective 0" in out


def test_qubo_energy_is_minus_the_cut(canned, monkeypatch):
    canned(canned_result(ALL_ZERO, 0))
    g = run_script(SCRIPT, monkeypatch)
    Q, linear, edges = g["Q"], g["linear"], g["edges"]
    rnd = random.Random(0)
    for x in [PARITY] + [[rnd.randrange(2) for _ in range(N)] for _ in range(5)]:
        energy = sum(v * x[i] * x[j] for (i, j), v in Q.items()) + sum(l * xi for l, xi in zip(linear, x))
        assert energy == -sum(1 for i, j in edges if x[i] != x[j])


def test_script_rejects_an_objective_that_disagrees_with_its_own_count(canned, monkeypatch):
    canned(canned_result(ALL_ZERO, -1))
    with pytest.raises(AssertionError, match="agree"):
        run_script(SCRIPT, monkeypatch)


def test_no_key_exits_with_the_trial_hint(monkeypatch):
    monkeypatch.delenv("FOURCOLORS_API_KEY", raising=False)
    monkeypatch.delenv("FOURCOLORS_KEY_FILE", raising=False)
    import runpy
    with pytest.raises(SystemExit, match="free trial"):
        runpy.run_path(str(SCRIPT), run_name="__main__")


def test_key_file_is_read(monkeypatch, tmp_path, canned):
    key = tmp_path / "k"
    key.write_text("from-file\n")
    monkeypatch.setenv("FOURCOLORS_KEY_FILE", str(key))
    monkeypatch.delenv("FOURCOLORS_API_KEY", raising=False)
    canned(canned_result(ALL_ZERO, 0))
    import os
    import runpy
    runpy.run_path(str(SCRIPT), run_name="__main__")
    assert os.environ["FOURCOLORS_API_KEY"] == "from-file"
