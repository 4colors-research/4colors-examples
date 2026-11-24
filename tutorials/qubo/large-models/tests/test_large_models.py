from pathlib import Path

import pytest

from canned import CannedJob, CannedQubo, canned_result, run_script

SCRIPT = Path(__file__).resolve().parents[1] / "large_models.py"
N = 50_000
MODEL_ID = "ab" * 32


@pytest.fixture
def no_wait(monkeypatch):
    """The script's polling loop sleeps 2 s per round; the tests skip the sleeping."""
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)


def test_upload_once_then_solve_by_id_and_stop_when_done(monkeypatch, capsys, no_wait):
    final = canned_result([0] * N, -12, billed_s=1.5)
    job = CannedJob(final, statuses=["queued", "running", "running", "finished"],
                    bests=[None, canned_result([0] * N, -10, status="running")])
    canned = CannedQubo([job], model_id=MODEL_ID).install(monkeypatch)
    g = run_script(SCRIPT, monkeypatch)
    out = capsys.readouterr().out.splitlines()
    assert len(g["edges"]) == 150_000 and len(g["linear"]) == N
    assert [c["call"] for c in canned.calls] == ["upload", "upload", "submit"]
    assert canned.calls[2]["input_id"] == MODEL_ID and canned.calls[2]["time_limit_s"] == 120
    assert out[0] == MODEL_ID and out[1] == "True"
    assert "best cut so far 10" in out
    assert out[-1] == "finished 12 1.5"


def test_stops_after_twenty_seconds_without_improvement(monkeypatch, capsys, no_wait):
    import time
    clock = iter(range(0, 1000, 5))                        # each read of the clock is 5 s later
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    final = canned_result([0] * N, -10, status="stopped")
    same = canned_result([0] * N, -10, status="running")
    job = CannedJob(final, statuses=["running"], bests=[same])
    CannedQubo([job], model_id=MODEL_ID).install(monkeypatch)
    run_script(SCRIPT, monkeypatch)
    out = capsys.readouterr().out.splitlines()
    assert job.stopped and out.count("best cut so far 10") == 1
    assert out[-1].startswith("stopped 10")
