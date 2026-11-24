from pathlib import Path

from canned import CannedJob, CannedQubo, canned_result, run_script

SCRIPT = Path(__file__).resolve().parents[1] / "quickstart.py"


def test_solve_then_watch(monkeypatch, capsys):
    # Hand-written answers for the 4-variable model: x = (1, 0, 0, 1) has energy -5.
    first = canned_result([1, 0, 0, 1], -5, billed_s=1.0)
    better = canned_result([1, 0, 0, 1], -5, status="running")
    canned = CannedQubo([first, CannedJob(canned_result([1, 0, 0, 1], -5), incumbents=[better],
                                          statuses=["running"])]).install(monkeypatch)
    g = run_script(SCRIPT, monkeypatch)
    out = capsys.readouterr().out.splitlines()
    assert [c["call"] for c in canned.calls] == ["solve", "submit"]
    assert [c["time_limit_s"] for c in canned.calls] == [1, 60]
    assert out[:4] == ["finished", "-5", "[1, 0, 0, 1]", "1.0"]
    assert out[4] == "running" and out[5:] == ["-5", "-5"]
    assert g["result"].status == "finished"


def test_the_model_is_the_one_the_docs_describe(monkeypatch):
    CannedQubo([canned_result([0] * 4, 0), canned_result([0] * 4, 0)]).install(monkeypatch)
    g = run_script(SCRIPT, monkeypatch)
    Q, linear = g["Q"], g["linear"]

    def energy(x):
        return sum(v * x[i] * x[j] for (i, j), v in Q.items()) + sum(l * xi for l, xi in zip(linear, x))
    assert energy([1, 0, 0, 1]) == -5 and energy([1, 0, 1, 0]) == -2
    assert energy([1, 1, 1, 1]) == -1 and energy([0, 0, 0, 0]) == 0
