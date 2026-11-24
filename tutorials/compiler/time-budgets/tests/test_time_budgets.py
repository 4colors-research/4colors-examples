from pathlib import Path

from canned import CannedCompileJob, CannedCompiler, canned_compile, gate_count, run_script

SCRIPT = Path(__file__).resolve().parents[1] / "time_budgets.py"


def test_default_budget_then_twenty_seconds_streamed(monkeypatch, capsys):
    canned = CannedCompiler([canned_compile(700, 400, billed_s=3.2),
                             CannedCompileJob(canned_compile(600, 350, billed_s=20.0),
                                              incumbents=[canned_compile(650, 380)])]).install(monkeypatch)
    g = run_script(SCRIPT, monkeypatch)
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "156 cz"
    assert [(c["call"], c["time_budget_s"]) for c in canned.calls] == [("compile", None), ("submit", 20)]
    assert out[1] == "700 400 3.2"
    assert out[2:] == ["two-qubit gates 650, depth 380", "two-qubit gates 600, depth 350", "600 20.0"]
    assert gate_count(g["circuit"], "cx") == 250 and gate_count(g["circuit"], "rz") == 250
    assert canned.calls[0]["circuit"] is g["circuit"]
