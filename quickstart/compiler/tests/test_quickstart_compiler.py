from pathlib import Path

from canned import CannedCompileJob, CannedCompiler, canned_compile, gate_count, run_script

SCRIPT = Path(__file__).resolve().parents[1] / "quickstart.py"


def test_compile_then_watch(monkeypatch, capsys):
    final = canned_compile(4, 6, layout=[10, 11, 12, 13, 14])
    canned = CannedCompiler([final, CannedCompileJob(canned_compile(4, 5), incumbents=[canned_compile(5, 7)])])
    canned.install(monkeypatch)
    g = run_script(SCRIPT, monkeypatch)
    out = capsys.readouterr().out.splitlines()
    assert [c["call"] for c in canned.calls] == ["compile", "submit"]
    assert [c["time_budget_s"] for c in canned.calls] == [10, 60]
    assert out[:3] == ["4 6", "[10, 11, 12, 13, 14]", "verified"]
    assert out[-2:] == ["5", "4"]
    assert gate_count(g["ghz"], "cx") == 4 and gate_count(g["ghz"], "h") == 1


def test_the_target_is_a_description_not_a_name(monkeypatch):
    canned = CannedCompiler([canned_compile(4, 6), canned_compile(4, 6)]).install(monkeypatch)
    run_script(SCRIPT, monkeypatch)
    target = canned.calls[0]["target"]
    assert not isinstance(target, str)
    assert target.description["n_qubits"] == 156 and target.description["two_qubit_gate"] == "cz"
