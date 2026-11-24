from pathlib import Path

from fourcolors.quantum import compiler

from canned import CannedCompiler, canned_compile, gate_count, run_script

SCRIPT = Path(__file__).resolve().parents[1] / "custom_target.py"


def run(monkeypatch):
    canned = CannedCompiler([canned_compile(900, 300), canned_compile(800, 280)]).install(monkeypatch)
    return canned, run_script(SCRIPT, monkeypatch)


def test_qaoa_on_a_three_regular_graph(monkeypatch):
    _, g = run(monkeypatch)
    edges = g["edges"]
    degree = [0] * 100
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    assert len(edges) == 150 and set(degree) == {3}
    assert gate_count(g["qaoa"], "cx") == 2 * 2 * 150      # two layers, two cx per edge
    assert gate_count(g["qaoa"], "rx") == 2 * 100


def test_grid_description_is_accepted_by_the_client(monkeypatch):
    _, g = run(monkeypatch)
    grid = g["grid"]
    assert grid["n_qubits"] == 144 and len(grid["couplers"]) == 2 * 12 * 11
    # The client's own check of a target description, run locally before anything is sent.
    opt = compiler.target_option(grid)
    assert isinstance(opt, dict) and opt


def test_compiles_for_both_targets(monkeypatch, capsys):
    canned, g = run(monkeypatch)
    out = capsys.readouterr().out.splitlines()
    assert [c["time_budget_s"] for c in canned.calls] == [120, 120]
    assert canned.calls[0]["target"].description["n_qubits"] == 156
    assert canned.calls[1]["target"] is g["grid"]
    assert out == ["heavy-hex 156 900 300 verified", "12x12 grid 800 280 verified"]
