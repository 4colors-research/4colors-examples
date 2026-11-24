"""Offline tests of the adder post: the adders add, the check catches a wrong circuit, the selection rule and the
plumbing. No call reaches the service."""
import itertools
import json

import pytest

from conftest import load

adders = load("adders")


@pytest.mark.parametrize("kind,n", [("cuccaro", 3), ("vbe", 3)])
def test_adders_add_every_input(kind, n):
    """Exhaustive on 3 bits: the lowered block maps every basis input to the classical sum."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    regs = adders.registers(kind, n)
    width = sum(len(v) for v in regs.values())
    qc = adders.circuit(kind, n)
    for a, b, cin in itertools.product(range(1 << n), range(1 << n), (0, 1)):
        prep = QuantumCircuit(width)
        for name, v, k in (("a", a, n), ("b", b, n), ("cin", cin, 1)):
            for i, q in enumerate(regs[name]):
                if (v >> i) & 1:
                    prep.x(q)
        out = Statevector(prep.compose(qc)).probabilities_dict()
        (key, p), = [(k, p) for k, p in out.items() if p > 1e-9]
        bits = [int(x) for x in reversed(key)]
        got = {name: sum(bits[q] << i for i, q in enumerate(idx)) for name, idx in regs.items()}
        want = adders.expected(kind, n, a, b, cin)
        assert got == {k: want[k] for k in regs}


def test_sizes_reach_100_qubits():
    for kind, sizes in adders.SIZES.items():
        assert adders.raw(kind, max(sizes)).num_qubits == 100


def test_check_passes_qiskit_output_and_catches_a_wrong_one():
    from qiskit import transpile
    import circuitkit as ck

    sets, _ = ck.heavy_hex_coupler_sets()
    qc = adders.circuit("cuccaro", 4)
    out = transpile(qc, target=sets["full"], optimization_level=3, seed_transpiler=3)
    assert adders.check(out, "cuccaro", 4, trials=4)["adder_check"] == "returns_sum"
    bad = out.copy()
    bad.x(out.layout.final_index_layout()[adders.registers("cuccaro", 4)["b"][0]])
    r = adders.check(bad, "cuccaro", 4, trials=4)
    assert r["adder_check"] == "wrong" and r["adder_check_failures"]


def test_selection_shows_only_checked_strict_wins():
    plot = load("plot")

    def rows(ours, rival, check="returns_sum"):
        base = {"kind": "result", "family": "cuccaro", "bits": 49, "budget_s": 10.0, "seed": 1, "qubits": 100,
                "two_qubit_depth": 5, "toffoli_counted_input": 600}
        return [{**base, "arm": "4colors", "two_qubit_gates": ours, "adder_check": check},
                {**base, "arm": "qiskit-o3-best", "two_qubit_gates": rival, "over_budget": False},
                {**base, "arm": "qiskit-o3-once", "two_qubit_gates": rival + 10}]
    assert plot.cells(rows(800, 1000))[("cuccaro", 49, 10.0)]["shown"]
    assert not plot.cells(rows(995, 1000))[("cuccaro", 49, 10.0)]["shown"]
    assert not plot.cells(rows(800, 1000, check="wrong"))[("cuccaro", 49, 10.0)]["shown"]


def test_run_and_plot_offline(canned, tmp_path):
    run, plot = load("run"), load("plot")
    out = tmp_path / "a"
    run.main(["--out", str(out), "--kinds", "cuccaro", "--bits", "3", "--seeds", "1", "--budgets", "0.2",
              "--largest-budget", "0", "--max-qiskit-trials", "1", "--qiskit-streams", "1", "--check-trials", "2"])
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    res = [r for r in rows if r["kind"] == "result"]
    assert {r["arm"] for r in res} == {"4colors", "qiskit-o3-best", "qiskit-o3-once"}
    assert all(r["adder_check"] == "returns_sum" for r in res)       # canned returns the input: it adds
    circ = next(r for r in rows if r["kind"] == "circuit")
    assert circ["toffoli_counted_input"] == circ["input_2q"] + 6 * circ["toffolis"]
    plot.main([str(out), "--figs", str(tmp_path / "figs"), "--budget", "0.2"])
    assert "Internal (not for publication): cells shown" in (tmp_path / "figs" / "selection.md").read_text()
    assert "Cells shown" not in plot.caption({})   # owner 23 Nov: no tallies in the caption
