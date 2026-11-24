"""Offline tests of the report card's model code: generator, D2 pre-pass, selection rule, run and plot plumbing.
None of them calls the service (the `canned` fixture returns the input circuit unchanged)."""
import json

import numpy as np
import pytest

from conftest import load

F = load("families")
dq = load("dq")
report = load("report")


def test_29_families_three_sizes_each_fit_the_chip():
    assert len(F.FAMILIES) == 29
    assert len(F.plan()) == 29 * 3 * 2
    for name, fam in F.FAMILIES.items():
        assert len(fam.sizes) == 3 and fam.mechanism
    qc, _ = F.make("kicked_ising_hh", 127, 101)
    assert qc.num_qubits == 127
    qc, meta = F.make("surface", 7, 101)
    assert qc.num_qubits == 97 and meta["ancillas"] == 48


def test_generator_is_deterministic_and_hashes_the_text():
    a, _ = F.make("uccsd", 12, 101)
    b, _ = F.make("uccsd", 12, 101)
    c, _ = F.make("uccsd", 12, 102)
    assert F.instance_sha256(a) == F.instance_sha256(b)
    assert F.instance_sha256(a) != F.instance_sha256(c)          # amplitudes are seeded by the instance
    assert a.name == "uccsd-n12-i101"


@pytest.mark.parametrize("family,size", [("uccsd", 12), ("hubbard1d", 4), ("hubbard2d", 3)])
def test_reference_state_makes_the_circuit_act_on_zero(family, size):
    """The fix for the earlier generator: from |0...0> these circuits now do something."""
    from qiskit.quantum_info import Statevector

    qc, meta = F.make(family, size, 101, measure=False)
    psi = Statevector(qc)
    assert abs(psi.data[0]) ** 2 < 1e-6                           # not the vacuum any more
    assert meta["reference"] in ("hartree_fock", "neel")


def test_uccsd_reference_fills_the_occupied_orbitals():
    qc, _ = F.FAMILIES["uccsd"].build(12, 101)
    xs = [qc.find_bit(i.qubits[0]).index for i in qc.data[:4]]
    assert [i.operation.name for i in qc.data[:4]] == ["x"] * 4 and xs == [0, 1, 2, 3]


def test_grid_is_open():
    e = F.grid_edges(3)
    assert len(e) == 12 and (0, 2) not in e and (0, 6) not in e


def test_dq_exact_for_qiskit_on_a_small_circuit_and_detects_a_wrong_one():
    from qiskit import QuantumCircuit, transpile
    import circuitkit as ck

    sets, _ = ck.heavy_hex_coupler_sets()
    qc = QuantumCircuit(4)
    qc.h(0); qc.cx(0, 3); qc.cx(1, 3); qc.rz(0.3, 3); qc.cx(2, 0)
    out = transpile(qc, target=sets["full"], optimization_level=3, seed_transpiler=5)
    f = dq.fidelities(qc, out)
    assert f is not None and min(f) > 1 - 1e-9
    wrong = out.copy()
    wrong.rx(0.4, wrong.layout.initial_index_layout(filter_ancillas=True)[1])
    assert min(dq.fidelities(qc, wrong)) < 0.999
    row = dq.measure_dq(qc, sets["full"], qiskit_seed=5)
    assert row["dq"] == 0.0 and row["dq_states"] == 8


def test_dq_unmeasured_with_resets_and_options():
    import circuitkit as ck

    sets, _ = ck.heavy_hex_coupler_sets()
    qc, _ = F.make("repcode", 5, 101)
    assert dq.measure_dq(qc, sets["full"], qiskit_seed=1)["dq"] is None
    assert dq.ours_options(None) == {"approximation_mode": "exact"}
    assert dq.ours_options(0.0) == {"approximation_mode": "exact"}
    assert dq.ours_options(0.3) == {"max_approximation_error": 0.1}


def _rows(family, ours, best, once, *, over=False, refused=False, q=16):
    rows = [{"kind": "circuit", "circuit_id": f"{family}-i{i}", "family": family, "qubits": q, "domain": "d",
             "sha256": f"h{i}"} for i in range(2)]
    for i in range(2):
        for seed in (1, 2, 3):
            base = {"kind": "result", "circuit_id": f"{family}-i{i}", "family": family, "qubits": q, "seed": seed,
                    "budget_s": 10.0, "input_2q": 50}
            rows.append({**base, "arm": "4colors", "two_qubit_gates": None if refused else ours, "two_qubit_depth": 5,
                         "refusal_code": "input_too_large" if refused else None, "verification_label": "unverified"})
            rows.append({**base, "arm": "qiskit-o3-best", "two_qubit_gates": best, "two_qubit_depth": 9,
                         "over_budget": over})
            rows.append({**base, "arm": "qiskit-o3-once", "two_qubit_gates": once, "two_qubit_depth": 9})
    return rows


def test_selection_shows_only_strict_wins_against_the_lower_qiskit_arm():
    rows = (_rows("win", 70, 100, 120) + _rows("tie", 99, 100, 120) + _rows("loss", 130, 100, 120)
            + _rows("onceworse", 95, 120, 100) + _rows("refused", 10, 100, 100, refused=True)
            + _rows("over", 10, 100, 100, over=True))
    by = report.cells(rows)
    verdicts = {k[0]: c["verdict"] for k, c in by.items()}
    assert verdicts == {"win": "win", "tie": "tie", "loss": "loss", "onceworse": "win", "refused": "no answer",
                        "over": "apart: Qiskit overran T"}
    assert [k[0] for k in report.shown(by)] == ["win", "onceworse"]
    assert by[("onceworse", 16)]["median_rival"] == 100                   # the lower of the two Qiskit arms
    assert "Cells shown" not in report.caption(by, 6)   # owner 23 Nov: no tallies in the caption
    assert "we ran 6 families; 2 are shown" in report.internal_counts(by, 6)
    table = report.card_table(by)
    assert "tie" not in table and "loss" not in table and "| win (d)" in table


def test_run_and_plot_offline(canned, tmp_path):
    run = load("run")
    plot = load("plot")
    out = tmp_path / "card"
    run.main(["--out", str(out), "--families", "tfim2d,ghz", "--sizes-index", "0", "--instances", "101",
              "--seeds", "1", "--budgets", "0.2", "--max-qiskit-trials", "1", "--qiskit-streams", "1"])
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    meta = rows[0]
    assert meta["kind"] == "meta" and meta["generator_version"] == F.GENERATOR_VERSION
    circ = [r for r in rows if r["kind"] == "circuit"]
    assert len(circ) == 2 and all(len(r["sha256"]) == 64 for r in circ)
    assert {r["arm"] for r in rows if r["kind"] == "result"} == {"4colors", "qiskit-o3-best", "qiskit-o3-once"}
    ours = [r for r in rows if r.get("arm") == "4colors"]
    assert all(r["job_id"].startswith("canned") and r["options"] == {"approximation_mode": "exact", "verify": "wait"}
               for r in ours)
    assert len(canned.calls) == 2 and canned.calls[0]["time_budget_s"] == 0.2
    plot.main([str(out), "--figs", str(tmp_path / "figs")])
    sel = (tmp_path / "figs" / "selection.md").read_text()
    assert "we ran 2 families" in sel
