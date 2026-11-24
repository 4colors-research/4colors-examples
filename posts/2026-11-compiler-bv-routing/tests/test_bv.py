"""Offline tests for the BV post: the circuit, the metric function, the device sets, and run.py + plot.py
end to end on canned results (the input circuit returned unchanged; no API call)."""
import json

import pytest
from conftest import load

import circuitkit as ck


@pytest.mark.parametrize("secret", ["1", "101101", "0000011", "1111111111"])
def test_bv_returns_the_secret(secret):
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    qc = ck.bernstein_vazirani(secret)
    sim = AerSimulator(method="statevector", seed_simulator=1)
    counts = sim.run(transpile(qc, sim, seed_transpiler=1), shots=64).result().get_counts()
    assert counts == {secret[::-1]: 64}          # noiseless: the secret every shot (Qiskit prints bit 0 last)


def test_bv_interaction_graph_is_a_star():
    qc = ck.bernstein_vazirani("1101")
    edges = ck.interaction_edges(qc)
    assert edges == [[0, 4, 1], [1, 4, 1], [3, 4, 1]]       # one CNOT per 1 bit, all into the ancilla
    assert ck.circuit_metrics(qc)["two_qubit_gates"] == 3


def test_metric_agrees_with_qiskit_counts():
    from qiskit import transpile

    sets, failed = ck.heavy_hex_coupler_sets()
    qc = ck.bernstein_vazirani("1" * 8)
    out = transpile(qc, target=sets["full"], optimization_level=1, seed_transpiler=3)
    m = ck.circuit_metrics(out, failed)
    assert m["two_qubit_gates"] == out.count_ops().get("cz", 0)
    assert m["two_qubit_depth"] == out.depth(lambda i: i.operation.num_qubits == 2 and i.operation.name != "barrier")
    used = {tuple(sorted(out.find_bit(q).index for q in i.qubits)) for i in out.data if i.operation.name == "cz"}
    assert m["gates_on_failed_couplers"] == 0 or used & failed
    # a circuit that sits on a failed coupler is counted
    from qiskit import QuantumCircuit
    bad = QuantumCircuit(156)
    a, b = sorted(failed)[0]
    bad.cz(a, b)
    bad.cz(0, 1)
    assert ck.circuit_metrics(bad, failed)["gates_on_failed_couplers"] == 1


def test_the_4colors_target_is_fakefez_with_its_calibration():
    """No named targets: the 4colors arm sends FakeFez's description and published calibration (from_qiskit),
    the same device Qiskit's "full" arm gets; "working" is that Target without the failed couplers."""
    sets, failed = ck.heavy_hex_coupler_sets()
    t = ck.heavy_hex_target()
    assert sorted(tuple(sorted(c)) for c in t.description["couplers"]) == ck.device_couplers(sets["full"])
    assert {tuple(sorted(c["qubits"])) for c in t.calibration["couplers"] if c["error"] >= 0.999} == failed
    assert set(ck.device_couplers(sets["working"])) == set(ck.device_couplers(sets["full"])) - failed
    assert sets["working"]["sx"] == sets["full"]["sx"]         # errors kept on everything else


def test_metric_on_a_real_compile_result():
    """`circuit_metrics(res.to_qiskit())` on a CompileResult built from a service record (no API)."""
    import fourcolors
    from fourcolors.client import JobInfo

    text = ("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nbit _bit0;\nbit _bit1;\n"
            "sx $2;\ncz $2, $3;\nsx $3;\ncz $3, $4;\n_bit0 = measure $2;\n_bit1 = measure $3;\n")
    record = {"status": "finished", "circuit": {"format": "qasm3", "text": text},
              "metrics": {"two_qubit_gates": 2}, "mapping": {"initial_layout": [2, 3], "final_layout": [2, 3]},
              "verification": {"label": "skipped"}, "target": {"kind": "custom", "n_qubits": 156},
              "engine": {"version": "1.0.0"}, "seed": 1}
    res = fourcolors.quantum.compiler.from_job(JobInfo(id="j1", status="done", result=record))
    fields = ck.result_fields(res, failed=[(3, 4)])
    assert fields["two_qubit_gates"] == 2 == fields["service_metrics"]["two_qubit_gates"]
    assert fields["gates_on_failed_couplers"] == 1
    assert ck.coupler_counts(res.to_qiskit()) == [[2, 3, 1], [3, 4, 1]]
    assert fields["engine_version"] == "1.0.0" and fields["job_id"] == "j1"


def test_run_and_plot_offline(canned, tmp_path):
    run, plot = load("run"), load("plot")
    run.main(["--out", str(tmp_path), "--sizes", "6,9", "--budgets", "0.01", "--seeds", "1",
              "--max-qiskit-trials", "1"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    kinds = {r["kind"] for r in rows}
    assert {"meta", "device", "interaction", "trial", "result"} <= kinds
    res = ck.results(rows)
    arms = {(r["arm"], r["coupler_set"]) for r in res}
    assert arms == {("4colors", ck.HEAVY_HEX_DEVICE), ("qiskit", "working"), ("qiskit", "full"), ("qiskit", "lower_of_both")}
    for r in res:
        assert set(ck.ROW_KEYS) <= set(r)
        secret = next(x["secret"] for x in rows if x["kind"] == "interaction" and x["qubits"] == r["qubits"])
        assert r["input_2q"] == secret.count("1") and r["two_qubit_gates"] >= r["input_2q"]
    four = ck.results(rows, arm="4colors")
    assert all(r["coupler_counts"] and r["job_id"].startswith("canned") for r in four)
    assert all(c["target"] is ck.heavy_hex_target() and c["verify"] == "skip" for c in canned.calls)
    assert all(c["approximation_mode"] == "exact" for c in canned.calls)          # sent explicitly
    # the default is the random secret, recorded; every compiled circuit (both arms) returns it
    stars = [r for r in rows if r["kind"] == "interaction"]
    assert all(r["circuit_id"].endswith("_random") and len(r["secret"]) == r["qubits"] - 1 for r in stars)
    assert all(r["clifford_check"] == "returns_secret" for r in res), [r.get("clifford_check_reason") for r in res]
    assert all(sum(r["block_counts"]["two_qubit_gates"].values()) <= r["two_qubit_gates"] for r in res)
    plot.main([str(tmp_path), "--figs", str(tmp_path / "figs")])
    assert {p.name for p in (tmp_path / "figs").iterdir()} == {"fig_graphs.png", "fig_counts.png", "fig_where.png",
                                                               "selection.md"}
    # with Qiskit's primary arm made worse, every cell is a win: shown, with its block table
    worse = [{**r, "two_qubit_gates": r["two_qubit_gates"] + 1000}
             if r.get("kind") == "result" and r.get("coupler_set") == "full" else r for r in rows]
    (tmp_path / "worse.jsonl").write_text("".join(json.dumps(r) + "\n" for r in worse))
    plot.main([str(tmp_path / "worse.jsonl"), "--figs", str(tmp_path / "figs2")])
    sel = (tmp_path / "figs2" / "selection.md").read_text()
    assert "0.01 s: n = 6, 9" in sel and "| 9 | 0.01 | 4colors |" in sel


def test_clifford_check_paths_and_a_wrong_secret():
    """The tableau path on a gate-by-gate Clifford circuit, the MPS path on Qiskit's re-synthesized output, and a
    failure (never a skip) for a wrong secret or a non-Clifford circuit."""
    from qiskit import transpile

    bv_check = load("bv_check")
    secret = ck.random_secret(40, seed=40)
    qc = ck.bernstein_vazirani(secret)
    r = bv_check.returns_secret(qc, secret)
    assert r["ok"] and r["method"] == "clifford_tableau"
    sets, _ = ck.heavy_hex_coupler_sets()
    out = transpile(qc, target=sets["full"], optimization_level=3, seed_transpiler=2)
    r = bv_check.returns_secret(out, secret)
    assert r["ok"] and r["method"] in ("clifford_tableau", "mps_64_shots")
    flipped = secret[:-1] + ("0" if secret[-1] == "1" else "1")
    assert not bv_check.returns_secret(out, flipped)["ok"]
    assert not bv_check.returns_secret(qc, flipped)["ok"]
    qc2 = ck.bernstein_vazirani(secret, measure=False)
    qc2.h(0); qc2.t(0); qc2.h(0)                                 # no longer Clifford, and no longer BV
    qc2.add_register(__import__("qiskit").ClassicalRegister(len(secret)))
    qc2.measure(range(len(secret)), range(len(secret)))
    assert bv_check.check_fields(qc2, secret)["clifford_check"] == "wrong"


def test_selection_rule_shows_only_strict_wins():
    plot = load("plot")

    def res(arm, cset, n, b, g, check="returns_secret"):
        return {"kind": "result", "arm": arm, "coupler_set": cset, "qubits": n + 1, "budget_s": b, "seed": 1,
                "two_qubit_gates": g, "clifford_check": check if arm == "4colors" else None}

    rows = [res("4colors", "fakefez", 50, 10, 80), res("qiskit", "full", 50, 10, 120),     # win: shown
            res("4colors", "fakefez", 75, 10, 130), res("qiskit", "full", 75, 10, 130),    # tie: left out
            res("4colors", "fakefez", 100, 10, 150, "wrong"), res("qiskit", "full", 100, 10, 300),  # failed check
            res("qiskit", "working", 50, 10, 60)]                                           # extra arm: ignored
    shown, left = plot.select_cells(rows)
    assert shown == {10: [50]}
    assert {(n, why) for n, _, why in left} == {(75, "tie or behind"), (100, "a 4colors circuit failed the Clifford check")}


def test_block_classes():
    """A CNOT then a SWAP is one 2-CZ-class block ("cx_move"); a SWAP alone is a "swap"; a CNOT is a "cx"."""
    from qiskit import QuantumCircuit, transpile

    blocks = load("blocks")
    basis = ["cz", "sx", "rz", "x"]
    for build, cls in ((lambda q: (q.cx(0, 1), q.swap(0, 1)), "cx_move"), (lambda q: q.swap(0, 1), "swap"),
                       (lambda q: q.cx(1, 0), "cx")):
        q = QuantumCircuit(2)
        build(q)
        got = blocks.block_counts(transpile(q, basis_gates=basis, optimization_level=0))["blocks"]
        assert got[cls] == 1 and sum(got.values()) == 1, (cls, got)
