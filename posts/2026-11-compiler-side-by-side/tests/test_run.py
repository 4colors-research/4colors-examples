"""Offline tests for the side-by-side kit: builders, the D2 rule, the checks, the selection rule, and the pipeline
snapshot -> run.py -> check.py -> inspect_cell.py -> select_cells.py with canned 4colors results (nothing compiled
by the service; the Qiskit and TKET arms run on tiny circuits)."""
import json

import numpy as np
import pytest

import circuits
from conftest import load


def _basis(qc):
    from qiskit.quantum_info import Statevector

    p = Statevector(qc).probabilities()
    k = int(np.argmax(p))
    return format(k, f"0{qc.num_qubits}b")[::-1], float(p[k])         # bits[q] = qubit q


@pytest.mark.parametrize("a,b", [(0, 0), (3, 5), (7, 7), (6, 1)])
def test_cuccaro_adds(a, b):
    qc = circuits.cuccaro_adder(3, a, b)
    assert qc.num_qubits == 8 and all(len(i.qubits) <= 2 for i in qc.data)
    bits, p = _basis(qc)
    assert p > 1 - 1e-9
    assert circuits.cuccaro_sum(3, bits) == ((a + b) % 8, (a + b) >> 3)
    assert all(int(bits[1 + 2 * i]) == (a >> i) & 1 for i in range(3)) and bits[0] == "0"   # a kept, ancilla clean


def test_lattice_and_chain_counts():
    assert len(circuits.grid_edges(6, 6)) == 60 and len(circuits.grid_edges(5, 5)) == 40
    tf = circuits.build("tfim2d-n36")
    assert tf.count_ops()["rzz"] == 60 and tf.num_qubits == 36
    he = circuits.build("heis2d-n25")
    assert he.count_ops()["rxx"] == he.count_ops()["ryy"] == he.count_ops()["rzz"] == 40
    ch = circuits.build("ising1d-n32")
    assert ch.count_ops()["rzz"] == 62 and circuits.control_floor(ch) == 124


def test_every_circuit_builds_measured_and_two_local():
    for name, info in circuits.CIRCUITS.items():
        qc = circuits.build(name)
        assert all(len(i.qubits) <= 2 for i in qc.data if i.operation.name != "barrier"), name
        assert qc.count_ops().get("measure", 0) >= qc.num_qubits - (1 if info["family"] == "bernstein_vazirani" else 0)
    assert circuits.build("bv-n48").count_ops()["cx"] == 24
    assert circuits.build("bv-n100").num_qubits == 101
    assert sum(1 for i in circuits.CIRCUITS.values() if i.get("record_only")) == 1


def test_chain_fits_the_device():
    """The control is a native fit: the heavy-hex map of a 156-qubit device holds a 32-qubit path."""
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    adj = {}
    for a, b in FakeKingston().coupling_map.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def dfs(path, seen):
        if len(path) == 32:
            return True
        return any(dfs(path + [n], seen | {n}) for n in sorted(adj[path[-1]]) if n not in seen)

    assert any(dfs([s], {s}) for s in sorted(adj) if len(adj[s]) == 1 or s == 0)


def test_d2_options():
    run = load("run")
    assert run.d2_options([]) is None
    assert run.d2_options([0.0, None]) is None                      # one unmeasured rival: the cell waits
    assert run.d2_options([0.0, 1e-14]) == {"approximation_mode": "exact"}
    assert run.d2_options([3e-6, 1e-7]) == {"max_approximation_error": 1e-7}    # never looser than any rival


def _transpiled(qc, seed=1):
    from qiskit import transpile
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    return transpile(qc, target=FakeKingston().target, optimization_level=3, seed_transpiler=seed)


def test_checks_pass_correct_and_catch_wrong():
    import circuitkit as ck
    from qiskit import qasm3

    check = load("check")
    run = load("run")
    bv = ck.bernstein_vazirani("10110")
    out = _transpiled(bv)
    assert check.clifford_secret(out, "10110")["status"] == "PASS"
    assert check.clifford_secret(out, "10111")["status"] == "FAIL"
    qft = circuits._measured(ck.qft(4))
    out = _transpiled(qft)
    lay = {"initial": run.initial_layout(out, 4)}
    r = check.check_one(qasm3.dumps(qft), qasm3.dumps(out), lay, "states")
    assert r["status"] == "PASS" and r["distance"] < 1e-9
    bad = out.copy_empty_like()
    for i in out.data:
        bad.append(i)
        if i.operation.name == "cz":
            bad.rz(0.05, i.qubits[0])
            break
    for i in out.data[len(bad.data) - 1:]:
        bad.append(i)
    r = check.check_one(qasm3.dumps(qft), qasm3.dumps(bad), lay, "states")
    assert r["status"] in ("APPROX", "FAIL") and r["distance"] > 1e-6


def _rep(circuit, arms):
    """A minimal inspect report: {arm: [(two_qubit_gates, status, distance, extra)]}."""
    def seed(n2, st, d, extra):
        return {"stem": f"x{n2}", "metrics": {"two_qubit_gates": n2}, "check": {"status": st, "distance": d}, **extra}
    return {"device": "dev", "circuit": circuit,
            "arms": {a: {"seeds": [seed(*s) for s in ss]} for a, ss in arms.items()}}


def test_selection_rule_shows_only_strict_verified_wins():
    sel = load("select_cells")
    t = {"wall_s": 1.0}
    win = _rep("bv-n48", {"ours_inst": [(10, "PASS", 0.0, {"billed_s": 0.5})], "qiskit_o3": [(20, "PASS", 0.0, t)]})
    tie = _rep("qft-n20", {"ours_inst": [(20, "PASS", 0.0, {})], "qiskit_o3": [(20, "PASS", 0.0, t)]})
    unver = _rep("qaoa3r-n20", {"ours_inst": [(5, "UNKNOWN", None, {})], "qiskit_o3": [(20, "PASS", 0.0, t)]})
    approx_ok = _rep("qft-n32", {"ours_T1": [(5, "APPROX", 1e-7, {"options": {"max_approximation_error": 2e-7}})],
                                 "qiskit_o3best_T1": [(9, "APPROX", 2e-7, {})]})
    approx_over = _rep("draper-n16", {"ours_T1": [(5, "APPROX", 3e-7, {"options": {"max_approximation_error": 2e-7}})],
                                      "qiskit_o3best_T1": [(9, "APPROX", 2e-7, {})]})
    heis = _rep("heis2d-n25", {"ours_inst": [(1, "PASS", 0.0, {})], "qiskit_o3": [(9, "PASS", 0.0, t)]})
    ctrl = _rep("ising1d-n32", {"ours_inst": [(124, "PASS", 0.0, {})], "qiskit_o3": [(124, "PASS", 0.0, t)]})
    rows = {r["circuit"]: r for r in sel.select([win, tie, unver, approx_ok, approx_over, heis, ctrl])}
    assert rows["bv-n48"]["shown"] and rows["bv-n48"]["same_time"]
    assert not rows["qft-n20"]["shown"] and not rows["qaoa3r-n20"]["shown"]
    assert rows["qft-n32"]["shown"] and not rows["draper-n16"]["shown"]
    assert not rows["heis2d-n25"]["shown"] and rows["ising1d-n32"]["reason"] == "control"
    assert "Internal (not for publication): pairings shown 3 of 7" in sel.markdown(list(rows.values()))


def test_pipeline_offline(canned, tmp_path, monkeypatch):
    import circuitkit as ck
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    monkeypatch.setitem(circuits.CIRCUITS, "bv-n4", {"cid": "T1", "family": "bernstein_vazirani", "check": "clifford"})
    monkeypatch.setitem(circuits.CIRCUITS, "qft-n4", {"cid": "T2", "family": "qft", "check": "states"})
    orig = circuits.build
    monkeypatch.setattr(circuits, "build", lambda n: {"bv-n4": lambda: ck.bernstein_vazirani("1011"),
                                                      "qft-n4": lambda: circuits._measured(ck.qft(4))}
                        .get(n, lambda: orig(n))())
    snap, out = tmp_path / "snap", tmp_path / "out"
    load("snapshot").save(FakeKingston(), str(snap), "fake:FakeKingston")
    run, check, insp, sel = load("run"), load("check"), load("inspect_cell"), load("select_cells")
    common = ["--snapshot", str(snap), "--out", str(out), "--circuits", "bv-n4,qft-n4"]
    run.main(common + ["--arms", "qiskit_o3,qiskit_o3best,tket_tk2", "--qiskit-seeds", "1,2", "--budgets", "0.01",
                       "--max-qiskit-trials", "1"])
    dev = out / json.loads((snap / "snapshot.json").read_text())["device"]
    check.main([str(dev)])
    out2 = tmp_path / "out2"                         # ours in its own record, bound read from the rivals' record
    run.main(["--snapshot", str(snap), "--out", str(out2), "--circuits", "bv-n4,qft-n4", "--arms", "ours",
              "--ours-seeds", "1", "--budgets", "0.01", "--rivals-from", str(dev),
              "--checks-from", str(dev / "checks.jsonl")])
    dev2 = out2 / dev.name
    check.main([str(dev2), "--out", str(tmp_path / "c2")])
    rivals = [json.loads(x) for x in (dev / "rivals.jsonl").read_text().splitlines()]
    ours = [json.loads(x) for x in (dev2 / "ours.jsonl").read_text().splitlines()]
    checks = [json.loads(x) for f in (dev / "checks.jsonl", tmp_path / "c2" / "checks.jsonl")
              for x in f.read_text().splitlines()]
    assert {r["arm"] for r in rivals} >= {"qiskit_o3", "qiskit_o3best_T0.01"}
    assert {r["arm"] for r in ours} == {"ours_inst", "ours_T0.01"} and len(ours) == 4
    assert all(c["status"] == "PASS" for c in checks), checks
    assert len(checks) == len([r for r in rivals if r.get("stem")]) + len(ours)
    assert all(c.get("approximation_mode") == "exact" for c in canned.calls)      # every rival measured 0
    assert all(c["target"].calibration is not None for c in canned.calls)          # ours got the published calibration
    reports = [insp.main([str(dev / cid), "--out", str(tmp_path / "insp"), "--records", str(dev2),
                          "--checks", str(tmp_path / "c2" / "checks.jsonl")]) for cid in ("bv-n4", "qft-n4")]
    assert all(s.get("check") for r in reports for a in r["arms"].values() for s in a["seeds"])
    assert all(s["agrees_with_record"] for r in reports for a in r["arms"].values() for s in a["seeds"])
    assert (tmp_path / "insp" / dev.name / "bv-n4" / "heat-qiskit_o3.svg").stat().st_size > 0
    rows = sel.select(reports)
    assert {r["pairing"] for r in rows} == {"same time", "same budget 0.01 s"}
    mp = load("make_preview").main([str(tmp_path / "insp")])
    assert len(mp) == 2


def test_waiting_cell_compiles_nothing(canned, tmp_path):
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    snap, out = tmp_path / "snap", tmp_path / "out"
    load("snapshot").save(FakeKingston(), str(snap), "fake:FakeKingston")
    load("run").main(["--snapshot", str(snap), "--out", str(out), "--circuits", "bv-n48", "--arms", "ours",
                      "--ours-seeds", "1"])
    dev = next(out.iterdir())
    rows = [json.loads(x) for x in (dev / "ours.jsonl").read_text().splitlines()]
    assert rows and all(r["status"] == "waiting_d2" for r in rows) and not canned.calls
