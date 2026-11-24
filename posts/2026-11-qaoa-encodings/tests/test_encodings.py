import json

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

import run
from qaoakit import apikit as ak, core

INST = dict(id="m3c-tiny", kind="max3cut", nodes=3, edges=[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 0.5)])


def diag_unitary_equal(qc, phases):
    u = Operator(qc).data
    d = np.exp(-1j * np.asarray(phases))
    k = np.argmax(np.abs(np.diag(u)))
    return np.allclose(u, np.diag(d) * (u[k, k] / d[k]), atol=1e-9)


def test_qubo_phase_layer_is_the_qubo_energy():
    Q, off = ak.max3cut_qubo(2, [(0, 1, 1.0)], penalty=2.5)
    qc = QuantumCircuit(6)
    run.qubo_phase(qc, Q, 0.37)
    idx = np.arange(64)
    X = ((idx[:, None] >> np.arange(6)) & 1).astype(float)
    E = np.einsum("bi,ij,bj->b", X, Q, X)
    assert diag_unitary_equal(qc, 0.37 * (-E))          # exp(-i gamma f), f = -E


def test_binary_phase_layer_is_the_cut():
    qc = QuantumCircuit(4)
    run.diag_phase(qc, [0, 1, 2, 3], run.edge_table_binary(1.5), 0.81)
    cut = run.binary_cost(2, [(0, 1, 1.5)])
    assert diag_unitary_equal(qc, 0.81 * cut)


def test_penalty_qubo_and_rows_agree_with_the_brute_force_optimum():
    opt = ak.max3cut_optimum(3, INST["edges"])
    assert opt == core.cost_vector(INST).max() == run.binary_cost(3, INST["edges"]).max() == 2.5
    Q, off = ak.max3cut_qubo(3, INST["edges"], penalty=run.penalty_weight(INST))
    idx = np.arange(2 ** 9)
    X = ((idx[:, None] >> np.arange(9)) & 1).astype(float)
    E = off + np.einsum("bi,ij,bj->b", X, Q, X)
    assert abs(-E.min() - opt) < 1e-9                   # the penalty never pays: minimum is a feasible optimum
    assert run.decode_onehot_cut(X[E.argmin()].astype(int).tolist(), INST) == (opt, True)


def test_xy_mixer_keeps_one_hot_and_penalty_mixer_does_not():
    qc, th, cost, feas = run.build(INST, "onehot-xy", 1)
    probs = Statevector(qc.assign_parameters([0.5, 0.8])).probabilities()
    assert abs(probs[feas].sum() - 1) < 1e-9
    qc, th, cost, feas = run.build(INST, "onehot-penalty", 1)
    probs = Statevector(qc.assign_parameters([0.5, 0.8])).probabilities()
    assert probs[feas].sum() < 0.99


def test_binary_start_state_is_biased_toward_color_2():
    """Uniform bits give color 2 half the time; the start-state ratio is the encoding's own baseline, below a uniform
    random coloring's 2/3 of the edges."""
    n = 2
    counts = np.bincount([run.decode_binary(i, n)[0] for i in range(4 ** n)], minlength=3) / 4 ** n
    assert np.allclose(counts, run.BINARY_COLOR_PROBS)
    opt = ak.max3cut_optimum(3, INST["edges"])
    W = sum(w for _, _, w in INST["edges"])
    cost, feas = run.scores(INST, "binary")
    assert abs(run.start_ratio("binary", cost, feas, opt) - (5 / 8) * W / opt) < 1e-12      # 1 - 1/16 - 1/16 - 1/4
    cost, feas = run.scores(INST, "onehot-xy")
    assert abs(run.start_ratio("onehot-xy", cost, feas, opt) - (2 / 3) * W / opt) < 1e-12   # W state: uniform coloring
    assert run.start_ratio("onehot-penalty", cost, feas, opt) < run.start_ratio("onehot-xy", cost, feas, opt)


def test_xy_start_ratio_matches_the_w_state():
    qc, th, cost, feas = run.build(INST, "onehot-xy", 1)
    probs = Statevector(qc.assign_parameters([0.0, 0.0])).probabilities()    # no QAOA layer: the start state
    opt = ak.max3cut_optimum(3, INST["edges"])
    assert abs(probs @ cost / opt - run.start_ratio("onehot-xy", cost, feas, opt)) < 1e-9


def test_brooks_certificate_matches_enumeration_and_refuses_k4():
    inst = run.reg3_max3cut(8, seed=2)
    assert run.brooks_optimum(inst) == ak.max3cut_optimum(8, inst["edges"]) == len(inst["edges"])
    k4 = dict(nodes=4, edges=[(a, b, 1.0) for a in range(4) for b in range(a + 1, 4)])
    assert run.brooks_optimum(k4) is None and ak.max3cut_optimum(4, k4["edges"]) == 5.0
    split = dict(nodes=4, edges=[(0, 1, 1.0), (2, 3, 1.0)])
    assert run.brooks_optimum(split) is None


def test_circuit_builds_large_graphs_without_score_vectors():
    inst = run.reg3_max3cut(24)
    for enc in run.ENCODINGS:
        qc, th = run.circuit(inst, enc, 1)
        assert qc.num_qubits == run.QUBITS_PER_NODE[enc] * 24 and len(th) == 2


def _q(enc, p, arm, gates, noisy, seed=1, check=True, refused=False):
    r = dict(kind="qaoa", encoding=enc, p=p, arm=arm, seed=seed, n_qubits=9, ideal_ratio=0.9, start_ratio=0.6)
    if refused:
        return dict(r, refusal_code="input_too_large")
    return dict(r, two_qubit_gates=gates, simulated=True, check_ok=check, noisy_ratio_mean=noisy, noisy_feasible=0.9)


def test_selection_shows_only_cells_ahead_on_gates_and_not_behind_on_noise():
    import plot

    rows = []
    for s in (1, 2):
        rows += [_q("win", 1, "4colors", 40, 0.80, s), _q("win", 1, "qiskit-o3", 60, 0.75, s),
                 _q("tie", 1, "4colors", 60, 0.80, s), _q("tie", 1, "qiskit-o3", 60, 0.80, s),
                 _q("noisier", 1, "4colors", 40, 0.70, s), _q("noisier", 1, "qiskit-o3", 60, 0.75, s),
                 _q("unchecked", 1, "4colors", 40, 0.80, s, check=s == 1), _q("unchecked", 1, "qiskit-o3", 60, 0.7, s),
                 _q("refused", 1, "4colors", 40, 0.80, s, refused=s == 2), _q("refused", 1, "qiskit-o3", 60, 0.7, s)]
    shown, left = plot.select_cells(rows)
    assert [e for e, _, _ in shown] == ["win"]
    why = {e: w for e, _, w in left}
    assert set(why) == {"tie", "noisier", "unchecked", "refused"} and "noisy" in why["noisier"]


def test_encodings_end_to_end_offline(canned, tmp_path):
    run.main(["--out", str(tmp_path), "--ps", "1", "--seeds", "1", "--budget", "0.01", "--trajectories", "4",
              "--batches", "2", "--e2", "1e-2", "--qubo-limits", "0.01", "--qubo-seeds", "1", "--maxiter", "60",
              "--count-seeds", "1"],
             instances=[INST], count_instances=[run.reg3_max3cut(6)], optimizer_instances=[run.reg3_max3cut(10)])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    q = [r for r in rows if r["kind"] == "qaoa"]
    assert {r["encoding"] for r in q} == set(run.ENCODINGS) and all(r["check_ok"] for r in q)
    assert {r["n_qubits"] for r in q} == {9, 6}
    assert all(0 < r["start_ratio"] <= 1 and r["tuned_on"].startswith("score") for r in q)
    cnt = [r for r in rows if r["kind"] == "count"]
    assert {(r["encoding"], r["n_qubits"]) for r in cnt} == {("onehot-penalty", 18), ("onehot-xy", 18), ("binary", 12)}
    assert all(r["arm"] == "4colors" and r["simulated"] is False for r in cnt)
    opt_rows = [r for r in rows if r["kind"] == "optimizer" and r.get("cut") is not None]
    assert len(opt_rows) == 4 and all(0.0 <= r["classical_ratio"] <= 1.0 for r in opt_rows)   # canned all-zeros answer
    big = [r for r in opt_rows if r.get("large")]
    assert big and all(r["optimum_certificate"] == "brooks" and r["exact_optimum"] == 15.0 for r in big)
    import plot

    plot.main(["--out", str(tmp_path), "--figs", str(tmp_path / "figs")])
    table = (tmp_path / "figs" / "table.md").read_text()
    assert (tmp_path / "figs" / "encodings.png").exists() and (tmp_path / "figs" / "compiler.png").exists()
    assert "binary" in table and "Brooks" in table and "m3c-reg3-n6-s1" in table
    assert "Left out" in (tmp_path / "figs" / "selection.md").read_text()


def test_arms_split_into_two_records_with_the_same_angles(canned, tmp_path):
    common = ["--ps", "1", "--seeds", "1", "--budget", "0.01", "--trajectories", "2", "--batches", "1",
              "--qubo-limits", "0.01", "--qubo-seeds", "1", "--maxiter", "40", "--encodings", "binary"]
    run.main(["--out", str(tmp_path / "api"), "--arms", "4colors", *common], instances=[INST], count_instances=[],
             optimizer_instances=[])
    run.main(["--out", str(tmp_path / "local"), "--arms", "qiskit-o3", *common], instances=[INST])
    api = [json.loads(x) for x in (tmp_path / "api" / "results.jsonl").read_text().splitlines()]
    loc = [json.loads(x) for x in (tmp_path / "local" / "results.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in loc} == {"qaoa"} and {r["arm"] for r in loc} == {"qiskit-o3"}
    qa = [r for r in api if r["kind"] == "qaoa"]
    assert {r["arm"] for r in qa} == {"4colors"} and qa[0]["angles"] == loc[0]["angles"]
