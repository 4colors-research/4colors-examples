import json

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

import run
from qaoakit import ansatz as az


def test_uccsd_style_keeps_particle_number():
    """Every excitation conserves the number of occupied orbitals: the state stays in the 2-particle sector."""
    qc, _ = az.build("uccsd-4-2")
    v = np.random.default_rng(3).uniform(-1, 1, len(qc.parameters))
    probs = Statevector(qc.assign_parameters(v)).probabilities()
    weight = np.array([bin(i).count("1") for i in range(16)])
    assert abs(probs[weight == 2].sum() - 1) < 1e-9
    assert abs(Statevector(qc.assign_parameters(np.zeros(len(v)))).probabilities()[0b0011] - 1) < 1e-9


def test_toy_hamiltonian_conserves_number():
    H = run.toy_hamiltonian(4).to_matrix()
    N = sum((np.eye(16) - SparsePauliOp.from_sparse_list([("Z", [i], 1)], 4).to_matrix()) / 2 for i in range(4))
    assert np.allclose(H, H.conj().T) and np.allclose(H @ N, N @ H)


def test_compile_late_end_to_end_offline(canned, tmp_path):
    run.main(["--out", str(tmp_path), "--ansatz", "qaoa-6-p1,uccsd-4-2", "--points", "3", "--maxiter", "12",
              "--budget", "0.01"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    traj = [r for r in rows if r["kind"] == "trajectory"]
    assert len(traj) == 2 and all(len(r["points"]) == 3 for r in traj)
    assert all(p["eval"] > 0 for r in traj for p in r["points"])          # the start point is never compiled
    uccsd = [r for r in traj if r["ansatz"].startswith("uccsd")][0]
    assert uccsd["start_compiled"] is False and uccsd["points"][0]["max_abs_angle"] > 0
    bound = [r for r in rows if r.get("mode", "").startswith("bound")]
    assert len(bound) == 2 * 3 * 3
    assert all(abs(r["fidelity_here"] - 1) < 1e-8 for r in bound)    # canned: the input circuit comes back unchanged
    param = [r for r in rows if r.get("mode") == "parametric"]
    assert len(param) == 4 and all(r["n_parameters"] == r["input_n_parameters"] for r in param)
    import plot

    plot.main(["--out", str(tmp_path)])
    for f in ("gates_along_trajectory.png", "fig_exact.png", "fig_per_block.png", "table.md", "cost.md",
              "selection.json"):
        assert (tmp_path / f).exists()
    q = [r for r in rows if r.get("arm") == "qiskit-o3" and r["mode"] == "bound-exact"]
    assert q and all(r["qiskit_approximation_degree"] == 1.0 for r in q)
    # D2: every 4colors compile sends its mode explicitly; the exact arms (compared with Qiskit) send "exact"
    sent = [c.get("approximation_mode") for c in canned.compiler.calls]
    assert sent and None not in sent and set(sent) == {"exact", "per_block"}
    ours = [r for r in rows if r.get("arm") == "4colors" and r.get("mode") in ("parametric", "bound-exact")]
    assert ours and all(r["approximation_mode"] == "exact" for r in ours)


def test_chosen_indices_skip_start_and_simplex():
    assert run.chosen_indices(120, 27, 8)[0] == 28 and run.chosen_indices(120, 27, 8)[-1] == 119
    assert 0 not in run.chosen_indices(12, 14, 3)                         # too short for the simplex rule
    assert len(run.chosen_indices(50, 4, 8)) == 8


def _row(spec, arm, mode, g, e=None, fid=1.0):
    r = dict(kind="compile", ansatz=spec, arm=arm, mode=mode, two_qubit_gates=g, fidelity_here=fid)
    if e is not None:
        r["eval"] = e
    return r


def test_selection_rule_exact_against_exact_only():
    """Shown only when ours is lower at every point, both exact to 1e-6; per_block rows never decide."""
    import plot

    rows = []
    for spec, ours, theirs, qfid in (("win", 80, 100, 1.0), ("tie", 100, 100, 1.0), ("approx-qiskit", 80, 100, 0.99),
                                     ("unmeasured", 80, 100, None)):
        rows += [_row(spec, "4colors", "parametric", 150), _row(spec, "qiskit-o3", "parametric", 160)]
        for e in (5, 9):
            rows += [_row(spec, "4colors", "bound-exact", ours, e), _row(spec, "qiskit-o3", "bound-exact", theirs, e, qfid),
                     _row(spec, "4colors", "bound-per_block", 10, e, 0.98)]
    sel = plot.select(rows)
    assert [s for s, (shown, _) in sel.items() if shown] == ["win"]
    assert "not lower" in sel["tie"][1] and "not exact" in sel["approx-qiskit"][1] and "not measured" in sel["unmeasured"][1]


def test_split_arms(canned, tmp_path):
    run.main(["--out", str(tmp_path / "q"), "--ansatz", "qaoa-6-p1", "--points", "2", "--maxiter", "8",
              "--budget", "0.01", "--skip-4colors"])
    rows = [json.loads(x) for x in (tmp_path / "q" / "results.jsonl").read_text().splitlines()]
    assert {r["arm"] for r in rows if r["kind"] == "compile"} == {"qiskit-o3"}
