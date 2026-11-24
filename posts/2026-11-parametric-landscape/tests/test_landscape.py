import json
import math

import numpy as np
from qiskit.quantum_info import Statevector

import run
from qaoakit import ansatz as az, apikit as ak, core


def test_sign_convention_p0_and_p1():
    """p = 0 (gamma = beta = 0) gives the uniform average cut, |E|/2; the circuit and the numpy simulator agree."""
    inst = az.reg3_instance(8, 1)
    cost = core.cost_vector(inst)
    qc, th = core.build(inst, "std-p1")
    for g, b in ((0.0, 0.0), (0.6155, 0.3927), (-0.6155, 0.3927)):
        circ = Statevector(qc.assign_parameters(dict(zip(th, [g, b])))).probabilities() @ cost
        fast = np.abs(core.qaoa_statevector(8, cost, [g, b])) ** 2 @ cost
        assert abs(circ - fast) < 1e-9
    assert abs(np.abs(core.qaoa_statevector(8, cost, [0, 0])) ** 2 @ cost - len(inst["edges"]) / 2) < 1e-9
    # the fixed angles improve on random; the mirrored gamma does worse than random
    good = np.abs(core.qaoa_statevector(8, cost, [0.6155, 0.3927])) ** 2 @ cost
    bad = np.abs(core.qaoa_statevector(8, cost, [-0.6155, 0.3927])) ** 2 @ cost
    assert good > len(inst["edges"]) / 2 > bad


def test_landscape_end_to_end_offline(canned, tmp_path):
    run.main(["--out", str(tmp_path), "--n", "8", "--grid", "4", "--budget", "0.01", "--trajectories", "4",
              "--e2", "1e-2"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    ideal = next(r for r in rows if r["arm"] == "ideal")
    noisy = [r for r in rows if r["condition"] == "noisy"]
    assert len(noisy) == 2 and all(r["check_ok"] for r in noisy)
    assert np.array(ideal["ratio_grid"]).shape == (4, 4)
    # noise pulls the landscape toward the random-assignment value: less contrast than ideal
    span = lambda g: np.ptp(np.array(g))
    assert all(span(r["ratio_grid"]) < span(ideal["ratio_grid"]) for r in noisy)
    import plot

    plot.main(["--out", str(tmp_path)])
    assert (tmp_path / "landscape-reg3-n8-s1.png").exists() and (tmp_path / "metrics.md").exists()


def test_ideal_landscape_is_mirror_symmetric():
    """f(pi - gamma, beta) = f(gamma, beta) for a 3-regular (odd-degree) graph, exactly, at off-grid points too."""
    inst = az.reg3_instance(8, 1)
    cost = core.cost_vector(inst)
    f = lambda g, b: float(np.abs(core.qaoa_statevector(8, cost, [g, b])) ** 2 @ cost)
    for g, b in ((0.3, 0.2), (0.6155, 0.3927), (1.2, 1.1), (2.9, 0.05)):
        assert abs(f(g, b) - f(math.pi - g, b)) < 1e-9
        assert abs(f(g + math.pi, b) - f(g, -b)) < 1e-9          # the gamma + pi shift
        assert abs(f(g, b) - f(-g, -b)) < 1e-9                   # time reversal


def test_mirror_plan_and_grid():
    plan, src = run.mirror_plan(8)
    assert plan == [0, 1, 2, 3, 4]
    assert [src[i] for i in range(8)] == [0, 1, 2, 3, 4, 3, 2, 1]
    plan, src = run.mirror_plan(7)
    assert plan == [0, 1, 2, 3] and [src[i] for i in range(7)] == [0, 1, 2, 3, 3, 2, 1]
    checks = run.check_rows(32, 2)
    assert len(checks) == 2 and all(run.mirror_plan(32)[1][i] != i for i in checks)
    # the mirrored ideal grid equals the full ideal grid
    inst = az.reg3_instance(8, 1)
    gammas, betas = run.grid_axes(8)
    full = run.ideal_grid(inst, gammas, betas)
    rows, _ = run.mirror_plan(8)
    half = run.ideal_grid(inst, [gammas[i] for i in rows], betas)
    assert np.allclose(run.mirrored(dict(zip(rows, half)), 8), full, atol=1e-9)


def test_prediction_formula_by_hand():
    import plot

    # 10 CZ at e2 = 0.01, 20 pulses at e1 = 0.001, readout 0.01: 0.99^10 * 0.999^20 * 0.98^2
    want = 0.99 ** 10 * 0.999 ** 20 * 0.98 ** 2
    assert abs(plot.predicted_kept(10, 20, 0.01, 0.001, 0.01) - want) < 1e-15
    assert plot.predicted_kept(0, 0, 0.01, 0.001, 0.0) == 1.0


def test_prediction_matches_a_global_depolarizing_toy():
    """For a toy channel that is exactly global depolarizing (keep the state with probability F, else the maximally
    mixed state), the contrast kept is F: the first-order model is exact there."""
    import plot

    inst = az.reg3_instance(8, 1)
    gammas, betas = run.grid_axes(6)
    ideal = np.array(run.ideal_grid(inst, gammas, betas))
    cost = core.cost_vector(inst)
    F = plot.predicted_kept(12, 30, 3e-3, 3e-4, 0.0)
    noisy = F * ideal + (1 - F) * cost.mean()
    assert abs(np.ptp(noisy) / np.ptp(ideal) - F) < 1e-12


def test_argmax_distance_sees_the_mirror():
    import plot

    gam = [math.pi * i / 4 for i in range(4)]
    bet = [0.0, 0.5]
    ideal = {"ratio_grid": [[0, 0], [0, 1], [0, 0], [0, 0.99]], "gammas": gam, "betas": bet}
    mirror = {"ratio_grid": [[0, 0], [0, 0.99], [0, 0], [0, 1]], "gammas": gam, "betas": bet}
    assert plot.argmax_distance(mirror, ideal) < 1e-12           # the mirror image of the best point is not a move


def test_mirrored_run_records_checks(canned, tmp_path):
    run.main(["--out", str(tmp_path), "--n", "8", "--grid", "6", "--budget", "0.01", "--trajectories", "4",
              "--e2", "1e-2", "--mirror-checks", "1"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    ideal = next(r for r in rows if r["arm"] == "ideal")
    assert ideal["ideal_symmetry_max_abs"] < 1e-9
    noisy = [r for r in rows if r["condition"] == "noisy"]
    assert all(r["mirrored"] and r["rows_simulated"] == 4 + 1 + 1 for r in noisy)   # rows 0..3, 1 check, 1 repeat
    assert all(r["mirror_check_max_abs"] is not None and r["repeat_check_max_abs"] is not None for r in noisy)
    assert all(r["one_qubit_pulses"] >= 0 for r in noisy)
    g = np.array(noisy[0]["ratio_grid"])
    assert np.allclose(g[1], g[5]) and np.allclose(g[2], g[4])                   # mirrored rows
    import plot

    plot.main(["--out", str(tmp_path)])
    text = (tmp_path / "metrics.md").read_text()
    assert "the full record, every graph included, is linked" in text.lower() and (tmp_path / "kept-vs-predicted.png").exists()


def test_qiskit_gets_the_published_map_and_calibration_ours_the_working_map():
    """Fairness rule: the rival gets the pure calibration. Qiskit's target is FakeFez as shipped (all 176 couplers,
    the 7 at error 1.0 included, error rates as published); the 4colors arm keeps the working-coupler description."""
    import circuitkit as ck

    t = ak.qiskit_target()
    assert len(ck.device_couplers(t)) == 176 and len(ck.failed_couplers(t)) == 7
    assert any(p is not None and p.error is not None for p in t["cz"].values())       # calibration present
    assert t is ck.fez().target                                                        # as shipped, not a copy
    ours = ak.device_target()
    assert len(ours["couplers"]) == 169 and "calibration" not in ours


def test_qiskit_arm_record_names_the_published_target(tmp_path):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    out, rec = ak.compile_qiskit(qc, budget_s=0.0, seed=1)
    assert rec["arm"] == "qiskit-o3" and rec["tries"] == 1 and rec["qiskit_couplers"] == 176
    assert rec["target"] == "fakefez" and rec["target_view"].startswith("published")
    assert out.num_qubits == 156 and rec["two_qubit_gates"] >= 2
    assert rec["gates_on_failed_couplers"] == 0 and 0 < rec["estimated_esp"] < 1


def test_one_scoring_rule_for_both_arms():
    """device_score prices any circuit on FakeFez's qubits from the published calibration: a gate on a coupler
    published at error 1.0 gives ESP 0, whichever compiler placed it."""
    from qiskit import QuantumCircuit

    a, b = sorted(ak.failed_couplers())[0]
    bad = QuantumCircuit(156)
    bad.cz(a, b)
    assert ak.device_score(bad) == {"gates_on_failed_couplers": 1, "estimated_esp": 0.0}
    good = QuantumCircuit(156)
    good.cz(*ak.target_couplers()[0])
    s = ak.device_score(good)
    assert s["gates_on_failed_couplers"] == 0 and 0 < s["estimated_esp"] < 1
