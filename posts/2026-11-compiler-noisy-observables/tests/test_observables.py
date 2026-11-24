"""Offline tests for the noisy-observables post (no API call; the noise model runs locally on tiny circuits)."""
import cmath
import json

import pytest

import circuitkit as ck

import os

_POST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import postload  # noqa: E402  (blog/tests/postload.py; blog/tests is on sys.path from conftest)


def load(name):
    """Import <post dir>/<name>.py under a name unique to this post (blog/tests/postload.py), so posts with a
    run.py or plot.py of their own never collide when their tests run in one session."""
    return postload.load(_POST, name)


circuits = load("domain_circuits")
observables = load("observables")
noise = load("noise")


# ---- the circuits do something to |0...0> -----------------------------------------------------------

@pytest.mark.parametrize("cid", circuits.POST_CIRCUITS)
def test_every_post_circuit_is_non_trivial(cid):
    qc, spec = circuits.build(cid)                     # build() refuses a trivial circuit
    assert circuits.trivial_reason(qc) is None
    assert qc.num_qubits in (8, 9, 12) and spec["family"] in ("chemistry", "finance", "materials")
    assert set(qc.count_ops()) <= {"cx", "rz", "sx", "x", "h", "measure", "barrier"}


def test_the_old_generators_are_flagged():
    """UCCSD without its Hartree-Fock reference and portfolio QAOA started from |0...0> are the identity on
    the vacuum; a bare reference state is a single basis state."""
    import circuitkit as ck
    assert circuits.trivial_reason(ck.uccsd(4, 2, reference=False, measure=False)) == "identity on |0>"
    assert circuits.trivial_reason(circuits.portfolio_xy(6, 3, seed=6, dicke=False)) == "identity on |0>"
    assert circuits.trivial_reason(ck.uccsd(4, 2, thetas=[0.0] * 15, measure=False)) == "single basis state"


def test_dicke_state_is_uniform_over_weight_k():
    for n, k in ((4, 2), (6, 3), (5, 1)):
        probs = circuits.ideal_probabilities(circuits.dicke_state(n, k))
        from math import comb

        assert len(probs) == comb(n, k) and all(key.count("1") == k for key in probs)
        assert all(abs(p - 1 / comb(n, k)) < 1e-9 for p in probs.values())


def test_portfolio_cost_layer_is_the_mean_variance_cost():
    """With beta = 0 the circuit is the cost phase alone: exp(-i gamma C(x)) with C the scaled cost, up to a
    global phase."""
    from qiskit.quantum_info import Operator

    n, gamma = 4, 0.37
    mu, sigma = circuits.portfolio_data(n, seed=4)
    _, _, scale = circuits.ising_terms(mu, sigma)
    qc = circuits.portfolio_xy(n, 2, seed=4, angles=((gamma,), (0.0,)), dicke=False)
    diag = Operator(qc).data.diagonal()
    phases = []
    for i, d in enumerate(diag):
        x = [(i >> j) & 1 for j in range(n)]                       # qubit j is bit j of the index
        phases.append(cmath.phase(d) + gamma * circuits.portfolio_cost(x, mu, sigma) / scale)
    assert all(abs(cmath.exp(1j * (p - phases[0])) - 1) < 1e-9 for p in phases)


def test_xy_mixer_conserves_the_budget_and_ideal_feasibility_is_one():
    qc, spec = circuits.build("portfolio_8")
    ideal = observables.measure(spec, circuits.ideal_probabilities(qc))
    assert abs(ideal["feasible_probability"] - 1) < 1e-9
    assert ideal["feasible_mean_cost"] is not None


def test_uccsd_ideal_stays_in_its_sector():
    qc, spec = circuits.build("uccsd_8")
    ideal = observables.measure(spec, circuits.ideal_probabilities(qc))
    assert abs(ideal["sector_probability"] - 1) < 1e-9
    assert abs(sum(ideal["occupations"]) - 2) < 1e-9 and max(ideal["occupations"]) < 0.999


def test_tfim_starts_magnetized_and_moves():
    m = [observables.measure(circuits.build(f"tfim_3x3_s{s}")[1],
                             circuits.ideal_probabilities(circuits.build(f"tfim_3x3_s{s}")[0]))["magnetization"]
         for s in (1, 4)]
    assert 0.5 < m[0] < 1 and m[1] < m[0]


# ---- observables ------------------------------------------------------------------------------------

def test_counts_keys_are_turned_qubit_zero_first():
    dist = observables.from_counts({"0 01": 3, "001": 1})          # clbit 0 is the last character
    assert dist == {"100": 1.0}
    assert observables.magnetization({"000": 1.0}) == 1 and observables.magnetization({"111": 1.0}) == -1
    assert observables.occupations({"10": 0.25, "11": 0.75}, 2) == [1.0, 0.75]
    assert observables.feasible_probability({"1100": 0.6, "1000": 0.4}, 2) == 0.6


def test_errors_and_standard_errors():
    spec = {"observable": "feasibility", "n": 4, "k": 2, "seed": 4}
    assert observables.error(spec, {"feasible_probability": 0.8}, {"feasible_probability": 1.0}) == pytest.approx(0.2)
    assert observables.standard_error(spec, {"feasible_probability": 0.5}, 10_000) == pytest.approx(0.005)
    occ = {"observable": "occupations"}
    assert observables.error(occ, {"occupations": [0.9, 0.2]}, {"occupations": [1.0, 0.0]}) == pytest.approx(0.2)


# ---- the noise model --------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fez_sim():
    fez = ck.fez()
    return fez, noise.simulator(fez)


def test_schedule_adds_idle_delays_only_on_used_qubits(fez_sim):
    from qiskit import transpile

    fez, _ = fez_sim
    qc = circuits.lower(circuits.portfolio_xy(4, 2, seed=4))
    out = noise.schedule(transpile(qc, target=fez.target, optimization_level=1, seed_transpiler=1), fez.target)
    delayed = {out.find_bit(q).index for i in out.data if i.operation.name == "delay" for q in i.qubits}
    assert delayed and delayed <= noise.used_qubits(out) and len(noise.used_qubits(out)) < 10


def test_noise_lowers_feasibility_on_a_small_portfolio(fez_sim):
    from qiskit import transpile

    fez, sim = fez_sim
    qc = circuits.lower(circuits.portfolio_xy(4, 2, seed=4))
    compiled = transpile(qc, target=fez.target, optimization_level=1, seed_transpiler=1)
    dist, meta = noise.noisy_distribution(compiled, sim, fez.target, shots=4000, seed=1)
    p = observables.feasible_probability(dist, 2)
    assert 0.5 < p < 1 and meta["sim_method"] == "density_matrix" and meta["idle_noise"]


# ---- selection and plumbing -------------------------------------------------------------------------

def test_selection_needs_fewer_gates_and_a_lower_error_beyond_shot_noise():
    plot = load("plot")
    r = lambda cid, arm, g, e: {"kind": "sim", "circuit_id": cid, "arm": arm, "two_qubit_gates": g, "error": e,
                                "error_se": 0.01}
    rows = [r("a", "4colors", 10, 0.10), r("a", "qiskit", 12, 0.20),      # shown
            r("b", "4colors", 10, 0.19), r("b", "qiskit", 12, 0.20),      # within shot noise
            r("c", "4colors", 12, 0.10), r("c", "qiskit", 12, 0.20),      # same gates
            r("d", "4colors", 14, 0.10), r("d", "qiskit", 12, 0.20)]      # more gates
    shown, left = plot.select(rows)
    assert shown == ["a"] and [c for c, _ in left] == ["b", "c", "d"]


def test_steps_reached_stops_at_the_first_break():
    plot = load("plot")
    rows = [{"circuit_id": f"tfim_3x3_s{s}", "arm": "4colors", "error": e} for s, e in
            ((1, 0.01), (2, 0.05), (3, 0.2), (4, 0.05))]
    assert plot.steps_reached(rows, "3x3", "4colors") == 2


def test_run_simulate_plot_offline(canned, tmp_path):
    """Plumbing on a fixed canned result (the logical circuit transpiled by Qiskit, chosen by the test)."""
    from qiskit import transpile

    run, simulate, plot = load("run"), load("simulate"), load("plot")
    qc, _ = circuits.build("tfim_3x3_s1")
    canned.circuit = transpile(qc, target=ck.fez().target, optimization_level=1, seed_transpiler=7)
    run.main(["--out", str(tmp_path / "rec"), "--circuits", "tfim_3x3_s1,tfim_3x3_s2", "--budgets", "0.01",
              "--seeds", "1", "--max-qiskit-trials", "1"])
    rows = [json.loads(x) for x in (tmp_path / "rec" / "results.jsonl").read_text().splitlines()]
    res = [r for r in rows if r["kind"] == "result"]
    assert {r["arm"] for r in res} == {"4colors", "qiskit"} and all(r["circuit_file"] for r in res)
    assert all(0 < r["estimated_esp"] < 1 for r in res)
    fez_pairs = set(ck.device_couplers(ck.fez().target))
    for r in res:                     # coupler_counts from the saved circuit: on FakeFez couplers, summing to 2q
        cc = r["coupler_counts"]
        assert cc and sum(n for _, _, n in cc) == r["two_qubit_gates"]
        assert {(a, b) for a, b, _ in cc} <= fez_pairs
    assert all(c["approximation_mode"] == "exact" for c in canned.calls)
    simulate.main([str(tmp_path / "rec"), "--out", str(tmp_path / "sim"), "--shots", "2000"])
    sims = [json.loads(x) for x in (tmp_path / "sim" / "sim.jsonl").read_text().splitlines()]
    assert sum(r["kind"] == "ideal" for r in sims) == 2 and sum(r["kind"] == "sim" for r in sims) == 4
    assert all(0 <= r["error"] <= 2 for r in sims if r["kind"] == "sim")
    plot.main([str(tmp_path / "sim"), "--figs", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / "selection.md").exists()
