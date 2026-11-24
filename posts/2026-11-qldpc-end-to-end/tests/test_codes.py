import itertools

import numpy as np
import pytest

import codes as cd


@pytest.mark.parametrize("name,n,k", [("bb_72_12_6", 72, 12), ("bb_90_8_10", 90, 8), ("bb_108_8_10", 108, 8),
                                      ("bb_144_12_12", 144, 12), ("bb_288_12_18", 288, 12), ("toric_3", 18, 2)])
def test_code_parameters(name, n, k):
    hx, hz, _ = cd.code(name)
    assert hx.shape[1] == n and cd.is_css(hx, hz) and cd.n_logical(hx, hz) == k
    assert set(hx.sum(axis=1)) == ({4} if name == "toric_3" else {6})
    L = cd.x_logicals(hx, hz)
    assert L.shape == (k, n)
    assert not ((hz.astype(int) @ L.T) % 2).any()                     # X logicals commute with the Z checks
    assert cd.rank(np.vstack([hx, L])) == cd.rank(hx) + k             # and are independent modulo the X checks


def test_gf2_nullspace():
    M = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    N = cd.nullspace(M)
    assert N.shape == (2, 4) and not ((M.astype(int) @ N.T) % 2).any() and cd.rank(N) == 2


def test_toric_distance_by_enumeration():
    hx, hz, d = cd.code("toric_3")
    assert cd.brute_force_distance(hx, hz) == d == 3


def test_distance_model_feasible_set_and_optimum_by_enumeration():
    """On a [[4,2,2]]-like toy code every assignment is enumerated: rows hold exactly when e commutes with the X
    check and anticommutes with l, for some slack; the least objective over feasible points is the true optimum."""
    hx = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    hz = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    assert cd.n_logical(hx, hz) == 2
    for l in cd.x_logicals(hx, hz):
        m = cd.distance_model(hx, l)
        feas_e, best = set(), None
        for x in itertools.product([0, 1], repeat=m.n):
            if m.feasible(x):
                e = cd.error_of(x, 4)
                assert cd.certify(hx, l, e)
                feas_e.add(tuple(e))
                best = m.energy(x) if best is None else min(best, m.energy(x))
        want = {e for e in itertools.product([0, 1], repeat=4) if cd.certify(hx, l, np.array(e))}
        assert feas_e == want and best == 2


def test_distance_model_sizes_on_bb72():
    hx, hz, _ = cd.code("bb_72_12_6")
    l = cd.x_logicals(hx, hz)[0]
    m = cd.distance_model(hx, l)
    n_vars, rows, (lsup, lslack) = cd.distance_layout(hx, l)
    assert m.n == n_vars == 72 + 36 * 2 + len(lslack) and m.m == 37
    assert cd.slack_bits(6) == 2 and cd.slack_bits(1) == 0 and cd.slack_bits(7) == 2 and cd.slack_bits(8) == 3


def test_certify_rejects_stabilizers_and_non_commuting():
    hx, hz, _ = cd.code("toric_3")
    l = cd.x_logicals(hx, hz)[0]
    assert not cd.certify(hx, l, hz[0])                                # a Z stabilizer commutes with l
    e = np.zeros(18, dtype=np.uint8)
    e[0] = 1
    assert not cd.certify(hx, l, e) or not ((hx.astype(int) @ e) % 2).any()


def test_tanner_chip():
    hx, hz, _ = cd.code("bb_72_12_6")
    c = cd.tanner_couplers(hx, hz)
    deg = np.bincount(np.array(c).ravel(), minlength=144)
    assert len(c) == 432 and set(deg) == {6}
    d = cd.tanner_description(hx, hz)
    assert d["n_qubits"] == 144 and d["two_qubit_gate"] == "cz" and len(d["couplers"]) == 432


def test_syndrome_round_and_its_check():
    hx, hz, _ = cd.code("toric_3")
    qc = cd.syndrome_round(hx, hz)
    assert qc.count_ops()["cx"] == int(hx.sum() + hz.sum()) and qc.num_clbits == 18
    assert cd.syndrome_check(qc, hx, hz)["syndrome_check"] == "pass"
    broken = qc.copy()
    del broken.data[3]                                                # drop one CNOT of a Z check
    assert cd.syndrome_check(broken, hx, hz)["syndrome_check"] == "fail"


def test_check_reads_a_routed_resynthesized_round():
    """A Qiskit-routed round (re-synthesized rotations, a layout) is read through its layout and its own measures."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    hx, hz, _ = cd.code("toric_3")
    qc = cd.syndrome_round(hx, hz)
    out = transpile(qc, coupling_map=CouplingMap.from_heavy_hex(5), basis_gates=["cz", "rz", "sx", "x"],
                    optimization_level=3, seed_transpiler=1)
    assert cd.syndrome_check(out, hx, hz)["syndrome_check"] == "pass"


def test_pauli_expectation_by_hand():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(1, 0)                                                  # Z1 back-propagates to Z0 Z1 ... then X0 Z1
    obs = cd.measured_observables(qc)[0]
    assert cd.expectation(obs, {0: "+", 1: "0"}) == pytest.approx(1.0)
    assert cd.expectation(obs, {0: "-", 1: "0"}) == pytest.approx(-1.0)
    assert cd.expectation(obs, {0: "0", 1: "0"}) == pytest.approx(0.0)
