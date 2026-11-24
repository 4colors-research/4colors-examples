import itertools
import random

import numpy as np
import pytest

import grouping as gp


def random_labels(n, m, seed):
    rng = random.Random(seed)
    out = set()
    while len(out) < m:
        lab = "".join(rng.choice("IXYZ") for _ in range(n))
        if set(lab) != {"I"}:
            out.add(lab)
    return sorted(out)


def test_commutation_rules_by_matrices():
    from qiskit.quantum_info import Pauli

    for a, b in itertools.product(["XI", "XX", "YZ", "ZZ", "IY", "XY"], repeat=2):
        A, B = Pauli(a).to_matrix(), Pauli(b).to_matrix()
        assert gp.commute_gc(a, b) == np.allclose(A @ B, B @ A)
    assert gp.commute_qwc("XI", "XZ") and not gp.commute_qwc("XX", "YY") and gp.commute_gc("XX", "YY")


@pytest.mark.parametrize("kind", ["gc", "qwc"])
def test_conflict_edges_match_pairwise_rule(kind):
    labels = random_labels(5, 60, seed=3)
    rule = gp.commute_gc if kind == "gc" else gp.commute_qwc
    ref = {(i, j) for i, j in itertools.combinations(range(len(labels)), 2) if not rule(labels[i], labels[j])}
    assert set(gp.conflict_edges(labels, kind, chunk=7)) == ref


def test_coloring_model_rows_and_objective_by_enumeration():
    edges = [(0, 1), (1, 2), (0, 2), (2, 3)]          # a triangle with a pendant: 3 colors needed
    m, k = 4, 3
    model = gp.coloring_model(m, edges, k)
    proper = 0
    for x in itertools.product([0, 1], repeat=m * k):
        colors = gp.decode(x, m, k)
        assert model.feasible(x) == (colors is not None and colors[0] == 0)
        if model.feasible(x):
            assert model.energy(x) == gp.n_conflicts(colors, edges)
            proper += model.energy(x) == 0
    assert proper == 2 * 2        # term 0 fixed to color 0: 2 ways for the triangle, 2 for the pendant
    assert gp.decode([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], m, k) is None


def test_two_colors_cannot_color_a_triangle():
    model = gp.coloring_model(3, [(0, 1), (1, 2), (0, 2)], 2)
    best = min(model.energy(x) for x in itertools.product([0, 1], repeat=6) if model.feasible(x))
    assert best == 1


def test_baselines_are_valid_groupings():
    terms = [(lab, 1.0 + i % 3) for i, lab in enumerate(random_labels(4, 40, seed=5))]
    labels = [t[0] for t in terms]
    for kind in ("gc", "qwc"):
        for fn in (gp.sorted_insertion, gp.qiskit_grouping):
            g = fn(terms, kind)
            assert gp.valid_grouping(labels, g, kind)
    assert not gp.valid_grouping(["XI", "ZI"], [[0, 1]], "gc")        # anticommuting pair in one group
    assert not gp.valid_grouping(["XI", "ZI"], [[0]], "gc")           # a term missing


def test_groups_from_colors_renumbers_and_drops_empty():
    assert gp.groups_from_colors([2, 0, 2, 5]) == [[0, 2], [1], [3]]


def test_modeled_shot_factor_by_hand():
    terms = [("XI", 3.0), ("IX", 4.0), ("ZZ", 1.0)]
    assert gp.modeled_shot_factor(terms, [[0], [1], [2]]) == pytest.approx(1.0)
    assert gp.modeled_shot_factor(terms, [[0, 1], [2]]) == pytest.approx((5 + 1) ** 2 / 8 ** 2)
    assert gp.shot_shares(terms, [[0, 1], [2]]) == pytest.approx([5 / 6, 1 / 6])
    assert gp.priced_cost(terms, [[0, 1], [2]], [0, 6]) == pytest.approx(1.0)
    assert gp.priced_cost(terms, [[0, 1], [2]], [0, None]) is None


def test_diagonalization_circuits_for_commuting_groups():
    from qiskit.quantum_info import Pauli, random_clifford

    for seed in range(6):
        n = 4
        cliff = random_clifford(n, seed=seed)
        zs = ["".join(random.Random(seed * 10 + j).choice("IZ") for _ in range(n)) for j in range(6)]
        zs = [z for z in zs if set(z) != {"I"}] or ["ZIII"]
        group = sorted({Pauli(z).evolve(cliff, frame="s").to_label().lstrip("-+i") for z in zs})
        qc = gp.diagonalization_circuit(group)
        assert gp.diagonalizes(qc, group)
        assert qc.count_ops().get("measure") is None
    assert not gp.needs_entangling(["XI", "XZ"]) and gp.needs_entangling(["XX", "ZZ"])
    with pytest.raises(ValueError):
        gp.diagonalization_circuit(["XI", "ZI"])


def test_dependent_group_is_handled():
    group = ["XX", "YY", "ZZ"]                # ZZ = -XX * YY: the dependent one must not be passed twice
    assert gp.independent(group) == ["XX", "YY"]
    assert gp.diagonalizes(gp.diagonalization_circuit(group), group)


def test_parse_openfermion_text():
    text = "(0.5+0j) [X0 Y1] +\n(-0.25+0j) [Z3] +\n(1.0+0j) []"
    assert gp.parse_openfermion(text) == [("IIYX", 0.5), ("ZIII", -0.25)]
    with pytest.raises(ValueError):
        gp.parse_openfermion("(0.5j) [X0]")


def test_hubbard_two_sites_by_hand():
    terms = gp.fermi_hubbard(2, 1, t=1.0, u=4.0)
    assert len(terms) == 10 and dict(terms)["IIIZ"] == -1.0 and dict(terms)["IIZZ"] == 1.0
    H = gp._sparse(terms).to_matrix()
    # half filling ground state of the two-site Hubbard model: u/2 - sqrt(u^2/4 + 4t^2), minus the dropped offset 2
    ev = np.linalg.eigvalsh(H)
    assert min(abs(ev - (2 - np.sqrt(8) - 2))) < 1e-9
    assert len(gp.heisenberg(3, 1)) == 6
