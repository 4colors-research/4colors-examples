import itertools

import pytest

import constrained_problems as cp


def all_vectors(n):
    for bits in itertools.product((0, 1), repeat=n):
        yield list(bits)


@pytest.mark.parametrize("lam", [0.5, 7.0, 120.0])
def test_assignment_penalty_matrix_matches_plain_python(lam):
    c = cp.assignment_instance(3, seed=1)
    m = cp.assignment_penalty_model(c, lam)
    assert m.n == 9 and m.m == 0
    for x in all_vectors(9):
        assert m.energy(x) == pytest.approx(cp.assignment_penalty_energy(c, lam, x))


@pytest.mark.parametrize("lam", [0.3, 4.0])
def test_knapsack_penalty_matrix_matches_plain_python(lam):
    w, v, cap = [3, 5, 4, 2], [4, 7, 5, 3], 7
    m = cp.knapsack_penalty_model(w, v, cap, lam)
    assert m.n == 4 + len(cp.slack_bits(cap))
    for x in all_vectors(m.n):
        assert m.energy(x) == pytest.approx(cp.knapsack_penalty_energy(w, v, cap, lam, x))


@pytest.mark.parametrize("cap", [1, 2, 7, 8, 13, 100])
def test_slack_bits_reach_exactly_zero_to_cap(cap):
    bits = cp.slack_bits(cap)
    reach = {sum(b for b, s in zip(bits, mask) if s) for mask in itertools.product((0, 1), repeat=len(bits))}
    assert reach == set(range(cap + 1))


def test_exact_models_rows_match_plain_python():
    c = cp.assignment_instance(3, seed=2)
    m = cp.assignment_exact_model(c)
    assert (m.n, m.m) == (9, 6)
    for x in all_vectors(9):
        assert m.feasible(x) == cp.assignment_feasible(3, x)
        assert m.energy(x) == cp.assignment_cost(c, x)
    w, v, cap = [3, 5, 4, 2, 6], [4, 7, 5, 3, 8], 9
    k = cp.knapsack_exact_model(w, v, cap)
    for x in all_vectors(5):
        assert k.feasible(x) == cp.knapsack_feasible(w, cap, x)
        assert k.energy(x) == -cp.knapsack_value(v, x)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_references_agree_with_brute_force(seed):
    c = cp.assignment_instance(6, seed)
    assert cp.assignment_optimum(c) == cp.assignment_brute_force(c)
    w, v, cap = cp.knapsack_instance(14, seed)
    assert cp.knapsack_optimum(w, v, cap) == cp.knapsack_brute_force(w, v, cap)


def test_strong_penalty_ground_state_is_the_optimum():
    """With lam large enough the penalty QUBO's minimum is feasible and optimal (checked by enumeration)."""
    c = cp.assignment_instance(3, seed=3)
    lam = 10 * cp.lam_scale("assignment", c)
    m = cp.assignment_penalty_model(c, lam)
    best = min(all_vectors(9), key=m.energy)
    assert cp.assignment_feasible(3, best) and cp.assignment_cost(c, best) == cp.assignment_optimum(c)


def test_weak_penalty_ground_state_is_infeasible():
    c = cp.assignment_instance(3, seed=3)
    m = cp.assignment_penalty_model(c, 0.01 * cp.lam_scale("assignment", c))
    assert not cp.assignment_feasible(3, min(all_vectors(9), key=m.energy))


def test_gap_signs():
    assert cp.gap("assignment", 110, 100) == pytest.approx(0.1)
    assert cp.gap("knapsack", 90, 100) == pytest.approx(0.1)
    assert cp.gap("knapsack", None, 100) is None
