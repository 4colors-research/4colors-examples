import itertools
import random

import pytest

import tsp


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_held_karp_against_brute_force(n):
    _, d = tsp.random_instance(n, seed=n)
    best, tour = tsp.held_karp(d)
    assert best == tsp.brute_force(d) == tsp.tour_length(d, tour)
    assert sorted(tour) == list(range(n))


@pytest.mark.parametrize("n", [3, 4, 5, 7])
def test_model_size_formula_matches_the_build(n):
    _, d = tsp.random_instance(n, seed=1)
    for penalty, m in ((False, tsp.exact_model(d)), (True, tsp.penalty_model(d, 10.0))):
        assert tsp.model_size(n, penalty) == {"n_vars": m.n, "n_quad": m.nnz, "n_rows": m.m}


def test_energy_of_a_tour_is_its_length_in_both_models():
    _, d = tsp.random_instance(6, seed=3)
    ex, pen = tsp.exact_model(d), tsp.penalty_model(d, 500.0)
    for perm in itertools.islice(itertools.permutations(range(6)), 50):
        x = tsp.encode(list(perm))
        assert tsp.decode(6, x) == list(perm)
        assert ex.feasible(x)
        assert ex.energy(x) == tsp.tour_length(d, list(perm))
        assert pen.energy(x) == pytest.approx(tsp.tour_length(d, list(perm)))


def test_penalty_matrix_matches_plain_python_on_every_vector():
    _, d = tsp.random_instance(3, seed=4)
    m = tsp.penalty_model(d, 7.5)
    for bits in itertools.product((0, 1), repeat=9):
        assert m.energy(list(bits)) == pytest.approx(tsp.penalty_energy(d, 7.5, list(bits)))


def test_exact_rows_admit_permutations_only():
    _, d = tsp.random_instance(3, seed=5)
    m = tsp.exact_model(d)
    feasible = [b for b in itertools.product((0, 1), repeat=9) if m.feasible(list(b))]
    assert len(feasible) == 6 and all(tsp.decode(3, b) is not None for b in feasible)


def test_penalty_samples_at_n5():
    _, d = tsp.random_instance(5, seed=6)
    m = tsp.penalty_model(d, 3.0)
    rnd = random.Random(0)
    for _ in range(200):
        x = [rnd.randint(0, 1) for _ in range(25)]
        assert m.energy(x) == pytest.approx(tsp.penalty_energy(d, 3.0, x))


def test_decode_rejects_non_permutations():
    assert tsp.decode(3, [1, 0, 0, 1, 0, 0, 0, 0, 1]) is None
    assert tsp.decode(2, [1, 1, 0, 0]) is None


def test_qubo_holds_2n_copies_of_the_routing_input():
    assert tsp.routing_input_size(5) == 10
    for n in (10, 100):
        # n^2 (n - 1) distance terms in the QUBO: 2n copies of the routing input
        assert tsp.model_size(n, False)["n_quad"] == 2 * n * tsp.routing_input_size(n)
