import itertools

import pytest

import examplekit
import daily_maxcut as dm


def test_next_day_changes_exactly_the_share():
    edges, w = dm.base_graph(40, 3, seed=5)
    w1 = dm.next_day(edges, w, 1, 0.05, seed=5)
    changed = [e for e in edges if w1[e] != w[e]]
    assert len(changed) == round(0.05 * len(edges)) and all(1 <= w1[e] <= 10 for e in edges)
    assert dm.next_day(edges, w, 1, 0.05, seed=5) == w1          # deterministic


def test_days_chain_and_graph_fixed():
    ds = dm.days(30, 3, 4, 0.1, seed=2)
    assert [d for d, _, _ in ds] == [0, 1, 2, 3, 4]
    assert all(e == ds[0][1] for _, e, _ in ds)


def test_model_energy_is_minus_cut():
    edges, w = dm.base_graph(10, 3, seed=1)
    m = dm.model(10, edges, w)
    for bits in itertools.product((0, 1), repeat=10):
        assert m.energy(list(bits)) == -dm.cut(edges, w, bits)


def test_brute_force_reference_matches_enumeration():
    edges, w = dm.base_graph(12, 3, seed=3)
    best, x = examplekit.brute_force_maxcut(12, edges, w)
    m = dm.model(12, edges, w)
    assert best == max(-m.energy(list(b)) for b in itertools.product((0, 1), repeat=12))
    assert dm.cut(edges, w, x) == best


def test_time_to_target():
    traj = [{"t_search_s": 0.1, "objective": -5}, {"t_search_s": 0.4, "objective": -8},
            {"t_search_s": 1.2, "objective": -9}]
    assert examplekit.time_to_target(traj, -8) == 0.4
    assert examplekit.time_to_target([[0.1, -5], [0.4, -8]], -9) is None
    assert examplekit.time_to_target(traj, 4, sense="max") is None
