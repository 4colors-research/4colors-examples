"""Unit tests of the MIS model code on hand-made inputs. Nothing here calls the service.

    python -m pytest -q tests
"""
import itertools

import pytest

import examplekit
import kings


def brute_mis(n, edges):
    best = 0
    for mask in range(1 << n):
        if all(not (mask >> u & 1 and mask >> v & 1) for u, v in edges):
            best = max(best, bin(mask).count("1"))
    return best


def test_full_3x3_kings_graph():
    pos, edges = kings.lattice(3, filling=1.0, seed=1)
    assert len(pos) == 9
    assert len(edges) == 20                     # 6 horizontal + 6 vertical + 8 diagonal
    assert brute_mis(9, edges) == 4             # the four corners
    assert kings.block_bound(pos) == 4          # blocks {0,1}x{0,1}, {0,1}x{2}, {2}x{0,1}, {2}x{2}


def test_arrays_match_lists():
    for L, f, s in [(5, 0.8, 1), (7, 0.5, 3), (10, 0.8, 9)]:
        pos, edges = kings.lattice(L, f, s)
        pa, ea = kings.lattice_arrays(L, f, s)
        assert [tuple(p) for p in pa.tolist()] == pos
        assert [tuple(e) for e in ea.tolist()] == edges


def test_bound_and_exact_on_random_small_lattices():
    for seed in range(1, 6):
        pos, edges = kings.lattice(4, 0.8, seed)
        n = len(pos)
        exact = brute_mis(n, edges)
        assert kings.exact_mis(n, edges) == exact
        assert exact <= kings.block_bound(pos)


def test_violations_and_repair():
    edges = [(0, 1), (1, 2), (2, 3)]
    assert kings.violations([1, 0, 1, 0], edges) == 0
    assert kings.violations([1, 1, 1, 0], edges) == 2
    y = kings.repair([1, 1, 1, 1], edges)
    assert kings.violations(y, edges) == 0 and y == [1, 0, 1, 0]


def test_rows_model_admits_exactly_the_independent_sets():
    pos, edges = kings.lattice(3, 0.8, 2)
    n = len(pos)
    m = kings.rows_model(n, edges)
    assert m.m == len(edges) and m.sense == "max"
    for bits in itertools.product([0, 1], repeat=n):
        assert bool(m.feasible(list(bits))) == (kings.violations(bits, edges) == 0)
        if m.feasible(list(bits)):
            assert m.energy(list(bits)) == sum(bits)


def test_penalty_model_has_the_same_optimum():
    pos, edges = kings.lattice(3, 0.9, 4)
    n = len(pos)
    m = kings.penalty_model(n, edges, P=2)
    best = min(m.energy(list(b)) for b in itertools.product([0, 1], repeat=n))
    assert -best == brute_mis(n, edges)
    for bits in itertools.product([0, 1], repeat=n):
        assert m.energy(list(bits)) == -sum(bits) + 2 * kings.violations(bits, edges)


def test_side_and_memory_estimate_for_a_million_atoms():
    L = kings.side_for_atoms(1_000_000)
    assert L == 1118                            # 0.8 * 1118^2 = 999,939 expected atoms
    assert examplekit.service_memory_gb(1_000_000) < 8   # rule of thumb; the job's own check decides


def test_calibrated_sweeps():
    import rival

    assert rival.calibrated_sweeps(0.001, 1.0) == 1000
    assert rival.calibrated_sweeps(10.0, 1.0) == 1
    assert rival.calibrated_sweeps(0.0, 1.0) == 1
