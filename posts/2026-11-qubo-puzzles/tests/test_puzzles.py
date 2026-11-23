import itertools

import pytest

import puzzles as pz

SMALL_SUDOKU = "1.3..41.2..34.2."      # 4x4, a few blanks


def test_wikipedia_sudoku_reference_and_model():
    grid = pz.parse_grid(pz.WIKIPEDIA_PUZZLE)
    sol = pz.sudoku_backtrack(grid)
    assert pz.sudoku_valid(grid, sol)
    for eliminate in (True, False):
        m, variables = pz.sudoku_model(grid, eliminate=eliminate)
        x = [1 if sol[r][c] == d else 0 for r, c, d in variables]
        assert m.feasible(x) and m.energy(x) == 0
        assert pz.sudoku_decode(grid, variables, x) == sol
        x[next(k for k, xi in enumerate(x) if xi)] = 0          # drop one digit: a row breaks
        assert not m.feasible(x)
    full, _ = pz.sudoku_model(grid, eliminate=False)
    small, _ = pz.sudoku_model(grid, eliminate=True)
    assert full.n == 729 and small.n < full.n


def test_small_sudoku_every_feasible_assignment_is_the_solution():
    grid = pz.parse_grid(SMALL_SUDOKU)
    m, variables = pz.sudoku_model(grid)
    assert m.n <= 16
    feasible = [x for x in itertools.product((0, 1), repeat=m.n) if m.feasible(list(x))]
    assert len(feasible) == 1
    assert pz.sudoku_decode(grid, variables, feasible[0]) == pz.sudoku_backtrack(grid)


def test_sudoku_valid_rejects_bad_grids():
    grid = pz.parse_grid(pz.WIKIPEDIA_PUZZLE)
    sol = pz.sudoku_backtrack(grid)
    bad = [row[:] for row in sol]
    bad[0][2], bad[0][3] = bad[0][3], bad[0][2]
    assert not pz.sudoku_valid(grid, bad)


@pytest.mark.parametrize("n,count", [(4, 2), (5, 10)])
def test_nqueens_feasible_set_is_the_known_count(n, count):
    """4-queens has 2 solutions, 5-queens 10: the rows admit exactly those."""
    m = pz.nqueens_model(n)
    sols = []
    for cols in itertools.product(range(n), repeat=n):           # one queen per row, every column choice
        x = [0] * (n * n)
        for r, c in enumerate(cols):
            x[r * n + c] = 1
        if m.feasible(x):
            assert pz.nqueens_valid(n, pz.nqueens_decode(n, x))
            sols.append(cols)
    assert len(sols) == count


def test_nqueens_8_known_solution():
    cols = [0, 4, 7, 5, 2, 6, 1, 3]
    x = [1 if c == cols[r] else 0 for r in range(8) for c in range(8)]
    assert pz.nqueens_model(8).feasible(x) and pz.nqueens_valid(8, pz.nqueens_decode(8, x))
    assert not pz.nqueens_valid(8, [(r, r) for r in range(8)])


def test_us_graph_shape():
    nodes, edges = pz.us_edges()
    assert (len(nodes), len(edges)) == (49, 107)
    assert all(u in pz.US_ADJ[v].split() for u, v in edges)       # adjacency lists are symmetric


def test_coloring_objective_counts_conflicts():
    nodes, edges = ["a", "b", "c", "d"], [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]
    m = pz.coloring_model(nodes, edges, 3)
    for cols in itertools.product(range(3), repeat=4):
        x = [1 if c == cols[i] else 0 for i in range(4) for c in range(3)]
        colors = pz.coloring_decode(nodes, 3, x)
        assert m.feasible(x) and m.energy(x) == len(pz.coloring_conflicts(edges, colors))
    assert pz.min_conflicts(nodes, edges, 3) == 1 and pz.min_conflicts(nodes, edges, 4) == 0


def test_all_exact_coloring_rows():
    nodes, edges = ["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")]
    m2 = pz.coloring_model(nodes, edges, 2, all_exact=True)
    assert not any(m2.feasible(list(x)) for x in itertools.product((0, 1), repeat=6))   # a triangle needs 3
    m3 = pz.coloring_model(nodes, edges, 3, all_exact=True)
    assert m3.feasible([1, 0, 0, 0, 1, 0, 0, 0, 1])


def test_us_map_four_colors_needs_no_conflict():
    nodes, edges = pz.us_edges()
    assert pz.min_conflicts(nodes, edges, 4) == 0


def test_odd_wheel_needs_a_conflict_with_three_colors():
    rim = ["CA", "OR", "ID", "UT", "AZ"]                           # Nevada's neighbors form a 5-cycle
    edges = [("NV", s) for s in rim] + [tuple(sorted((rim[i], rim[(i + 1) % 5]))) for i in range(5)]
    assert pz.min_conflicts(["NV"] + rim, edges, 3) == 1
