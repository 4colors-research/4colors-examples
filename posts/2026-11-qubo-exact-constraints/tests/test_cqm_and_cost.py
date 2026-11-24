"""The dimod CQM path and the cost-of-search facts the post states, offline (nothing is solved)."""
import pytest

import constrained_problems as cp

dimod = pytest.importorskip("dimod")


def test_knapsack_post_model_sizes():
    """The post's 40-item knapsack (instance seed 7): 9 slack bits and a fully dense 49-variable penalty matrix,
    against 40 variables, no quadratic terms and one row with an exact row. Model facts, not results."""
    w, v, cap = cp.knapsack_instance(40, 7)
    bits = cp.slack_bits(cap)
    assert len(bits) == 9 and sum(bits) == cap
    pen = cp.knapsack_penalty_model(w, v, cap, lam=1.0)
    assert (pen.n, pen.nnz) == (49, 49 * 48 // 2) == (49, 1176)
    ex = cp.knapsack_exact_model(w, v, cap)
    assert (ex.n, ex.nnz, ex.m) == (40, 0, 1)


@pytest.mark.parametrize("n", [12, 30, 40, 50])
def test_assignment_model_sizes(n):
    """Penalty: n^2 variables, n * n(n-1) quadratic terms (every pair inside each of 2n rows), no rows.
    Exact: n^2 variables, no quadratic terms, 2n rows."""
    c = cp.assignment_instance(n, 7)
    pen = cp.assignment_penalty_model(c, 1.0)
    assert (pen.n, pen.nnz, pen.m) == (n * n, n * n * (n - 1), 0)
    ex = cp.assignment_exact_model(c)
    assert (ex.n, ex.nnz, ex.m) == (n * n, 0, 2 * n)


def test_cqm_becomes_the_same_exact_rows():
    """The client turns the CQM into a model with rows (client-side, nothing sent); its rows admit exactly the
    answers the plain-Python judge calls feasible, and its energy is the original objective."""
    import itertools

    from fourcolors.optimizer.ocean import cqm_to_model
    import cqm_path

    c = cp.assignment_instance(3, 2)
    m = cqm_to_model(cqm_path.assignment_cqm(c))
    assert (m.n, m.m) == (9, 6)
    for bits in itertools.product((0, 1), repeat=9):
        x = list(bits)
        assert m.feasible(x) == cp.assignment_feasible(3, x)
        assert m.energy(x) == pytest.approx(cp.assignment_cost(c, x))
    w, v, cap = [3, 5, 4, 2, 6], [4, 7, 5, 3, 8], 9
    k = cqm_to_model(cqm_path.knapsack_cqm(w, v, cap))
    assert (k.n, k.m, k.nnz) == (5, 1, 0)
    for bits in itertools.product((0, 1), repeat=5):
        x = list(bits)
        assert k.feasible(x) == cp.knapsack_feasible(w, cap, x)
        assert k.energy(x) == pytest.approx(-cp.knapsack_value(v, x))


def test_vector_reads_labels_in_judge_order():
    import cqm_path

    sample = {f"x_{i}_{j}": int(i == j) for i in range(2) for j in range(2)}
    assert cqm_path.vector("assignment", 2, sample) == [1, 0, 0, 1]
    assert cqm_path.vector("knapsack", 3, {"x0": 1, "x1": 0, "x2": 1}) == [1, 0, 1]


def test_cqm_script_offline(canned, monkeypatch, tmp_path):
    """Script C end to end on a fixed canned answer: the identity assignment and the empty knapsack, both feasible;
    the three checks agree on them."""
    import examplekit
    import fourcolors.optimizer.ocean as ocean
    import cqm_path

    monkeypatch.setattr(ocean, "Client", lambda *a, **k: None)
    monkeypatch.setattr(ocean.qubo, "solve", canned.solve)
    n = 3
    canned.solution = lambda call: ([int(i == j) for i in range(n) for j in range(n)] if call["n"] == n * n
                                    else [0] * call["n"])
    cqm_path.main(["--out", str(tmp_path), "--seeds", "1", "2", "--assignment-n", str(n), "--knapsack-n", "6"])
    rows = examplekit.Results(tmp_path / "results.jsonl").rows()
    assert len(rows) == 4 and all(r["kind"] == "cqm" for r in rows)
    for r in rows:
        assert r["feasible"] and r["service_feasible"] and r["dimod_feasible"] and r["checks_agree"]
        assert r["job_id"].startswith("canned-")
    a = [r for r in rows if r["problem"] == "assignment"]
    assert all(r["objective"] == sum(cp.assignment_instance(n, 7)[i][i] for i in range(n)) for r in a)
    assert {c["time_limit_s"] for c in canned.calls} == {1.0}
