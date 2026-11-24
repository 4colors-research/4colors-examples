import examplekit
import plot
import solve_tsp


def test_solve_tsp_end_to_end(canned, tmp_path):
    solve_tsp.main(["--out", str(tmp_path / "t"), "--sizes", "4", "--lam", "0.01", "2", "--seeds", "1",
                    "--time-limit", "1", "--size-table-max", "20"])
    rows = examplekit.Results(tmp_path / "t" / "results.jsonl").rows()
    sizes = [r for r in rows if r["kind"] == "size"]
    assert {r["n"] for r in sizes} == {5, 10, 20}
    ref = [r for r in rows if r["kind"] == "reference"][0]
    assert ref["method"].startswith("held-karp")
    solves = {(r["arm"], r["lam_factor"]): r for r in rows if r["kind"] == "solve"}
    ex = solves[("exact", None)]
    # canned all-zeros answers visit no city: every check rejects them, and the checks agree
    assert ex["feasible"] is False and ex["service_constraints"]["feasible"] is False and ex["n_rows"] == 8
    assert ex["energy_python"] == ex["solver_objective"]
    assert not solves[("penalty", 0.01)]["feasible"] and not solves[("penalty", 2.0)]["feasible"]
    plot.main([str(tmp_path / "t"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "tsp_feasibility.png").exists() and (tmp_path / "fig" / "tsp_model_size.png").exists()


def test_tour_answers_give_the_encoding_figure_and_no_reference_above_exact_max(canned, tmp_path):
    import tsp

    canned.solution = lambda call: tsp.encode([0, 2, 1, 3]) if call["n"] == 16 else [0] * call["n"]
    solve_tsp.main(["--out", str(tmp_path / "t"), "--sizes", "4", "5", "--exact-max", "4", "--lam", "2",
                    "--seeds", "1", "--time-limit", "1", "--size-table-max", "10"])
    rows = examplekit.Results(tmp_path / "t" / "results.jsonl").rows()
    assert [r["n"] for r in rows if r["kind"] == "reference"] == [4]          # exact optimum only where computed
    assert [r["n"] for r in rows if r["kind"] == "instance"] == [5]
    four = [r for r in rows if r["kind"] == "solve" and r["n"] == 4]
    assert all(r["feasible"] and r["tour"] == [0, 2, 1, 3] and r["gap"] >= 0 for r in four)
    assert all(r["gap"] is None for r in rows if r["kind"] == "solve" and r["n"] == 5)
    assert all(r["routing_input"] == r["n"] * (r["n"] - 1) // 2 for r in rows if r["kind"] == "size")
    ex, pts = plot.encoding_example(rows)
    assert ex["n"] == 4 and len(pts) == 4
    s = plot.summarize(rows)
    assert s["exact rows"] == [(4, 1.0, 1), (5, 0.0, 1)]
    plot.main([str(tmp_path / "t"), "--out", str(tmp_path / "fig")])
    for f in ("tsp_model_size.png", "tsp_feasibility.png", "tsp_encoding.png"):
        assert (tmp_path / "fig" / f).exists()


class _CannedJob:
    """A fixed, canned failed job (as JobFailed carries it): no service is reached."""

    class _Error:
        code = "model_too_large"

        def __str__(self):
            return "model_too_large: the model is over the plan's limit"

    def __init__(self):
        self.id = "canned-refused-1"
        self.error = self._Error()
        self.result = {"status": "refused", "limit": {"max_vars": 20},
                       "reason": {"code": "model_too_large", "message": "over the plan's limit", "field": "model"}}


def test_refused_and_failed_jobs_are_rows_and_the_run_goes_on(canned, monkeypatch, tmp_path):
    import fourcolors.qubo
    from fourcolors.errors import InvalidRequest, JobFailed

    solved = fourcolors.qubo.solve                     # the canned solve the fixture installed

    def refusing(m, *a, **kw):
        if m.n > 20 and m.m:                            # n = 5, exact rows: a failed (refused) job
            raise JobFailed(_CannedJob())
        if m.n > 20:                                    # n = 5, penalties: the request itself refused
            raise InvalidRequest("canned refusal: over the plan's limit", code="limit_exceeded", field="model")
        return solved(m, *a, **kw)

    monkeypatch.setattr(fourcolors.qubo, "solve", refusing)
    solve_tsp.main(["--out", str(tmp_path / "t"), "--sizes", "4", "5", "--lam", "2", "--seeds", "1", "2",
                    "--time-limit", "1", "--size-table-max", "5"])
    rows = examplekit.Results(tmp_path / "t" / "results.jsonl").rows()
    solves = [r for r in rows if r["kind"] == "solve"]
    assert len(solves) == 2 * 2 * 2                    # every (n, arm, seed) has a row: nothing dropped
    four = [r for r in solves if r["n"] == 4]
    assert all(r.get("status") != "refused" and "refusal_code" not in r for r in four)
    five = {r["arm"]: r for r in solves if r["n"] == 5 and r["seed"] == 1}
    ex, pen = five["exact"], five["penalty"]
    assert ex["status"] == "refused" and ex["refusal_code"] == "model_too_large"
    assert ex["refusal_field"] == "model" and ex["job_id"] == "canned-refused-1"
    assert ex["limit"] == {"max_vars": 20} and ex["billed_s"] == 0.0
    assert pen["status"] == "refused" and pen["refusal_code"] == "limit_exceeded" and pen["job_id"] is None
    for r in (ex, pen):
        assert r["feasible"] is None and r["tour"] is None and r["n_vars"] == 25 and r["reference"] is not None
    assert plot.refusals(rows) == {("exact rows", 5): 2, ("penalty, lam = 2 x max distance", 5): 2}
    assert [t[0] for t in plot.summarize(rows)["exact rows"]] == [4]   # refusals are not answers
    plot.main([str(tmp_path / "t"), "--out", str(tmp_path / "fig")])


def test_errors_not_about_the_model_are_raised(canned, monkeypatch, tmp_path):
    import fourcolors.qubo
    import pytest
    from fourcolors.errors import AuthenticationError

    def no_key(*a, **kw):
        raise AuthenticationError("canned: no key", code="unauthorized", status=401)

    monkeypatch.setattr(fourcolors.qubo, "solve", no_key)
    with pytest.raises(AuthenticationError):
        solve_tsp.main(["--out", str(tmp_path / "t"), "--sizes", "4", "--lam", "2", "--seeds", "1",
                        "--time-limit", "1", "--size-table-max", "5"])
