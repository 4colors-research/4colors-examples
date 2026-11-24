"""The scripts end to end with canned all-zeros results (no API, nothing solved): rows, checks, plots."""
import examplekit
import exact_rows
import plot
import sweep_penalty


def test_sweep_then_exact_then_plot(canned, tmp_path):
    sweep_penalty.main(["--out", str(tmp_path / "pen"), "--time-limit", "1", "--seeds", "1", "2",
                        "--assignment-n", "3", "--knapsack-n", "8", "--factors", "0.01", "10"])
    exact_rows.main(["--out", str(tmp_path / "ex"), "--time-limit", "1", "--seeds", "1",
                     "--assignment-n", "3", "--knapsack-n", "8"])
    pen = examplekit.Results(tmp_path / "pen" / "results.jsonl").rows()
    ex = examplekit.Results(tmp_path / "ex" / "results.jsonl").rows()
    assert len([r for r in pen if r["kind"] == "penalty"]) == 2 * 2 * 2
    assert len(ex) == 2
    # the knapsack penalty model has slack bits; the exact one does not
    kp = [r for r in pen if r["kind"] == "penalty" and r["problem"] == "knapsack"]
    ke = [r for r in ex if r["problem"] == "knapsack"]
    assert kp[0]["n_vars"] > ke[0]["n_vars"] == 8 and ke[0]["n_rows"] == 1
    for r in ex:
        # the canned answer is all zeros: the checks must agree with each other on it, whatever it is worth
        assert r["checks_agree"]
        assert r["model_energy_python"] == r["solver_objective"]
        assert r["job_id"] and r["billed_s"] is not None
    for r in pen:
        if r["kind"] == "penalty":
            assert r["model_energy_python"] == r["solver_objective"]
            assert abs(r["penalty_energy_python"] - r["solver_objective"]) < 1e-6
    empty = [r for r in pen if r["problem"] == "assignment" and r["kind"] == "penalty"]
    assert not any(r["feasible"] for r in empty)     # the all-zeros assignment covers no task
    assert {c["seed"] for c in canned.calls} == {1, 2}
    plot.main([str(tmp_path / "pen"), str(tmp_path / "ex"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "assignment_n3_penalty_sweep.png").stat().st_size > 0
    assert (tmp_path / "fig" / "feasibility_vs_lambda.png").stat().st_size > 0
    table = (tmp_path / "fig" / "cost_of_search.md").read_text()
    assert "| knapsack | 8 | exact | 1 | 8 | 0 | 1 | 0 | none needed |" in table
