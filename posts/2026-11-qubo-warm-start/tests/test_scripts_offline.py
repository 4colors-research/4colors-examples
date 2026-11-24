import examplekit
import plot
import warm_vs_cold


def test_warm_vs_cold_end_to_end(canned, tmp_path):
    # canned answers: the warm start sent back unchanged, else a hand-made alternating split
    canned.solution = lambda call: call["warm_start"] or [i % 2 for i in range(call["n"])]
    warm_vs_cold.main(["--out", str(tmp_path / "w"), "--n", "14", "--days", "2", "--time-limit", "1",
                       "--seeds", "1", "2", "--day0-time-limit", "1"])
    rows = examplekit.Results(tmp_path / "w" / "results.jsonl").rows()
    solves = [r for r in rows if r["kind"] == "solve"]
    assert len(solves) == 1 + 2 * 2 * 2
    assert len([r for r in rows if r["kind"] == "carry_over"]) == 2
    for r in solves:
        assert r["cut"] == -r["energy_python"] == -r["solver_objective"]
    # warm jobs got yesterday's answer; cold jobs got none
    warm_calls = [c for c in canned.calls if c["warm_start"] is not None]
    assert len(warm_calls) == 4 and all(len(c["warm_start"]) == 14 for c in warm_calls)
    plot.main([str(tmp_path / "w"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "warm_progress.png").exists()
    plot.main([str(tmp_path / "w"), "--out", str(tmp_path / "fig2"), "--quality", "0.005", "--tmax", "0.5"])
    assert (tmp_path / "fig2" / "warm_progress.png").exists()


def test_times_to_quality_uses_day_best():
    rows = [{"kind": "solve", "day": 1, "arm": "cold", "cut": 10, "trajectory": [[0.5, -9], [2.0, -10]]},
            {"kind": "solve", "day": 1, "arm": "warm", "cut": 10, "trajectory": [[0.0, -10]]}]
    assert plot.times_to_quality(rows, 0.0) == {"cold": [2.0], "warm": [0.0]}
    assert plot.times_to_quality(rows, 0.1) == {"cold": [0.5], "warm": [0.0]}


def test_first_points_verdicts(tmp_path):
    import json

    import first_points

    carry = {"kind": "carry_over", "day": 1, "energy_python": -50, "cut": 50}
    listed = {"kind": "solve", "day": 1, "arm": "warm", "seed": 1, "trajectory": [[0.0, -50], [1.7, -52]]}
    improved = {"kind": "solve", "day": 1, "arm": "warm", "seed": 2, "trajectory": [[0.3, -51]]}
    cold = {"kind": "solve", "day": 1, "arm": "cold", "seed": 1, "trajectory": [[0.0, -50]]}
    assert first_points.check([carry, listed, cold])["verdict"] == "warm start listed at t=0"
    assert first_points.check([carry, improved])["verdict"] == "first point is an improvement"
    assert first_points.check([carry, listed, improved])["verdict"] == "mixed"
    assert first_points.check([carry, cold])["verdict"] == "no warm jobs"
    p = tmp_path / "r.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in [carry, listed]))
    assert first_points.main([str(p)]) == 0


def test_first_points_on_offline_run(canned, tmp_path):
    import first_points

    warm_vs_cold.main(["--out", str(tmp_path / "w"), "--n", "12", "--days", "1", "--time-limit", "1",
                       "--seeds", "1", "--day0-time-limit", "1"])
    rep = first_points.check(examplekit.Results(tmp_path / "w" / "results.jsonl").rows())
    assert rep["warm_jobs"] == 1 and rep["verdict"] != "no warm jobs"
