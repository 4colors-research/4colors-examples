import itertools

import pytest

import examplekit
import maxcut_model
import plot
import stop_rule
import watch


class FakeClock:
    """A clock that advances only when the code sleeps."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_model_energy_is_minus_cut():
    edges, m = maxcut_model.build(10, seed=2)
    for bits in itertools.islice(itertools.product((0, 1), repeat=10), 0, 1024, 7):
        assert m.energy(list(bits)) == -maxcut_model.cut(edges, bits)


def test_patience_stops_after_the_last_gain(canned):
    _, m = maxcut_model.build(60, seed=3)
    job = canned.submit(m, 30, seed=1)
    clock = FakeClock()
    final, why = stop_rule.run_with_patience(job, patience=3, min_gain=0.0, poll=1, clock=clock, sleep=clock.sleep)
    assert why == "no_progress" and final.stopped and final.solution is not None
    # the canned job shows one fixed incumbent from the first poll on; the rule stops 3 polls later and keeps it
    assert clock.t == pytest.approx(3)
    assert final.objective == m.energy(job.x)


def test_job_that_ends_first_is_returned_as_is(canned):
    _, m = maxcut_model.build(60, seed=3)
    job = canned.submit(m, 30, seed=1)
    job.result()                                      # the job has ended before the first peek
    clock = FakeClock()
    final, why = stop_rule.run_with_patience(job, patience=3, min_gain=0.0, poll=1, clock=clock, sleep=clock.sleep)
    assert why == "ended" and not final.stopped


def test_ctrl_c_keeps_the_best(canned):
    _, m = maxcut_model.build(60, seed=3)
    job = canned.submit(m, 30, seed=1)
    calls = {"n": 0}

    def sleep(_):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt

    final, why = stop_rule.run_with_patience(job, patience=99, min_gain=0.0, poll=1, sleep=sleep)
    assert why == "interrupted" and final.stopped and final.solution is not None


def test_watch_and_stop_scripts_end_to_end(canned, tmp_path):
    watch.main(["--out", str(tmp_path / "w"), "--n", "40", "--time-limit", "10"])
    stop_rule.main(["--out", str(tmp_path / "s"), "--n", "40", "--time-limit", "10", "--patience", "0",
                    "--poll", "0"])
    w = examplekit.Results(tmp_path / "w" / "results.jsonl").rows()
    s = examplekit.Results(tmp_path / "s" / "results.jsonl").rows()
    inc = [r for r in w if r["kind"] == "incumbent"]
    fin = [r for r in w if r["kind"] == "final"][0]
    assert inc and all(b["cut"] >= a["cut"] for a, b in zip(inc, inc[1:]))       # each incumbent is better
    assert fin["cut"] == -fin["energy_python"] == -fin["solver_objective"] and fin["billed_s"] == 10
    sf = [r for r in s if r["kind"] == "final"]
    assert len(sf) == 2 and all(r["why"] == "no_progress" and r["stopped"] for r in sf)
    assert all(r["billed_s"] < r["time_limit_s"] for r in sf)
    plot.main([str(tmp_path / "w"), str(tmp_path / "s"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "billed.png").exists()


def test_cadence_summary(tmp_path):
    import json

    import cadence

    rows = [{"kind": "final", "job_id": "j1", "arm": "watch", "seed": 1,
             "trajectory": [[0.04, -10], [1.72, -12], [3.40, -13], [5.10, -14]]},
            {"kind": "poll", "job_id": "j1"},
            {"kind": "final", "job_id": "j2", "arm": "watch", "seed": 2, "trajectory": []}]
    s = cadence.summary(rows[0])
    assert s["points"] == 4 and s["first_s"] == 0.04
    assert abs(s["median_gap_s"] - 1.68) < 1e-9 and abs(s["max_gap_s"] - 1.70) < 1e-9
    assert cadence.summary(rows[2])["median_gap_s"] is None
    assert cadence.gaps([{"t_search_s": 1.0}, {"t_search_s": 2.5}]) == [1.5]
    p = tmp_path / "results.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert cadence.main([str(p)]) == 0
