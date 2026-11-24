"""The run and plot scripts on fixed canned answers (tests/canned.py): rows written, checks run, figures drawn.
Nothing is solved."""
import examplekit
import plot
import run
import traffic


def test_end_to_end_on_a_grid_with_a_hand_made_answer(canned, tmp_path):
    G, _ = traffic.grid_city(10, 10)
    cars = traffic.random_cars(G, 30, 1)
    routes = [traffic.candidate_routes(G, s, t, 3) for s, t in cars]
    br, _ = traffic.best_response(routes, traffic.all_shortest(routes))
    answer = traffic.encode(routes, br)                    # hand-picked: the best-response choice, encoded
    canned.solution = lambda call: answer
    run.main(["--out", str(tmp_path / "t"), "--graph", "grid:10x10", "--cars", "30", "--seeds", "1",
              "--time-limit", "1"])
    rows = examplekit.Results(tmp_path / "t" / "results.jsonl").rows()
    inst = [r for r in rows if r["kind"] == "instance"][0]
    assert inst["n_rows"] == 30 and not inst["skipped_over_max_terms"]
    sol = [r for r in rows if r["kind"] == "solve"][0]
    assert sol["valid"] and sol["model_feasible"] and sol["service_feasible"]
    assert sol["congestion"] == sol["energy_python"] == traffic.congestion(routes, br)
    methods = {r["method"] for r in rows if r["kind"] == "baseline"}
    assert methods == {"shortest", "random", "best-response"}
    plot.main([str(tmp_path / "t"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "map.png").exists()


def test_all_zeros_is_invalid_and_count_only_sends_nothing(canned, tmp_path):
    run.main(["--out", str(tmp_path / "z"), "--graph", "grid:8x8", "--cars", "10", "--seeds", "1"])
    sol = [r for r in examplekit.Results(tmp_path / "z" / "results.jsonl").rows() if r["kind"] == "solve"][0]
    assert sol["valid"] is False and sol["model_feasible"] is False
    n_calls = len(canned.calls)
    run.main(["--out", str(tmp_path / "c"), "--graph", "grid:8x8", "--cars", "10", "20", "--count-only"])
    assert len(canned.calls) == n_calls
    insts = [r for r in examplekit.Results(tmp_path / "c" / "results.jsonl").rows() if r["kind"] == "instance"]
    assert [r["cars"] for r in insts] == [10, 20]


def test_over_max_terms_is_skipped_and_recorded(canned, tmp_path):
    run.main(["--out", str(tmp_path / "s"), "--graph", "grid:8x8", "--cars", "20", "--max-terms", "1"])
    rows = examplekit.Results(tmp_path / "s" / "results.jsonl").rows()
    assert [r for r in rows if r["kind"] == "instance"][0]["skipped_over_max_terms"]
    assert not [r for r in rows if r["kind"] == "solve"] and not canned.calls


def test_selection_rule():
    rows = [dict(kind="solve", arm="4colors", budget_s=1.0, cars=c, congestion=v, valid=True)
            for c, v in [(418, 90), (4180, 100), (10000, 80)]]
    rows += [dict(kind="baseline", method="best-response", cars=c, congestion=v)
             for c, v in [(418, 95), (4180, 100), (10000, 85)]]
    rows += [dict(kind="baseline", method="shortest", cars=c, congestion=200) for c in (418, 4180, 10000)]
    assert plot.winning_cells(rows) == [(1.0, 418), (1.0, 10000)]       # the tie at 4180 is dropped
