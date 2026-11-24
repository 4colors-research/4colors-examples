"""The run and plot scripts on fixed canned answers (tests/canned.py): rows written, checks run, figures drawn.
Nothing is solved."""
import examplekit
import kings
import plot
import run


def test_end_to_end_with_a_hand_made_answer(canned, tmp_path):
    L = kings.side_for_atoms(30)
    pos, edges = kings.lattice(L, kings.FILLING, 1)
    n = len(pos)
    # A hand-made independent set: atoms on even rows and even columns (never adjacent in the King's graph).
    answer = [int(r % 2 == 0 and c % 2 == 0) for r, c in pos]
    assert kings.violations(answer, edges) == 0
    canned.solution = lambda call: answer if call["n"] == n else [0] * call["n"]
    run.main(["--out", str(tmp_path / "g"), "--atoms", "30", "--seeds", "1", "--time-limit", "1",
              "--no-license-check"])
    rows = examplekit.Results(tmp_path / "g" / "results.jsonl").rows()
    inst = [r for r in rows if r["kind"] == "instance"][0]
    assert inst["n"] == n and inst["exact_mis"] <= inst["block_bound"]
    solves = {r["arm"]: r for r in rows if r["kind"] == "solve"}
    rows_arm = solves["rows"]
    assert rows_arm["independent"] and rows_arm["model_feasible"] and rows_arm["service_feasible"]
    assert rows_arm["set_size"] == sum(answer) and rows_arm["engine_version"] == "canned"
    assert solves["penalty"]["independent"] and solves["penalty"]["service_feasible"] is None
    plot.main([str(tmp_path / "g"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "lattice.png").exists()
    assert (tmp_path / "fig" / "gap_vs_size.png").exists()


def test_a_violating_answer_is_repaired_and_flagged(canned, tmp_path):
    canned.solution = lambda call: [1] * call["n"]        # every atom: violates every edge
    run.main(["--out", str(tmp_path / "v"), "--atoms", "20", "--arms", "penalty", "--seeds", "1",
              "--no-license-check"])
    r = [r for r in examplekit.Results(tmp_path / "v" / "results.jsonl").rows() if r["kind"] == "solve"][0]
    assert r["violations"] > 0 and not r["independent"]
    assert r["set_size"] < r["set_size_raw"]


def test_selection_rule_drops_sizes_where_the_rival_is_not_behind():
    rows = []
    for n, ours, theirs in [(100, 0.01, 0.02), (1000, 0.03, 0.03), (10000, 0.02, 0.05)]:
        rows.append(dict(kind="solve", arm="rows", budget_s=1.0, n=n, gap_to_bound=ours, independent=True))
        rows.append(dict(kind="solve", arm="sa-fill", budget_s=1.0, n=n, gap_to_bound=theirs, independent=True))
    rows.append(dict(kind="solve", arm="rows", budget_s=1.0, n=50, gap_to_bound=0.0, independent=False))
    rows.append(dict(kind="solve", arm="sa-fill", budget_s=1.0, n=50, gap_to_bound=0.1, independent=True))
    assert plot.winning_sizes(rows) == {1.0: [100, 10000]}        # tie at 1000 and the dependent set at 50 dropped


def test_refused_job_is_recorded(canned, tmp_path, monkeypatch):
    import fourcolors
    import fourcolors.qubo

    def refuse(*a, **k):
        raise fourcolors.errors.InvalidRequest("refused for the test", code="model_too_large")

    monkeypatch.setattr(fourcolors.qubo, "solve", refuse)
    run.main(["--out", str(tmp_path / "x"), "--atoms", "20", "--arms", "rows", "--seeds", "1",
              "--no-license-check"])
    r = [r for r in examplekit.Results(tmp_path / "x" / "results.jsonl").rows() if r["kind"] == "solve"][0]
    assert r["error_code"] == "model_too_large"
