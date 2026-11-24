"""Offline tests: the two generators, the evaluator, the 4CQB file, the selection rule and run.py / plot.py with
fixed canned answers for upload() and solve() (nothing is solved)."""
import itertools
import json
import types

import numpy as np
import pytest

import examplekit as bk
import dense as D
import fourcolors
import fourcolors.qubo
import plot
import run


def explicit_energy(inst, x):
    """An independent re-derivation of the energy with plain Python loops."""
    e = 0
    for a, b, c in zip(inst["i"].tolist(), inst["j"].tolist(), inst["c"].tolist()):
        e += c * x[a] * x[b]
    if inst["linear"] is not None:
        e += sum(q * xi for q, xi in zip(inst["linear"].tolist(), x))
    return e


def test_reg_is_sorted_simple_and_regular_before_merging():
    inst = D.reg_instance(2000, 16, seed=5)
    i, j, c = inst["i"], inst["j"], inst["c"]
    assert np.all(i < j)
    key = i * inst["n"] + j
    assert np.all(np.diff(key) > 0)                                  # sorted, no repeats
    assert set(np.unique(c).tolist()) <= {-2, -1, 1, 2}              # +-1, or two merged +-1 of the same sign
    deg = np.bincount(np.concatenate([i, j]), weights=np.abs(np.concatenate([c, c])), minlength=inst["n"])
    assert deg.max() <= 16 and abs(deg.mean() - 16) < 0.1           # merged-away pairs only lower it
    assert inst["total_weight"] == int(c.sum())


def test_reg_is_deterministic_by_seed():
    a, b, c = D.reg_instance(500, 4, 1), D.reg_instance(500, 4, 1), D.reg_instance(500, 4, 2)
    assert D.sha256(a) == D.sha256(b) != D.sha256(c)


def test_reg_refuses_odd_n():
    with pytest.raises(ValueError):
        D.reg_instance(501, 4, 1)


def test_full_counts_ranges_and_block_independence():
    n = 600
    inst = D.full_instance(n, seed=3)
    assert len(inst["i"]) == n * (n - 1) // 2 == D.expected_terms("full", n)
    assert np.all(inst["i"] < inst["j"]) and np.all(np.diff(inst["i"] * n + inst["j"]) > 0)
    assert np.all(inst["c"] != 0) and np.abs(inst["c"]).max() <= 100
    assert np.abs(inst["linear"]).max() <= 100
    r, c, v = D._full_block(n, 3, 256, 512)                           # one block on its own equals its slice
    sel = (inst["i"] >= 256) & (inst["i"] < 512)
    assert np.array_equal(r, inst["i"][sel]) and np.array_equal(v, inst["c"][sel])
    assert D.sha256(inst) == D.sha256(D.full_instance(n, 3)) != D.sha256(D.full_instance(n, 4))


@pytest.mark.parametrize("inst", [D.reg_instance(8, 3, 1), D.full_instance(6, 1)], ids=["reg", "full"])
def test_energy_matches_explicit_loops_on_every_answer(inst):
    vals = (-1, 1) if inst["vartype"] == "SPIN" else (0, 1)
    for x in itertools.product(vals, repeat=inst["n"]):
        assert D.energy(inst, x) == explicit_energy(inst, x)


def test_cut_is_w_minus_h_over_two():
    inst = D.reg_instance(10, 3, 2)
    rng = np.random.default_rng(0)
    s = rng.choice([-1, 1], size=10)
    cut = sum(c for a, b, c in zip(inst["i"], inst["j"], inst["c"]) if s[a] != s[b])
    assert D.cut(inst, s) == cut


def test_energy_refuses_wrong_domain():
    with pytest.raises(ValueError):
        D.energy(D.full_instance(6, 1), [-1] * 6)


@pytest.mark.parametrize("inst", [D.reg_instance(200, 16, 1), D.full_instance(80, 1)], ids=["reg", "full"])
def test_file_round_trips_through_the_client_reader(tmp_path, inst):
    path = tmp_path / "m.4cqb.zst"
    info = D.write_file(inst, path, chunk=1000)                       # several slices
    m = fourcolors.read_file(str(path))
    assert m.n == inst["n"] and info["nnz"] == len(inst["i"])
    rng = np.random.default_rng(1)
    vals = [-1, 1] if inst["vartype"] == "SPIN" else [0, 1]
    for _ in range(5):
        x = rng.choice(vals, size=inst["n"])
        assert m.energy(x.tolist()) == D.energy(inst, x)


def test_size_report_for_degree_256():
    rep = D.size_report("reg", 100_000, 256)
    assert rep["couplings"] == 12_800_000 and rep["matrix_nonzeros"] == 25_600_000
    assert rep["file_bytes_raw"] == 40 + 800_000 + 16 * 12_800_000


def _row(arm, T, inst, seed, e, **kw):
    return {"event": "solve", "arm": arm, "budget_s": T, "instance": inst, "family": "reg", "degree": 16,
            "n": 100, "solver_seed": seed, "energy": e, **kw}


def test_selection_rule_keeps_only_clear_same_time_wins():
    rows = []
    for seed in (1, 2, 3):
        rows += [_row("4colors", 10, "a", seed, -1000), _row("neal", 10, "a", seed, -990)]         # 1 % win
        rows += [_row("4colors", 1, "a", seed, -999), _row("neal", 1, "a", seed, -999.2)]          # tie: out
        rows += [_row("4colors", 3, "a", seed, -995), _row("openjij", 3, "a", seed, -1000, over_budget=True),
                 _row("neal", 3, "a", seed, -980)]                                                 # overrun dropped
    rows += [_row("4colors", 60, "a", 1, None), _row("neal", 60, "a", 1, -900)]                    # no answer: out
    table, n_cells = plot.select_wins(rows)
    assert n_cells == 4
    got = {(t["budget_s"], t["best_rival"]) for t in table}
    assert got == {(10.0, "neal"), (3.0, "neal")}
    t10 = next(t for t in table if t["budget_s"] == 10)
    assert t10["ours_gap"] == 0 and abs(t10["rival_gap"] - 0.01) < 1e-12 and not t10["needs_10_seeds"]


def test_close_cell_needs_10_seeds():
    rows = []
    for seed in (1, 2, 3):
        rows += [_row("4colors", 10, "a", seed, -100000), _row("neal", 10, "a", seed, -99930)]     # 0.07 points
    table, _ = plot.select_wins(rows)
    assert table and table[0]["needs_10_seeds"]


class FakeClient:
    def license(self):
        return {"solvers": [{"id": "optimizer.qubo", "model_limits": {"max_terms": None}}]}


def _fake_api(monkeypatch, refuse_upload=False):
    held = {}

    def fake_upload(path, client=None):
        if refuse_upload:
            raise fourcolors.errors.InvalidRequest("too large for the test", code="model_too_large")
        held["m"] = fourcolors.read_file(path)                       # the upload must be a valid 4CQB file
        return "id-test"

    def fake_solve(input_id=None, time_limit_s=None, seed=None, client=None):
        m = held["m"]
        x = np.ones(m.n, dtype=int)                                   # canned: a fixed answer, nothing solved
        job = types.SimpleNamespace(id="canned-1", billed_s=time_limit_s, run_s=time_limit_s, wait_s=0.0)
        return types.SimpleNamespace(status="finished", reason=None, objective=m.energy(x.tolist()),
                                     solution=x.tolist(), job=job, engine={"version": "canned"},
                                     trajectory=[{"seq": 0, "t_search_s": 0.1, "objective": 0}])

    monkeypatch.setattr(fourcolors.qubo, "upload", fake_upload)
    monkeypatch.setattr(fourcolors.qubo, "solve", fake_solve)


def test_run_and_plot_end_to_end_offline(tmp_path, monkeypatch):
    _fake_api(monkeypatch)
    out = tmp_path / "out"
    rows = run.main(["--out", str(out), "--family", "reg", "--n", "200", "--degrees", "4", "16",
                     "--instance-seeds", "1", "--budgets", "1", "3", "--solver-seeds", "1", "2"], client=FakeClient())
    solves = [r for r in rows if r["event"] == "solve"]
    assert len(solves) == 2 * 2 * 2 and all(r["agrees"] for r in solves)
    assert all(r["engine_version"] == "canned" for r in solves)
    assert rows[0]["event"] == "license" and rows[0]["model_limits"] == {"max_terms": None}
    assert not list((out / "scratch").glob("*.zst"))                 # deleted after the upload
    plot.main([str(out / "results.jsonl"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "gap_by_budget.png").exists() and (tmp_path / "fig" / "wins.md").exists()
    json.dumps(rows)


def test_probe_sends_the_largest_model_once(tmp_path, monkeypatch):
    _fake_api(monkeypatch)
    rows = run.main(["--out", str(tmp_path / "p"), "--family", "reg", "--n", "200", "--degrees", "4", "16",
                     "--instance-seeds", "7", "8", "--probe"], client=FakeClient())
    solves = [r for r in rows if r["event"] == "solve"]
    assert len(solves) == 1 and solves[0]["degree"] == 16 and solves[0]["budget_s"] == 1.0 and solves[0]["probe"]


def test_refused_upload_is_recorded(tmp_path, monkeypatch):
    _fake_api(monkeypatch, refuse_upload=True)
    rows = run.main(["--out", str(tmp_path / "r"), "--family", "full", "--sizes", "50", "--instance-seeds", "1",
                     "--probe"], client=FakeClient())
    ref = [r for r in rows if r["event"] == "refused"]
    assert len(ref) == 1 and ref[0]["error_code"] == "model_too_large" and ref[0]["stage"] == "upload"
    assert not list((tmp_path / "r" / "scratch").glob("*.zst"))


def test_plan_only_needs_no_client(tmp_path):
    plan = run.main(["--out", str(tmp_path), "--family", "reg", "--plan-only"], client=None)
    assert [p["couplings"] for p in plan] == [200_000, 800_000, 3_200_000, 12_800_000]


def test_rival_harness_records_rescored_rows(tmp_path):
    """The rival plumbing on a tiny instance (dwave-samplers Tabu, 50 ms): rows carry our own re-scored energy."""
    import rivals

    rows = rivals.main(["--out", str(tmp_path / "rv"), "--family", "reg", "--n", "20", "--degrees", "3",
                        "--instance-seeds", "1", "--budgets", "0.05", "--solver-seeds", "1",
                        "--settings", "tabu"])
    (r,) = rows
    inst = D.reg_instance(20, 3, 1)
    assert r["event"] == "solve" and r["instance_sha256"] == D.sha256(inst)
    assert r["energy"] == r["reported_energy"]                         # the sampler's number agrees with ours


def test_rival_without_harness_stops(tmp_path):
    import rivals

    with pytest.raises(SystemExit):
        rivals.main(["--out", str(tmp_path), "--family", "reg", "--settings", "sb"])
