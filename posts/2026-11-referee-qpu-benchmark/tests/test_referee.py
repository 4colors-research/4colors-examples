"""Offline tests of the referee post: instances, Q-score arithmetic, the certificate, the QUBO model, the LR-QAOA
circuit, the tables and the plumbing. Nothing calls the service."""
import itertools
import json

import pytest

from conftest import load

R = load("referee")


def brute(n, edges, w=None):
    return max(R.cut_value(edges, x, w) for x in itertools.product((0, 1), repeat=n))


@pytest.mark.parametrize("n,seed", [(8, 1), (10, 2)])
def test_exact_maxcut_is_the_brute_force_optimum(n, seed):
    e = R.er_half(n, seed)
    assert R.exact_maxcut(n, e)[0] == brute(n, e)
    e, w = R.fc_weighted(n, seed)
    best, x = R.exact_maxcut(n, e, w, chunk=16)          # several chunks
    assert best == brute(n, e, w) == R.cut_value(e, x, w)


def test_model_energy_is_the_cut():
    e, w = R.fc_weighted(7, 3)
    m = R.maxcut_model(7, e, w)
    for x in itertools.product((0, 1), repeat=7):
        assert m.energy(list(x)) == R.cut_value(e, x, w)
    assert m.sense == "max" and m.m == 0


def test_qscore_arithmetic():
    n = 100
    assert R.qscore_beta(n * n / 8, n) == pytest.approx(0.0)
    assert R.qscore_beta(R.expected_optimum(n), n) == pytest.approx(1.0)
    assert R.beta_needed(n, R.expected_optimum(n) + 0.178 * n ** 1.5) == pytest.approx(2.0)


def test_er_half_density():
    e = R.er_half(200, 1)
    assert abs(len(e) / (200 * 199 / 2) - 0.5) < 0.02


def test_lr_qaoa_structure_and_ramp():
    import circuitkit as ck

    gam, bet = R.lr_ramp(4)
    assert gam == sorted(gam) and bet == sorted(bet, reverse=True) and bet[-1] == 0
    e, w = R.fc_weighted(6, 1)
    qc = R.lr_qaoa(6, e, w, p=3)
    assert ck.circuit_metrics(qc)["two_qubit_gates"] == 3 * len(e) == 45


def test_tables_and_selection():
    plot = load("plot")
    rows = [{"kind": "instance", "family": "er", "n": 10, "instance_seed": 1, "qscore_reference": R.expected_optimum(10)},
            {"kind": "exact", "family": "er", "n": 10, "instance_seed": 1, "optimum": 20},
            {"kind": "solve", "family": "er", "n": 10, "instance_seed": 1, "time_limit_s": 1.0, "cut": 20}]
    bk = plot.best_known(rows)
    v = bk[("er", 10, 1)]
    assert v["certified"] and v["reached_certificate"] and v["value"] == 20
    assert "yes (exhaustive)" in plot.beta_table(bk)
    base = {"kind": "result", "circuit_id": "c", "budget_s": 60.0, "seed": 1, "n": 16, "p": 3, "graph": "fc",
            "input_2q": 360}
    res = [{**base, "arm": "4colors", "two_qubit_gates": 900},
           {**base, "arm": "qiskit-o3-best", "two_qubit_gates": 1200, "over_budget": False},
           {**base, "arm": "qiskit-o3-once", "two_qubit_gates": 1300}]
    cb = plot.compile_cells(res)
    assert cb[("c", 60.0)]["shown"] and "| fc | 16 | 3 |" in plot.compile_table(cb)
    res[0]["two_qubit_gates"] = 1190
    assert not plot.compile_cells(res)[("c", 60.0)]["shown"]


def test_run_and_plot_offline(canned, canned_qubo, tmp_path, monkeypatch):
    run, plot = load("run"), load("plot")
    monkeypatch.setattr(run, "COMPILE", (("fc", 6, 1),))
    out = tmp_path / "ref"
    run.main(["--out", str(out), "--er-sizes", "10", "--fc-sizes", "8", "--instances", "1", "--time-limits", "1",
              "--seeds", "1", "--budgets", "0.2", "--max-qiskit-trials", "1", "--qiskit-streams", "1"])
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in rows} >= {"instance", "exact", "solve", "circuit", "result"}
    solve = [r for r in rows if r["kind"] == "solve"]
    assert all(r["cut"] == 0 for r in solve)                     # canned all-zeros: one side, cut 0
    plot.main([str(out), "--figs", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / "fig_beta.png").exists()
    assert "Compile cells shown:" in (tmp_path / "figs" / "selection.md").read_text()
