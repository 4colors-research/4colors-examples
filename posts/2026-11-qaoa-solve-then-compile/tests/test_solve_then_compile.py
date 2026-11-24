import itertools
import json

import numpy as np

from pathlib import Path

from qaoakit import apikit as ak, core

POST = Path(__file__).resolve().parents[1]


import postload  # noqa: E402  (blog/tests/postload.py; blog/tests is on sys.path from conftest)


def load(name):
    """Import <post dir>/<name>.py under a name unique to this post (blog/tests/postload.py), so posts with a
    run.py or plot.py of their own never collide when their tests run in one session."""
    return postload.load(POST, name)


run = load("run")
plot = load("plot")


def test_qubo_sign_convention():
    """x^T Q x = -cut(x) for every assignment: the optimizer's minimum is minus the maximum cut."""
    inst = run.reg3(8, 2)
    Q = ak.maxcut_matrix(8, inst["edges"])
    cost = core.cost_vector(inst)
    for bits in itertools.product((0, 1), repeat=8):
        x = np.array(bits)
        assert abs(x @ Q @ x + cost[core.bits_index(bits)]) < 1e-9


def test_success_and_swap_network():
    cost = np.array([0.0, 1.0, 2.0, 2.0])
    p = np.array([0.5, 0.3, 0.1, 0.1])
    s, q = run.success(p, cost, 2.0, 3)
    assert abs(q - 0.2) < 1e-12 and abs(s - (1 - 0.8 ** 3)) < 1e-12
    assert run.swap_network_two_qubit_gates(4) == 18


def test_threshold_interpolates_and_reports_never():
    assert plot.threshold([(1e-3, 0.9), (1e-2, 0.1)]) is not None
    t = plot.threshold([(1e-3, 0.9), (1e-2, 0.1)])
    assert 1e-3 < t < 1e-2 and abs(t - 10 ** -2.5) < 1e-9          # halfway in log(e2)
    assert plot.threshold([(1e-3, 0.4), (1e-2, 0.1)]) is None       # never reaches 1/2
    assert plot.threshold([(1e-3, 0.9), (1e-2, 0.6)]) == 1e-2       # above 1/2 everywhere: the last point


def _q(arm, p, inst, noisy, e2=1e-3, ang="optimized", s100=None, s1000=None, gates=10):
    r = dict(kind="qaoa", simulated=True, arm=arm, p=p, instance=inst, angles=ang, e2=e2, ideal_ratio=0.9,
             noisy_ratio_mean=noisy, two_qubit_gates=gates, random_ratio=0.5, classical_ratio=1.0)
    if s100 is not None:
        r.update(success_100=s100, success_1000=s1000, random_success_100=0.1, random_success_1000=0.6)
    return r


def test_ratio_selection_shows_only_strict_wins():
    q = [_q("4colors", 1, "a", 0.80), _q("qiskit-o3", 1, "a", 0.70),      # win
         _q("4colors", 2, "a", 0.75), _q("qiskit-o3", 2, "a", 0.75),      # tie: left out
         _q("4colors", 3, "a", 0.60), _q("qiskit-o3", 3, "a", 0.65)]      # loss: left out
    by_arm = {arm: plot.ratio_by_p(q, arm, 1e-3) for arm in plot.ARMS}
    assert plot.select_ps(by_arm) == [1]
    assert plot.ideal_by_p(q) == {1: 0.9, 2: 0.9, 3: 0.9}


def test_ratio_by_p_is_median_over_seeds_then_mean_over_graphs():
    q = [_q("4colors", 1, "a", x) for x in (0.1, 0.5, 0.9)] + [_q("4colors", 1, "b", 0.7)]
    assert abs(plot.ratio_by_p(q, "4colors", 1e-3)[1] - 0.6) < 1e-12


def test_breakeven_cells_cover_both_shot_counts_and_select():
    q = []
    for e2, f, k in ((1e-4, 0.9, 0.9), (1e-3, 0.8, 0.3), (1e-2, 0.1, 0.1)):
        q += [_q("4colors", 1, "a", 0.8, e2=e2, s100=f, s1000=f), _q("qiskit-o3", 1, "a", 0.7, e2=e2, s100=k, s1000=k)]
    cells = plot.breakeven_cells(q, [100, 1000])
    assert {c["S"] for c in cells} == {100, 1000}
    assert all(c["shown"] and c["4colors"] > c["qiskit-o3"] for c in cells)
    flipped = [dict(r, arm={"4colors": "qiskit-o3", "qiskit-o3": "4colors"}[r["arm"]]) for r in q]
    assert not any(c["shown"] for c in plot.breakeven_cells(flipped, [100]))


def test_end_to_end_offline(canned, tmp_path):
    run.main(["--out", str(tmp_path), "--compile-seeds", "1", "--seeds", "1", "--budget", "0.01", "--ps", "1,2",
              "--deep-ps", "3", "--deep-n", "8", "--e2-sweep", "1e-3,1e-2", "--shots", "10,100",
              "--trajectories", "4", "--batches", "2", "--qubo-limits", "0.01"],
             cert=[run.reg3(8, 1)], large=[])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    kinds = {r["kind"] for r in rows}
    assert kinds == {"optimizer", "qaoa"}                        # the later panels run only on request
    q = [r for r in rows if r["kind"] == "qaoa" and r.get("simulated")]
    # p = 1, 2 and the deep p = 3 x (optimized, fixed) x 2 arms x 2 noise levels
    assert len(q) == 3 * 2 * 2 * 2 and {r["p"] for r in q} == {1, 2, 3}
    lo = [r for r in q if r["e2"] == 1e-3]
    hi = [r for r in q if r["e2"] == 1e-2]
    assert np.mean([r["noisy_ratio_mean"] for r in lo]) > np.mean([r["noisy_ratio_mean"] for r in hi])
    assert all(0 <= r["success_10"] <= 1 and 0 <= r["success_100"] <= 1 for r in q)
    assert all(r["random_success_10"] is not None and r["random_success_100"] is not None for r in q)
    assert all("reference_is_optimum" in r for r in q)
    plot.main(["--out", str(tmp_path)])
    for f in ("optimizer.png", "ratio_vs_p.png", "breakeven.png", "breakeven.md", "cost.md", "selection.md"):
        assert (tmp_path / f).exists(), f
    assert not (tmp_path / "warm.md").exists()


def test_later_panels_still_run(canned, tmp_path):
    run.main(["--out", str(tmp_path), "--panels", "density,warm", "--compile-seeds", "1", "--seeds", "1",
              "--budget", "0.01", "--shots", "10", "--trajectories", "4", "--batches", "2",
              "--density-n", "8", "--densities", "0.3,0.9", "--qubo-limits", "0.01"], cert=[run.reg3(8, 1)], large=[])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert {"density_compile", "density_qubo", "warm_qaoa", "warm_classical"} <= {r["kind"] for r in rows}
    plot.main(["--out", str(tmp_path)])
    assert (tmp_path / "density.md").exists() and (tmp_path / "warm.md").exists()
