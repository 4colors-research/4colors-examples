import json

import networkx as nx
import numpy as np
from qiskit.quantum_info import Statevector

from pathlib import Path

from qaoakit import core

POST = Path(__file__).resolve().parents[1]


import postload  # noqa: E402  (blog/tests/postload.py; blog/tests is on sys.path from conftest)


def load(name):
    """Import <post dir>/<name>.py under a name unique to this post (blog/tests/postload.py), so posts with a
    run.py or plot.py of their own never collide when their tests run in one session."""
    return postload.load(POST, name)


run = load("run")
plot = load("plot")


def reg3(n, s=1):
    g = nx.random_regular_graph(3, n, seed=s)
    return dict(id=f"reg3-n{n}-s{s}", kind="maxcut", nodes=n, edges=[(int(a), int(b), 1.0) for a, b in sorted(g.edges)])


def m3c(n, s=1):
    g = nx.gnp_random_graph(n, 0.8, seed=s)
    return dict(id=f"m3c-n{n}-s{s}", kind="max3cut", nodes=n, edges=[(int(a), int(b), 1.0) for a, b in sorted(g.edges)])


def test_p0_expectation_is_the_uniform_average():
    """gamma = beta = 0: the state is uniform, so the expected cut is the brute-force average (|E|/2)."""
    inst = reg3(10)
    cost = core.cost_vector(inst)
    assert abs(np.abs(core.qaoa_statevector(10, cost, [0.0, 0.0])) ** 2 @ cost - cost.mean()) < 1e-9
    assert abs(cost.mean() - len(inst["edges"]) / 2) < 1e-9


def test_fixed_angles_on_the_tree_light_cone():
    """The tabulated p = 1, 2 angles give the tree cut fractions 0.6925 and 0.7559 on the 3-regular tree."""
    edges, nxt, frontier = [(0, 1, 1.0)], 2, [0, 1]
    for _ in range(2):
        new = []
        for v in frontier:
            for _ in range(2):
                edges.append((v, nxt, 1.0))
                new.append(nxt)
                nxt += 1
        frontier = new
    cost = core.maxcut_cost(nxt, edges)
    for p, want in ((1, 0.6925), (2, 0.7559)):
        probs = np.abs(core.qaoa_statevector(nxt, cost, core.fixed_angles(p))) ** 2
        assert abs((1 - core.zz_from_probs(nxt, probs, [(0, 1)])[0]) / 2 - want) < 1e-4


def test_xy_mixer_keeps_the_one_hot_subspace():
    inst = m3c(3)
    qc, th = core.build(inst, "xy-p2")
    probs = Statevector(qc.assign_parameters([0.7, 0.4, -0.3, 1.1])).probabilities()
    idx = np.arange(len(probs))
    onehot = np.ones(len(probs), bool)
    for v in range(3):
        onehot &= (((idx >> 3 * v) & 1) + ((idx >> 3 * v + 1) & 1) + ((idx >> 3 * v + 2) & 1)) == 1
    assert abs(probs[onehot].sum() - 1) < 1e-9


def test_rqaoa_recovers_the_optimum_on_tiny_graphs():
    for inst in (reg3(10, 1), reg3(10, 2), dict(nodes=10, edges=[(i, (i + 1) % 10, 1.0) for i in range(10)])):
        cost = core.maxcut_cost(inst["nodes"], inst["edges"])
        x, steps = core.rqaoa(inst["nodes"], inst["edges"], stop_at=6, seed=7)
        assert cost[core.bits_index(x)] == cost.max() and len(steps) == inst["nodes"] - 6


def test_parametric_warm_start_equals_the_bound_one():
    inst = reg3(8)
    x0 = core.local_optimum(inst)
    a, th = core.build(inst, "warm-p1", x0)
    b, tb, tw = core.build_warm_parametric(inst, 1)
    sa = Statevector(a.assign_parameters(dict(zip(th, [0.4, 0.9])))).data
    sb = Statevector(b.assign_parameters({**dict(zip(tb, [0.4, 0.9])), **dict(zip(tw, core.warm_thetas(x0)))})).data
    assert abs(abs(np.vdot(sa, sb)) - 1) < 1e-9
    cost = core.cost_vector(inst)
    assert np.allclose(np.abs(sa) ** 2, np.abs(core.qaoa_statevector(8, cost, [0.4, 0.9], x0)) ** 2)


def test_best_of_shots():
    cost = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([0.25] * 4)
    assert abs(core.best_of_shots(p, cost, 1) - 1.5) < 1e-12
    assert abs(core.best_of_shots(p, cost, 2) - (3 * 7 / 16 + 2 * 5 / 16 + 1 * 3 / 16)) < 1e-12


def test_zoo_end_to_end_offline(canned, tmp_path):
    run.main(["--out", str(tmp_path), "--seeds", "1", "--budget", "0.01", "--trajectories", "4", "--batches", "2",
              "--e2", "1e-2", "--qubo-limits", "0.01", "--qubo-seeds", "1", "--ma-maxiter", "20",
              "--rqaoa-noisy-seeds", "1"], instances=[reg3(10), m3c(3)])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    cl = [r for r in rows if r["kind"] == "classical"]
    assert len(cl) == 2 and all(r["classical_ratio"] is not None for r in cl)   # canned all-zeros answer
    q = [r for r in rows if r["kind"] == "qaoa"]
    names = {r["variant"] for r in q}
    assert {"std-p1", "std-p3", "fixed-p2", "ma-p1", "warm-local-p1", "warm-qubo-p1", "xy-p2"} <= names
    scored = [r for r in q if r.get("simulated")]
    assert scored and all(r["check_ok"] for r in scored)
    assert all(r["noisy_ratio_mean"] < r["ideal_ratio"] + 0.05 for r in scored)
    assert not [r for r in q if r["variant"] == "rqaoa-p1"]        # never priced at the first step's circuit
    var = [r for r in rows if r["kind"] == "variant" and r["variant"] == "rqaoa-p1"]
    assert var and var[0]["pricing"] == "recursion simulated classically"
    rq = [r for r in rows if r["kind"] == "rqaoa_noisy"]
    assert len(rq) == 2 and all(r["noisy_ratio_mean"] is not None and r["two_qubit_gates_total"] > 0 for r in rq)
    for r in rq:                                                    # priced over all steps
        steps = len(r["rqaoa_steps"])
        assert steps == 10 - 8 and r["compiles"] == steps and len(r["two_qubit_gates_per_step"]) == steps
        assert r["two_qubit_gates"] == sum(r["two_qubit_gates_per_step"]) == r["two_qubit_gates_total"]
        assert r["shots_total"] == steps * 1000 and r["pricing"] == "all recursion steps"
    plot.main(["--out", str(tmp_path)])
    for f in ("table.md", "table-full.md", "no_price.md", "selection.md", "zoo.png"):
        assert (tmp_path / f).exists(), f
    assert "gates per point" in (tmp_path / "table-full.md").read_text()
    assert "recursion simulated classically" in (tmp_path / "no_price.md").read_text()


def _row(variant, arm, noisy, gates=100, kind="qaoa", **kw):
    return dict(kind=kind, variant=variant, arm=arm, e2=3e-3, noisy_ratio_mean=noisy, two_qubit_gates=gates,
                random_ratio=0.5, simulated=True, **kw)


def test_price_and_no_price_reasons():
    assert plot.price(_row("std-p1", "4colors", 0.75)) == 100 / 25
    assert plot.price(_row("std-p1", "4colors", 0.5)) is None
    assert plot.no_price_reason(_row("std-p1", "4colors", 0.75)) is None
    assert plot.no_price_reason(_row("std-p1", "4colors", 0.4)) == "noisy ratio at or below random"
    refused = dict(kind="qaoa", variant="std-p1", arm="4colors", refusal_code="input_too_large", random_ratio=0.5)
    assert plot.no_price_reason(refused) == "refused"
    wide = dict(kind="qaoa", variant="std-p1", arm="4colors", simulated=False, two_qubit_gates=500, random_ratio=0.5)
    assert plot.no_price_reason(wide) == "too wide to simulate"
    rq = _row("rqaoa-p1", "4colors", None, gates=None, kind="rqaoa_noisy")
    assert plot.no_price_reason(rq) == "recursion step refused"


def test_no_price_rows_are_counted_not_dropped(tmp_path):
    rows = [dict(kind="variant", variant="std-p1", ideal_ratio=0.9),
            _row("std-p1", "4colors", 0.8, gates=50), _row("std-p1", "4colors", 0.78, gates=50),
            _row("std-p1", "4colors", 0.45, gates=50),
            dict(kind="qaoa", variant="std-p1", arm="4colors", refusal_code="x", random_ratio=0.5),
            _row("std-p1", "qiskit-o3", 0.7, gates=80),
            _row("std-p2", "4colors", 0.6), _row("std-p2", "qiskit-o3", 0.65)]          # a loss: left out
    c = plot.cells(rows, 3e-3)
    x = c[("std-p1", "4colors")]
    assert x["rows"] == 4 and x["priced"] == 2
    assert x["no_price"]["noisy ratio at or below random"] == 1 and x["no_price"]["refused"] == 1
    assert plot.select_variants(c) == ["std-p1"]
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    plot.main(["--out", str(tmp_path)])
    t = (tmp_path / "table.md").read_text()
    assert "std-p1" in t and "std-p2" not in t and " | 2 |" in t and "Shown:" not in t
    assert "std-p2" in (tmp_path / "selection.md").read_text()
    assert "| std-p1 | 4colors Quantum Compiler | 4 | 1 | 0 | 1 | 0 |" in (tmp_path / "no_price.md").read_text()


def test_sampled_correlations():
    rng = np.random.default_rng(0)
    probs = np.array([0.5, 0.0, 0.0, 0.5])                          # Z0 Z1 = +1 with certainty
    assert run.sampled_zz(2, probs, [(0, 1)], 100, rng) == [1.0]
    assert run.sampled_zz(2, np.full(4, 0.25), [(0, 1)], 0, rng) == [0.0]
