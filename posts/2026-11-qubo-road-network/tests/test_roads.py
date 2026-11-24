"""Offline tests: graph conversion, QUBO forms against brute force, the balance row, the baseline, run.py."""
import itertools
import types

import networkx as nx
import numpy as np
import pytest

import examplekit as bk
import draw_map
import fourcolors.qubo as fq
import plot
import roads
import run


def toy_roads(rows=3, cols=4):
    """An osmnx-like road graph: both directions, one parallel road, one loop road, coordinates, lengths."""
    G = nx.MultiDiGraph(crs="epsg:4326")
    for r in range(rows):
        for c in range(cols):
            G.add_node(1000 + r * cols + c, x=4.0 + 0.01 * c, y=52.0 + 0.01 * r)
    for r in range(rows):
        for c in range(cols):
            a = 1000 + r * cols + c
            for b in ([a + 1] if c + 1 < cols else []) + ([a + cols] if r + 1 < rows else []):
                G.add_edge(a, b, length=100.4 + c); G.add_edge(b, a, length=100.4 + c)
    G.add_edge(1000, 1001, length=50.0)          # a parallel, shorter road
    G.add_edge(1005, 1005, length=30.0)          # a loop road
    return G


def test_conversion_merges_directions_parallel_roads_and_loops():
    G = toy_roads()
    g = roads.to_simple(G, "unit")
    assert g["n"] == 12 and len(g["u"]) == 3 * 3 + 2 * 4      # grid edges only
    assert np.all(g["u"] < g["v"]) and set(g["w"].tolist()) == {1}
    gl = roads.to_simple(G, "length")
    assert gl["w"][0] == 50                                    # shortest parallel road, whole meters
    assert gl["w"].dtype == np.int64


def test_graphml_round_trip(tmp_path):
    ox = pytest.importorskip("osmnx")

    G = toy_roads()
    ox.save_graphml(G, tmp_path / "toy.graphml")
    H = roads.load_graph(graphml=tmp_path / "toy.graphml")
    assert roads.to_simple(H)["n"] == 12


def test_download_needs_permission():
    pytest.importorskip("osmnx")
    with pytest.raises(SystemExit):
        roads.load_graph(place="Nowhere")


def test_maxcut_model_equals_cut_and_brute_force():
    g = roads.to_simple(toy_roads(), "length")
    m = roads.maxcut_model(g)
    assert m.sense == "max" and m.integer
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 2, g["n"])
        assert m.energy(x) == roads.cut_value(g, x)
    edges = list(zip(g["u"].tolist(), g["v"].tolist()))
    wts = dict(zip(edges, g["w"].tolist()))
    best, _ = bk.brute_force_maxcut(g["n"], edges, wts)
    assert best == max(m.energy(np.array(b)) for b in itertools.product((0, 1), repeat=g["n"]))


def test_bisection_row_and_brute_force():
    g = roads.to_simple(toy_roads(), "unit")
    m, (lo, hi) = roads.bisection_model(g, imbalance=0.0)
    assert (lo, hi) == (6, 6) and len(m.constraints) == 1
    best = min(roads.cut_value(g, np.array(b)) for b in itertools.product((0, 1), repeat=12) if sum(b) == 6)
    assert best == 3                                           # a 3 x 4 grid splits across its 3 rows
    x = roads.geometric_split(g, lo, hi)
    assert m.feasible(x) and x.sum() == 6 and roads.cut_value(g, x) == best


def test_local_search_improves_and_keeps_balance():
    G = nx.grid_2d_graph(20, 20)
    G = nx.MultiDiGraph(nx.relabel_nodes(G, {v: i for i, v in enumerate(G.nodes)}))
    for i, v in enumerate(G.nodes):
        G.nodes[v]["x"], G.nodes[v]["y"] = float(i % 20), float(i // 20)
    g = roads.to_simple(G)
    x0 = np.random.default_rng(1).integers(0, 2, g["n"])
    x, rounds, _ = roads.local_search(g, x0, maximize=True, time_limit_s=5)
    assert roads.cut_value(g, x) > roads.cut_value(g, x0)
    s = 2 * x - 1                                              # a local optimum: no single flip helps
    A, _ = roads._adjacency(g)
    assert np.all(s * (A @ s) <= 0)
    lo, hi = roads.balance_bounds(g["n"], 0.0)
    xr = np.random.default_rng(2).permutation(np.r_[np.zeros(200, int), np.ones(200, int)])
    xb, _, _ = roads.local_search(g, xr, maximize=False, time_limit_s=5, balanced=True)
    assert xb.sum() == 200 and roads.cut_value(g, xb) < roads.cut_value(g, xr)


class FakeClient:
    pass


@pytest.mark.parametrize("problem", ["maxcut", "bisection"])
def test_run_end_to_end_offline(tmp_path, monkeypatch, problem):
    held = {}

    def fake_upload(model, client=None):
        held["m"] = model
        return "id"

    def fake_solve(input_id=None, time_limit_s=None, seed=None, client=None):
        m = held["m"]
        x = np.zeros(m.n, dtype=int); x[: m.n // 2] = 1           # balanced, valid answer
        return types.SimpleNamespace(status="finished", reason=None, objective=m.energy(x), solution=x.tolist(),
                                     job=types.SimpleNamespace(id="j", billed_s=1.0, run_s=1.0, wait_s=0.0))

    monkeypatch.setattr(fq, "upload", fake_upload)
    monkeypatch.setattr(fq, "solve", fake_solve)
    out = tmp_path / "o"
    rows = run.main(["--out", str(out), "--graphml", "toy.graphml", "--problem", problem, "--budgets", "1",
                     "--imbalance", "0", "--baseline-time", "1"], client=FakeClient(), graph=toy_roads())
    solve = [r for r in rows if r["event"] == "solve"][0]
    assert solve["agrees"] and solve.get("balanced", True)
    base = [r for r in rows if r["event"] == "baseline"][0]
    assert base.get("balanced", True)
    if problem == "maxcut":
        assert base["cut"] >= base["start_cut"]
    t = plot.table(rows)
    assert {k[1] for k in t} == {"4colors", "local-search"}
    plot.main([str(out / "results.jsonl"), "--out", str(tmp_path / "f")])
    sols = out / "solutions"
    png = draw_map.draw(next(sols.glob("*-graph.npz")), next(sols.glob("*-4colors-*.npy")), tmp_path / "map.png")
    assert png.exists()


def _t(ours, local, problem="maxcut"):
    rows = []
    for solver, cuts in (("4colors", ours), ("local-search", local)):
        for T, cut in cuts.items():
            rows.append(dict(event="solve" if solver == "4colors" else "baseline", solver=solver, instance="i",
                             problem=problem, budget_s=T, cut=cut))
    return plot.table(rows)


def test_selection_rule_charts_only_strict_wins():
    t = _t({10: 90, 60: 100, 300: 101}, {10: 95, 60: 100, 300: 99})      # loss, tie, win
    assert plot.shown_budgets(t, "i") == ([300], True)
    t = _t({10: 40, 60: 30}, {10: 45, 60: 30}, problem="bisection")        # fewer crossings wins
    assert plot.shown_budgets(t, "i") == ([10], True)
    t = _t({10: 90, 60: 95}, {10: 99, 60: 99})                            # no win: ours alone, no local line
    assert plot.shown_budgets(t, "i") == ([10, 60], False)


def test_graph_over_the_cap_is_not_uploaded(tmp_path, monkeypatch):
    def no_upload(*a, **k):
        raise AssertionError("must not upload a graph over the cap")

    monkeypatch.setattr(fq, "upload", no_upload)
    rows = run.main(["--out", str(tmp_path / "o"), "--graphml", "toy.graphml", "--max-nodes", "5"],
                    client=FakeClient(), graph=toy_roads())
    assert rows[-1]["event"] == "skipped" and rows[-1]["n"] > 5


def test_arms_run_separately(tmp_path):
    rows = run.main(["--out", str(tmp_path / "o"), "--graphml", "toy.graphml", "--arms", "local-search",
                     "--budgets", "1"], client=FakeClient(), graph=toy_roads())
    assert {r["event"] for r in rows} == {"graph", "baseline"}
