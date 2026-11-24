import itertools
import json
import os

import numpy as np
import pytest

import native as nv

POST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import postload  # noqa: E402  (blog/tests/postload.py; blog/tests is on sys.path from conftest)


def load(name):
    """Import <post dir>/<name>.py under a name unique to this post (blog/tests/postload.py), so posts with a
    run.py or plot.py of their own never collide when their tests run in one session."""
    return postload.load(POST_DIR, name)


# a hand-made "device": a 6-qubit ring, coupler (2, 3) noisy
RING = {(0, 1): 0.01, (1, 2): 0.01, (2, 3): 0.2, (3, 4): 0.01, (4, 5): 0.01, (0, 5): 0.01}


def test_problems_are_seeded_and_weighted():
    e, w = nv.problem("regular3", 20, 20)
    assert len(e) == 30 and set(w.values()) == {1} and nv.problem("regular3", 20, 20)[0] == e
    e, w = nv.problem("weighted_er", 50, 50)
    assert all(1 <= v <= 10 for v in w.values())


def test_qubo_energy_is_minus_the_cut_with_isolated_nodes():
    edges, w = [(0, 1), (1, 2)], {(0, 1): 3, (1, 2): 2}
    m = nv.qubo_model(4, edges, w)          # node 3 isolated: still a variable
    assert m.n == 4
    for x in itertools.product((0, 1), repeat=4):
        assert m.energy(list(x)) == pytest.approx(-nv.cut(edges, w, x))


def test_a_ring_problem_embeds_fully_on_a_ring_device():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    w = {e: 1 for e in edges}
    place = nv.native_placement(6, edges, w, 6, RING, seed=1, iters=3000)
    assert sorted(place) == list(range(6))                       # distinct qubits
    assert nv.native_edges(place, edges, RING) == edges          # every edge on a coupler


def test_native_keeps_a_subset_and_prefers_good_couplers():
    # a triangle cannot sit on a ring: at most 2 of its 3 edges land on couplers
    edges, w = [(0, 1), (1, 2), (0, 2)], {(0, 1): 1, (1, 2): 1, (0, 2): 1}
    place = nv.native_placement(3, edges, w, 6, RING, seed=2, iters=4000)
    kept = nv.native_edges(place, edges, RING)
    assert len(kept) == 2 and set(kept) <= set(edges)
    used = {tuple(sorted((place[a], place[b]))) for a, b in kept}
    assert (2, 3) not in used                                     # calibration-aware: the noisy coupler avoided


def test_placement_score_counts_only_edges_on_couplers():
    edges, w = [(0, 1), (0, 2)], {(0, 1): 2, (0, 2): 5}
    assert nv.placement_score([0, 1, 3], edges, w, RING, aware=False) == 2
    assert nv.placement_score([0, 1, 3], edges, w, RING, aware=True) == pytest.approx(2 * 0.99)


def test_random_and_spectral_sparsifiers_keep_k_distinct_edges():
    e, w = nv.problem("weighted_er", 30, 3)
    for es in (nv.random_edges(e, 10, 1), nv.spectral_edges(30, e, w, 10, 1)):
        assert len(es) == len(set(es)) == 10 and set(es) <= set(e)
    assert nv.spectral_edges(30, e, w, 10, 1) == nv.spectral_edges(30, e, w, 10, 1)


def test_effective_resistance_on_hand_made_graphs():
    # a path of two unit edges: each edge is a bridge, R = 1; a triangle: R = 2/3 per edge
    assert np.allclose(nv.effective_resistance(3, [(0, 1), (1, 2)], {(0, 1): 1, (1, 2): 1}), [1, 1])
    tri = [(0, 1), (1, 2), (0, 2)]
    assert np.allclose(nv.effective_resistance(3, tri, {e: 1 for e in tri}), [2 / 3] * 3)


def test_native_circuit_sits_on_couplers_and_counts():
    edges = [(0, 1), (1, 2)]
    w = {e: 1 for e in edges}
    qc = nv.qaoa_circuit(6, edges, w, p=2, qubit_of=[3, 4, 5], nodes=range(3))
    assert qc.num_qubits == 6 and qc.count_ops()["rzz"] == 4 and qc.count_ops()["measure"] == 3
    assert nv.on_couplers(qc, RING)
    bad = nv.qaoa_circuit(6, [(0, 2)], {(0, 2): 1}, p=1, qubit_of=[0, 1, 2], nodes=range(3))
    assert not nv.on_couplers(bad, RING)


def test_device_graph_drops_failed_couplers():
    import circuitkit as ck

    full = ck.heavy_hex_coupler_sets()[0]["full"]
    n, couplers = nv.device_graph(full)
    failed = ck.failed_couplers(full)
    assert n == 156 and not (set(couplers) & failed)
    assert len(couplers) + len(failed) == len(ck.device_couplers(full))


def test_run_and_plot_plumbing_with_canned_answers(canned, canned_compiler, tmp_path):
    run = load("run")
    plot = load("plot")
    canned.solution = lambda call: [i % 2 for i in range(call["n"])]
    run.main(["--out", str(tmp_path / "r"), "--problems", "reg3_20", "--seeds", "1", "--time-limit", "1",
              "--budgets", "1", "--layers", "1", "--max-qiskit-trials", "1"])
    rows = [json.loads(l) for l in (tmp_path / "r" / "results.jsonl").read_text().splitlines()]
    var = {r["variant"]: r for r in rows if r["kind"] == "variant"}
    k = var["native"]["kept_edges"]
    assert var["full"]["kept_edges"] == 30 and var["random"]["kept_edges"] == var["spectral"]["kept_edges"] == k
    solves = [r for r in rows if r["kind"] == "solve"]
    assert {r["variant"] for r in solves} == set(var) and all(r["exact_full_cut"] for r in solves)
    assert all(r["cut_on_full"] >= r["cut_on_variant"] for r in solves)    # the full graph has every kept edge
    res = [r for r in rows if r["kind"] == "result"]
    assert {(r["variant"], r["arm"]) for r in res} == {("full", "4colors"), ("full", "qiskit"), ("native", "4colors"),
                                                       ("random", "4colors"), ("spectral", "4colors")}
    plot.main([str(tmp_path / "r"), "--figs", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / "selection.md").exists() and (tmp_path / "figs" / "fig_tradeoff.png").exists()


def test_layer_selection_shows_only_strict_wins():
    plot = load("plot")
    g = {("a", "full", 1, "4colors"): 10, ("a", "full", 1, "qiskit"): 12,
         ("a", "full", 2, "4colors"): 20, ("a", "full", 2, "qiskit"): 20,
         ("b", "full", 1, "4colors"): 5}
    shown, both = plot.layer_cells(g)
    assert shown == [("a", 1)] and both == [("a", 1), ("a", 2)]
