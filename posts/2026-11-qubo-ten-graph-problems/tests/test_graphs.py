"""Offline tests: each model's optimum (by enumerating every assignment) equals the answer an independent
brute force on the graph gives, the validators reject bad answers, and run.py writes and checks its rows on canned all-zeros answers."""
import itertools
import json

import numpy as np
import pytest

nx = pytest.importorskip("networkx")

import graphs


def model_optimum(m):
    """(best energy, best x) over every feasible 0/1 assignment, by the model's own exact checks."""
    best, arg = None, None
    for bits in itertools.product((0, 1), repeat=m.n):
        x = np.array(bits, dtype=np.int8)
        if not m.feasible(x):
            continue
        e = m.energy(x)
        if best is None or (e > best if m.sense == "max" else e < best):
            best, arg = e, list(bits)
    return best, arg


def subsets(nodes):
    for k in range(len(nodes) + 1):
        yield from (set(c) for c in itertools.combinations(nodes, k))


def solved(prob):
    """The model optimum, decoded and checked: it must be valid and score exactly the energy."""
    e, x = model_optimum(prob.model)
    ans = prob.decode(x)
    v = prob.check(ans)
    assert v["valid"], (prob.name, ans)
    assert v["score"] == e, (prob.name, v, e)
    return v["score"], ans


def test_max_cut_matches_brute_force():
    G = nx.random_regular_graph(3, 8, seed=1)
    best = max(sum(1 for u, v in G.edges() if (u in S) != (v in S)) for S in subsets(list(G)))
    assert solved(graphs.max_cut(G))[0] == best


def test_independent_set_clique_and_cover():
    G = nx.cycle_graph(7)                                       # alpha(C7) = 3, clique 2, cover 4
    assert solved(graphs.max_independent_set(G))[0] == 3
    assert solved(graphs.max_clique(G))[0] == 2
    assert solved(graphs.min_vertex_cover(G))[0] == 4
    H = nx.gnp_random_graph(8, 0.5, seed=3)
    alpha = max(len(S) for S in subsets(list(H)) if all(not H.has_edge(u, v) for u, v in itertools.combinations(S, 2)))
    omega = max(len(S) for S in subsets(list(H)) if all(H.has_edge(u, v) for u, v in itertools.combinations(S, 2)))
    assert solved(graphs.max_independent_set(H))[0] == alpha
    assert solved(graphs.max_clique(H))[0] == omega
    assert solved(graphs.min_vertex_cover(H))[0] == H.number_of_nodes() - alpha      # Gallai


def test_penalty_form_has_the_same_optimum():
    G = nx.gnp_random_graph(8, 0.4, seed=5)
    e, x = model_optimum(graphs.penalty_form_independent_set(G))
    assert -e == solved(graphs.max_independent_set(G))[0]
    S = {i for i, b in enumerate(x) if b}
    assert all(not G.has_edge(u, v) for u, v in itertools.combinations(S, 2))


def test_coloring_counts_conflicts():
    assert solved(graphs.coloring(nx.cycle_graph(5), 2))[0] == 1       # an odd cycle needs 3 colors
    assert solved(graphs.coloring(nx.cycle_graph(5), 3))[0] == 0
    assert solved(graphs.coloring(nx.complete_graph(4), 3))[0] == 1
    p = graphs.coloring(nx.path_graph(3), 2)
    assert not p.check({0: 0, 1: -1, 2: 1})["valid"]                    # an uncolored node is invalid


def test_bisection_is_balanced():
    G = nx.random_regular_graph(3, 8, seed=2)
    best = min(sum(1 for u, v in G.edges() if (u in S) != (v in S)) for S in subsets(list(G)) if len(S) == 4)
    score, S = solved(graphs.bisection(G))
    assert score == best and len(S) == 4


def test_dominating_set():
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))     # gamma(3x3 grid) = 3
    assert solved(graphs.min_dominating_set(G))[0] == 3
    assert not graphs.min_dominating_set(G).check({0})["valid"]


def test_max_weight_matching_matches_networkx():
    G = graphs.weighted(nx.gnp_random_graph(7, 0.45, seed=4), seed=4)
    ref = nx.max_weight_matching(G)
    want = sum(G[u][v]["weight"] for u, v in ref)
    assert solved(graphs.max_weight_matching(G))[0] == want
    p = graphs.max_weight_matching(nx.path_graph(3))
    assert not p.check([(0, 1), (1, 2)])["valid"]


def test_min_maximal_matching():
    assert solved(graphs.min_maximal_matching(nx.path_graph(4)))[0] == 1   # the middle edge
    assert solved(graphs.min_maximal_matching(nx.cycle_graph(6)))[0] == 2
    p = graphs.min_maximal_matching(nx.path_graph(5))
    assert not p.check([(0, 1)])["valid"]                                # (2,3) and (3,4) can still be added


def test_tsp_matches_brute_force():
    G = graphs.tsp_instance(4, seed=7)
    best = min(sum(G[t[i]][t[(i + 1) % 4]]["weight"] for i in range(4)) for t in itertools.permutations(range(4)))
    score, tour = solved(graphs.tsp(G))
    assert score == best and sorted(tour) == [0, 1, 2, 3]
    assert not graphs.tsp(G).check([0, 1, 1, None])["valid"]


def test_instances_are_reproducible_and_build():
    a = graphs.instances("small", 7)
    b = graphs.instances("small", 7)
    assert [sorted(G.edges()) for _, G, _ in a] == [sorted(G.edges()) for _, G, _ in b]
    assert [n for n, _, _ in a] == list(graphs.BUILDERS)
    for name, G, kw in graphs.instances("default", 7):
        m = graphs.build(name, G, **kw).model
        assert 0 < m.n <= 200, (name, m.n)


def test_run_end_to_end_offline(canned, tmp_path):
    import run

    run.main(["--out", str(tmp_path), "--size", "small", "--seeds", "1", "2", "--time-limit", "0.5"])
    rows = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 20 and {r["problem"] for r in rows} == set(graphs.BUILDERS)
    for r in rows:                     # canned all-zeros answers: the checks run and agree, whatever they say
        assert isinstance(r["valid"], bool), r["problem"]
        if r["n_rows"]:
            assert r["service_feasible"] == r["model_feasible"], r["problem"]
        if r["valid"]:
            assert r["model_feasible"] is not False, r["problem"]
        assert r["job_id"] and "engine_version" in r and "client_version" in r
    assert all(c["time_limit_s"] == 0.5 for c in canned.calls)
