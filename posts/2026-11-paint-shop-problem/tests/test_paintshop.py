import itertools
import json

import numpy as np
import pytest

import paintshop as ps
import os

POST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import postload  # noqa: E402  (blog/tests/postload.py; blog/tests is on sys.path from conftest)


def load(name):
    """Import <post dir>/<name>.py under a name unique to this post (blog/tests/postload.py), so posts with a
    run.py or plot.py of their own never collide when their tests run in one session."""
    return postload.load(POST_DIR, name)


plot = load("plot")
solve = load("solve")

WORD = [0, 1, 0, 2, 1, 2]          # hand-made: cars 0, 1, 2, each twice


def test_random_word_has_every_car_twice_and_is_seeded():
    w = ps.random_word(10, seed=4)
    assert ps.check_word(w) == 10 and w == ps.random_word(10, seed=4) and w[0] == 0
    with pytest.raises(ValueError):
        ps.check_word([0, 1, 1])


def test_colors_and_changes_on_a_hand_made_line():
    # x = (0, 1, 0): positions 0 1 0 2 1 2 -> 0 1 1 0 0 1 -> 3 changes
    assert ps.colors(WORD, [0, 1, 0]) == [0, 1, 1, 0, 0, 1]
    assert ps.color_changes(WORD, [0, 1, 0]) == 3
    # every car's two bodies get different colors, whatever x is
    for x in itertools.product((0, 1), repeat=3):
        s = ps.colors(WORD, x)
        for c in range(3):
            assert sorted(s[i] for i, v in enumerate(WORD) if v == c) == [0, 1]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_qubo_and_ising_energies_equal_the_color_changes(seed):
    w = ps.random_word(7, seed=seed) + []
    w = w if seed != 3 else [0, 0, 1, 2, 2, 1, 3, 4, 3, 4, 5, 6, 5, 6]   # adjacent twins included
    n = ps.check_word(w)
    J, const = ps.ising_couplings(w)
    m = ps.model(w)
    for x in itertools.product((0, 1), repeat=n):
        ch = ps.color_changes(w, x)
        assert ps.qubo_energy(w, x) == ch
        assert m.energy(list(x)) == ch
        assert ps.ising_energy(J, const, [1 - 2 * v for v in x]) == pytest.approx(ch)


def test_flip_symmetry_and_brute_force_matches_enumeration():
    w = ps.random_word(9, seed=5)
    best, x = ps.brute_force(w)
    allx = min(ps.color_changes(w, x) for x in itertools.product((0, 1), repeat=9))
    assert best == allx == ps.color_changes(w, x)
    assert ps.color_changes(w, x) == ps.color_changes(w, [1 - v for v in x])


def test_greedy_is_feasible_and_never_better_than_the_optimum():
    for seed in range(1, 6):
        w = ps.random_word(10, seed=seed)
        g = ps.greedy(w)
        assert len(g) == 10 and set(g) <= {0, 1}
        assert ps.color_changes(w, g) >= ps.brute_force(w)[0]
    # hand-made: greedy on 0 1 0 2 1 2 -> first bodies copy the previous color
    assert ps.greedy(WORD) == [0, 0, 1]
    assert ps.colors(WORD, ps.greedy(WORD)) == [0, 0, 1, 1, 1, 0]


def test_qaoa_circuit_has_one_rzz_per_coupling_per_layer():
    w = ps.random_word(12, seed=2)
    J, _ = ps.ising_couplings(w)
    for p in (1, 2):
        qc = ps.qaoa_circuit(w, p=p, measure=False)
        assert qc.num_qubits == 12
        assert qc.count_ops()["rzz"] == p * len(J)
    deg = np.zeros(12, dtype=int)
    for a, b in J:
        deg[a] += 1
        deg[b] += 1
    assert deg.max() <= 4                     # each car sits next to at most four others


def test_multicar_model_rows_and_references():
    classes, quotas = [0, 1, 0, 1, 0, 1], [2, 1]
    m = ps.multicar_model(classes, quotas)
    assert m.m == 2
    for x in itertools.product((0, 1), repeat=6):
        assert m.energy(list(x)) == ps.count_changes(x)
        assert bool(m.feasible(list(x))) == ps.multicar_feasible(classes, quotas, x)
    h = ps.multicar_sequence_heuristic(classes, quotas)
    assert h == [1, 1, 1, 0, 0, 0] and ps.multicar_feasible(classes, quotas, h)
    best, x = ps.multicar_brute_force(classes, quotas)
    assert ps.multicar_feasible(classes, quotas, x) and best == ps.count_changes(x) <= ps.count_changes(h)
    c2, q2 = ps.random_multicar(40, 4, seed=1)
    assert sorted(set(c2)) == [0, 1, 2, 3] and all(0 < q < c2.count(k) for k, q in enumerate(q2))


def test_solve_plumbing_binary_with_a_canned_answer(canned, tmp_path):
    canned.solution = lambda call: [0] * call["n"]
    solve.main(["--out", str(tmp_path), "--kind", "binary", "--sizes", "8", "--instances", "1",
                "--seeds", "1", "--time-limit", "1"])
    rows = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    ref = next(r for r in rows if r["kind"] == "reference")
    s = next(r for r in rows if r["kind"] == "solve")
    w = ps.random_word(8, seed=1)
    assert s["color_changes"] == ps.color_changes(w, [0] * 8) == s["energy_python"]
    assert s["optimum"] == ref["optimum"] == ps.brute_force(w)[0]
    assert s["job_id"].startswith("canned-") and s["solution"] == [0] * 8


def test_solve_plumbing_multicar_checks_quotas(canned, tmp_path):
    solve.main(["--out", str(tmp_path), "--kind", "multicar", "--sizes", "8", "--instances", "1",
                "--seeds", "1", "--time-limit", "1"])
    s = next(json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines() if '"solve"' in l)
    assert s["quotas_met"] is False and s["model_feasible"] is False      # canned all-zeros misses the quotas
    assert s["n_rows"] == 4


def test_selection_shows_only_strict_wins_with_optimum():
    rows = [{"kind": "solve", "kind_model": "binary", "n": 10, "instance": 1, "color_changes": 3, "optimum": 3,
             "greedy": 4},
            {"kind": "solve", "kind_model": "binary", "n": 12, "instance": 1, "color_changes": 4, "optimum": 3,
             "greedy": 5},                                            # not optimal: out
            {"kind": "solve", "kind_model": "binary", "n": 100, "instance": 1, "color_changes": 20, "greedy": 20}]  # tie
    _, sizes, shown = plot.solve_cells(rows, "binary", "greedy")
    assert sizes == [10, 12, 100] and shown == [10]
    comp = [{"kind": "result", "family": "paint_shop_qaoa", "cars": 20, "layers": 1, "budget_s": 10, "arm": "4colors",
             "two_qubit_gates": 50},
            {"kind": "result", "family": "paint_shop_qaoa", "cars": 20, "layers": 1, "budget_s": 10, "arm": "qiskit",
             "coupler_set": "full", "two_qubit_gates": 60},
            {"kind": "result", "family": "paint_shop_qaoa", "cars": 40, "layers": 1, "budget_s": 10, "arm": "4colors",
             "two_qubit_gates": 90},
            {"kind": "result", "family": "paint_shop_qaoa", "cars": 40, "layers": 1, "budget_s": 10, "arm": "qiskit",
             "coupler_set": "working", "two_qubit_gates": 80}]       # only the extra arm: not compared
    med, shown = plot.compile_cells(comp)
    assert shown == [(20, 1, 10)]


def test_compile_and_plot_plumbing(canned_compiler, tmp_path):
    comp = load("compile")

    comp.main(["--out", str(tmp_path / "c"), "--sizes", "8", "--layers", "1", "--budgets", "1", "--seeds", "1",
               "--max-qiskit-trials", "1"])
    rows = [json.loads(l) for l in (tmp_path / "c" / "results.jsonl").read_text().splitlines()]
    four = [r for r in rows if r["kind"] == "result" and r["arm"] == "4colors"]
    qk = [r for r in rows if r["kind"] == "result" and r["arm"] == "qiskit" and r["coupler_set"] == "full"]
    assert len(four) == 1 and len(qk) == 1 and four[0]["couplings"] == qk[0]["couplings"]
    assert four[0]["job_id"].startswith("canned-")
    plot.main([str(tmp_path / "c"), "--figs", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / "selection.md").exists()
