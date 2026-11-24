"""Unit tests of the mRMR QUBO, the references, the AUC helper and the script plumbing (no service calls)."""
import itertools

import numpy as np

import examplekit
import features as fs
import plot
import run


def toy():
    rel = np.array([0.30, 0.28, 0.10, 0.05, 0.20])
    red = np.zeros((5, 5))
    red[0, 1] = red[1, 0] = 0.25          # 0 and 1 are near copies
    red[1, 4] = red[4, 1] = 0.02
    red[2, 3] = red[3, 2] = 0.01
    return rel, red


def test_model_energy_and_row_on_every_assignment():
    rel, red = toy()
    m = fs.model(rel, red, 0.6, 2)
    for x in itertools.product([0, 1], repeat=5):
        assert abs(m.energy(list(x)) - fs.energy(rel, red, 0.6, x)) < 1e-12
        assert bool(m.feasible(list(x))) == (sum(x) == 2)


def test_exact_and_greedy_on_a_case_where_greedy_is_fooled():
    rel, red = toy()
    best, x = fs.exact_best(rel, red, 0.5, 2, chunk=3)
    brute = min(fs.energy(rel, red, 0.5, c) for c in itertools.product([0, 1], repeat=5) if sum(c) == 2)
    assert abs(best - brute) < 1e-12
    assert fs.chosen(x) == [0, 4]                         # the redundant pair (0, 1) is avoided
    g = fs.greedy_qubo(rel, red, 0.5, 2)
    assert sum(g) == 2 and fs.energy(rel, red, 0.5, g) >= best - 1e-12
    p = fs.greedy_mrmr(rel, red, 2)
    assert fs.chosen(p)[0] == 0 or 0 in fs.chosen(p)      # Peng's greedy starts from the most relevant
    # a hand-made case where forward selection is suboptimal: one strong feature redundant with two good ones
    rel2 = np.array([1.0, 0.8, 0.8])
    red2 = np.array([[0, 0.9, 0.9], [0.9, 0, 0.0], [0.9, 0.0, 0]])
    g2 = fs.greedy_qubo(rel2, red2, 0.5, 2)
    e2, x2 = fs.exact_best(rel2, red2, 0.5, 2)
    assert fs.chosen(g2)[0] == 0 or 0 in fs.chosen(g2)
    assert fs.chosen(x2) == [1, 2] and e2 < fs.energy(rel2, red2, 0.5, g2)


def test_mutual_info_basics():
    a = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    assert abs(fs.mutual_info(a, a) - np.log(2)) < 1e-12           # a fair bit carries log 2 nats about itself
    assert abs(fs.mutual_info(np.array([0, 0, 1, 1] * 2), np.array([0, 1] * 4))) < 1e-12    # independent columns
    col = np.arange(100.0)
    codes = fs.discretize(col, 4)
    assert set(codes) == {0, 1, 2, 3} and np.bincount(codes).tolist() == [25, 25, 25, 25]
    assert fs.discretize(np.array([3.0, 1.0, 3.0]), 8).tolist() == [1, 0, 1]


def test_auc_by_ranks():
    y = np.array([0, 0, 1, 1])
    assert fs.auc(np.array([0.1, 0.2, 0.3, 0.4]), y) == 1.0
    assert fs.auc(np.array([0.4, 0.3, 0.2, 0.1]), y) == 0.0
    assert fs.auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5


def test_loaders(tmp_path):
    g = tmp_path / "german.data-numeric"
    g.write_text("   1   6   4  2\n   2  48   2  1\n   4  12   1  1\n")
    X, y, names = fs.load_german_numeric(str(g))
    assert X.shape == (3, 3) and y.tolist() == [1, 0, 0] and names[0] == "a01"
    c = tmp_path / "b.csv"
    c.write_text("Bankrupt?, x1, name, x2\n1,0.5,a,3\n0,0.1,b,4\n")
    X, y, names = fs.load_csv(str(c), "Bankrupt?")
    assert names == ["x1", "x2"] and y.tolist() == [1, 0] and X.shape == (2, 2)


def test_run_end_to_end_and_plots(canned, tmp_path):
    k = 3
    canned.solution = lambda call: [1] * k + [0] * (call["n"] - k)        # a fixed, feasible answer
    run.main(["--out", str(tmp_path / "f"), "--dataset", "synthetic:300:10:1", "--k", str(k), "--alpha", "0.5",
              "0.9", "--seeds", "1", "2", "--auc"])
    rows = examplekit.Results(tmp_path / "f" / "results.jsonl").rows()
    inst = [r for r in rows if r["kind"] == "instance"][0]
    assert inst["d"] == 10 and inst["n_rows"] == 300
    refs = [r for r in rows if r["kind"] == "reference"]
    assert {r["method"] for r in refs} == {"exact", "greedy_qubo", "greedy_mrmr"} and len(refs) == 6
    solves = [r for r in rows if r["kind"] == "solve"]
    assert len(solves) == 4
    for r in solves:
        assert r["features"] == [0, 1, 2] and r["model_feasible"] and r["service_feasible"]
        assert abs(r["energy_python"] - r["solver_objective"]) < 1e-9 and 0 <= r["cv_auc"] <= 1
    ex = {r["alpha"]: r["energy_python"] for r in refs if r["method"] == "exact"}
    assert all(r["energy_python"] >= ex[r["alpha"]] - 1e-12 for r in solves)    # nothing beats the enumeration
    plot.main([str(tmp_path / "f"), "--out", str(tmp_path / "fig"), "--show", "synthetic-300x10-s1:3:0.5"])
    assert (tmp_path / "fig" / "fs_vs_greedy.png").exists() and (tmp_path / "fig" / "fs_selected.png").exists()


def test_wins_rule_drops_ties_and_losses():
    rows = []
    for alpha, ours, greedy in [(0.5, -1.0, -0.9), (0.7, -1.0, -1.0), (0.9, -0.8, -1.0)]:
        rows += [{"kind": "reference", "method": "greedy_qubo", "dataset": "d", "k": 2, "alpha": alpha,
                  "energy_python": greedy},
                 {"kind": "reference", "method": "exact", "dataset": "d", "k": 2, "alpha": alpha,
                  "energy_python": min(ours, greedy)},
                 {"kind": "solve", "dataset": "d", "k": 2, "alpha": alpha, "model_feasible": True,
                  "energy_python": ours}]
    cs = plot.cells(rows)
    w = plot.wins(cs)
    assert len(cs) == 3 and [c["alpha"] for c in w] == [0.5]
    assert w[0]["at_optimum"] and abs(w[0]["gain_pct"] - 100 * 0.1 / 0.9) < 1e-9


def test_dataset_specs(tmp_path):
    c = tmp_path / "data.csv"
    c.write_text("Bankrupt?,x1,x2\n1,0.5,3\n0,0.1,4\n")
    name, X, y, names, digest = run.load(f"csv:taiwan-bankruptcy:Bankrupt?:{c}")
    assert name == "taiwan-bankruptcy" and names == ["x1", "x2"] and len(digest) == 64
    g = tmp_path / "german.data-numeric"
    g.write_text("1 2 1\n3 4 2\n")
    assert run.load(f"german:{g}")[0] == "german-credit"
    assert run.load("synthetic:50:6:2")[0] == "synthetic-50x6-s2"
