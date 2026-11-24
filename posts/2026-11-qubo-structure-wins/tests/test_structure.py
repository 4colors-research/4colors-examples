"""Offline tests: the penalty builders against their definitions, the exact references against brute force, the
scorer, the selection rule, and run.py / rivals.py / plot.py plumbing (canned answers; nothing is solved by 4colors)."""
import dataclasses
import itertools
import json

import pytest

import examplekit as bk
import plot
import rivals
import run
import structure as S

TINY = {"problem": "knapsack", "name": "tiny", "n": 4, "seed": 0, "w": [3, 4, 5, 2], "v": [4, 5, 6, 3], "cap": 7}


def brute_knapsack(inst):
    best = 0
    for x in itertools.product((0, 1), repeat=inst["n"]):
        if sum(w for w, xi in zip(inst["w"], x) if xi) <= inst["cap"]:
            best = max(best, sum(v for v, xi in zip(inst["v"], x) if xi))
    return best


@pytest.mark.parametrize("cap", [1, 2, 7, 10, 37])
def test_slack_bits_reach_exactly_0_to_cap(cap):
    bits = S.slack_bits(cap)
    sums = {sum(b for b, s in zip(bits, x) if s) for x in itertools.product((0, 1), repeat=len(bits))}
    assert sums == set(range(cap + 1))


def test_penalty_qubo_equals_its_definition_on_every_assignment():
    m = S.knapsack_penalty_model(TINY)
    Q, lin, off, n, lam = S.knapsack_penalty_qubo(TINY)
    assert n == 4 + 3 and lam == 7
    assert all(isinstance(v, int) for v in list(Q.values()) + lin + [off])     # integral: detection needs it
    for x in itertools.product((0, 1), repeat=n):
        assert m.energy(list(x)) == S.knapsack_penalty_energy(TINY, x)


def test_penalty_qubo_minimum_is_the_feasible_optimum():
    m = S.knapsack_penalty_model(TINY)
    best = min(itertools.product((0, 1), repeat=m.n), key=lambda x: m.energy(list(x)))
    assert S.knapsack_judge(TINY, best)["feasible"]
    assert -m.energy(list(best)) == brute_knapsack(TINY) == S.knapsack_optimum(TINY)


@pytest.mark.parametrize("seed", range(6))
def test_dp_optimum_equals_brute_force(seed):
    inst = S.knapsack_instance(12, seed, wmax=50, spread=10)
    assert S.knapsack_optimum(inst) == brute_knapsack(inst)


def test_rows_model_feasible_iff_weight_fits():
    m = S.knapsack_rows_model(TINY)
    for x in itertools.product((0, 1), repeat=4):
        assert bool(m.feasible(list(x))) == S.knapsack_judge(TINY, x)["feasible"]


def test_numpart_energy_is_the_squared_difference():
    inst = S.numpart_instance(8, 1, bits=6)
    m = S.numpart_model(inst)
    diffs = []
    for s in itertools.product((-1, 1), repeat=8):
        d = S.numpart_diff(inst, s)
        assert m.energy(list(s)) == d * d
        diffs.append(d)
    opt = min(diffs)
    assert S.numpart_lower_bound(inst) <= opt <= S.karmarkar_karp(inst["a"])


def test_numpart_coefficients_stay_exact_at_the_largest_size():
    inst = S.numpart_instance(400, 3)
    J, off = S.numpart_ising(inst)
    assert max(J.values()) < 2 ** 53 and off < 2 ** 53
    with pytest.raises(ValueError):
        S.numpart_instance(10, 1, bits=21)


def test_score_marks_infeasible_and_proof():
    ref = S.reference(TINY)
    assert S.score(TINY, ref, [1, 1, 1, 1])["gap"] is None            # overweight
    assert S.score(TINY, ref, [0, 0, 0, 0])["gap"] == 1.0
    inst = {"problem": "numpart", "a": [3, 1, 2], "n": 3}
    r = S.reference(inst)
    assert S.score(inst, r, [1, -1, -1]) == {"diff": 0, "proved_optimal": True}


def test_extra_fields_keeps_unknown_result_attributes():
    @dataclasses.dataclass
    class R:
        status: str = "finished"
        objective: int = 0
        some_new_report: dict = dataclasses.field(default_factory=lambda: {"rows": 1})
        _private: int = 0

    assert run.extra_fields(R()) == {"some_new_report": {"rows": 1}}


def _r(arm, problem, T, inst, seed, **kw):
    return {"kind": "solve", "arm": arm, "problem": problem, "n": 100, "budget_s": T, "instance": inst,
            "solver_seed": seed, **kw}


def test_selection_rule():
    rows = []
    for s in (1, 2, 3):
        rows += [_r("4colors-penalty", "knapsack", 10, "k", s, gap=0.0),
                 _r("neal", "knapsack", 10, "k", s, gap=0.01),                         # win
                 _r("4colors-penalty", "knapsack", 1, "k", s, gap=0.001),
                 _r("tabu", "knapsack", 1, "k", s, gap=0.0012),                        # tie band: out
                 _r("4colors-penalty", "numpart", 10, "p", s, diff=1),
                 _r("openjij", "numpart", 10, "p", s, diff=0, over_budget=True),       # overran: dropped
                 _r("neal", "numpart", 10, "p", s, diff=40),
                 _r("4colors-penalty", "knapsack", 60, "k", s, gap=None),              # ours infeasible: out
                 _r("neal", "knapsack", 60, "k", s, gap=0.5)]
    table, n_cells = plot.select_wins(rows)
    assert n_cells == 4
    assert {(t["problem"], t["budget_s"], t["best_rival"]) for t in table} == {("knapsack", 10.0, "neal"),
                                                                               ("numpart", 10.0, "neal")}


def test_run_rivals_and_plot_offline(canned, tmp_path):
    canned.solution = lambda call: [1] * call["n"]          # fixed canned answer: all ones, nothing solved
    rows = run.main(["--out", str(tmp_path / "s"), "--sizes-knapsack", "20", "--sizes-numpart", "12",
                     "--instance-seeds", "1", "--budgets", "1", "--seeds", "1"])
    solves = [r for r in rows if r["kind"] == "solve"]
    assert {r["arm"] for r in solves} == {"4colors-penalty", "4colors-rows"}
    k = [r for r in solves if r["problem"] == "knapsack"]
    assert all(r["feasible"] is False and r["gap"] is None for r in k)    # all ones overloads the knapsack
    assert next(r for r in solves if r["arm"] == "4colors-rows")["n_rows"] == 1
    npp = next(r for r in solves if r["problem"] == "numpart")
    inst = S.numpart_instance(12, 1)
    assert npp["diff"] == sum(inst["a"]) and npp["model_energy"] == npp["diff"] ** 2
    rv = rivals.main(["--out", str(tmp_path / "rv"), "--sizes-knapsack", "20", "--sizes-numpart", "12",
                      "--instance-seeds", "1", "--budgets", "0.05", "--seeds", "1", "--settings", "tabu"])
    assert len(rv) == 2 and all("error" not in r for r in rv)
    assert next(r for r in rv if r["problem"] == "numpart")["diff"] is not None
    plot.main([str(tmp_path / "s" / "results.jsonl"), str(tmp_path / "rv" / "results.jsonl"),
               "--out", str(tmp_path / "fig")])
    for f in ("knapsack_gap.png", "numpart_difference.png", "penalty_vs_row.png", "wins.md"):
        assert (tmp_path / "fig" / f).exists()
    json.dumps(rows)


def test_sb_without_harness_stops(tmp_path):
    with pytest.raises(SystemExit):
        rivals.main(["--out", str(tmp_path), "--settings", "sb"])
