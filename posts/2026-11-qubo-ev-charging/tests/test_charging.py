"""Offline tests of the EV charging model: generator, rows, penalty form, reference, checker, plumbing.
Nothing calls the service; the `canned` fixture returns a fixed assignment."""
import itertools
import json

import numpy as np
import pytest

import charging as ch


def small(site_kw=None):
    inst = ch.generate(3, 2, n_slots=10, seed=4, starts_per_vehicle=2, chargers_per_vehicle=2, site_kw=site_kw)
    assert 6 <= len(inst.tasks) <= 14
    return inst


def all_x(n):
    for bits in itertools.product((0, 1), repeat=n):
        yield list(bits)


def test_generator_is_deterministic_and_tasks_fit_windows():
    a, b = ch.for_tasks(1000, seed=2), ch.for_tasks(1000, seed=2)
    assert a.tasks == b.tasks and 800 <= len(a.tasks) <= 1200
    for v, c, s in a.tasks:
        veh = a.vehicles[v]
        assert veh["arrival"] <= s and s + veh["length"] <= veh["departure"] <= a.n_slots


@pytest.mark.parametrize("site_kw", [None, 22])
def test_rows_admit_exactly_the_valid_schedules(site_kw):
    inst = small(site_kw)
    m, pw = ch.model(inst), ch.pairwise_model(inst)
    for x in all_x(len(inst.tasks)):
        valid = ch.check(inst, ch.decode(inst, x))["valid"]
        assert m.feasible(x) == valid == pw.feasible(x)
    if site_kw is None:
        assert m.m < pw.m or pw.m == 0                                   # cliques: fewer rows than edges


def brute(inst):
    best = max(all_x(len(inst.tasks)), key=lambda x: ch.check(inst, ch.decode(inst, x))["value"]
               if ch.check(inst, ch.decode(inst, x))["valid"] else -1)
    return ch.check(inst, ch.decode(inst, best))["value"]


def test_penalty_form_has_the_same_optimum():
    inst = small()
    lin, quad, P = ch.penalty_bqm(inst)
    def energy(x):
        return sum(lin[i] * x[i] for i in lin) + sum(v * x[i] * x[j] for (i, j), v in quad.items())
    xmin = min(all_x(len(inst.tasks)), key=energy)
    assert ch.check(inst, ch.decode(inst, xmin))["valid"]
    assert -energy(xmin) == brute(inst)


@pytest.mark.parametrize("site_kw", [None, 22])
def test_milp_reference_is_the_optimum(site_kw):
    inst = small(site_kw)
    ref = ch.milp_reference(inst, 10)
    assert ref["optimal"] and ref["reference_value"] == brute(inst)
    assert ch.check(inst, ch.decode(inst, ref["reference_x"]))["valid"]


def test_check_catches_conflicts():
    inst = small()
    v, c, s = inst.tasks[0]
    same_vehicle = [t for t in inst.tasks if t[0] == v][:2]
    assert not ch.check(inst, same_vehicle)["valid"]
    bad_window = [(v, c, inst.vehicles[v]["departure"])]
    assert "outside its window" in ch.check(inst, bad_window)["problems"][0]
    assert ch.check(inst, [])["valid"] and ch.check(inst, [])["value"] == 0


def test_run_and_plot_offline(canned, tmp_path):
    import run, plot

    out = tmp_path / "ev"
    run.main(["--out", str(out), "--sizes", "100", "--instances", "1", "--time-limits", "1", "--seeds", "1",
              "--reference-time", "10"])
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    solve = next(r for r in rows if r["kind"] == "solve")
    assert solve["valid"] and solve["value"] == 0 and solve["model_feasible"]   # canned: all zeros
    ref = next(r for r in rows if r["kind"] == "reference")
    assert ref["optimal"] and ref["reference_valid"] and ref["reference_value"] > 0
    plot.main([str(out), "--figs", str(tmp_path / "figs")])
    sel = (tmp_path / "figs" / "selection.md").read_text()
    assert "Cells shown: 0 of 1 against the reference" in sel                 # an empty schedule is never shown


def test_selection_rules():
    import plot

    base = {"size": 1000, "time_limit_s": 1.0, "n_tasks": 1000}
    rows = [{"kind": "reference", "size": 1000, "optimal": True, "reference_value": 500},
            {**base, "kind": "solve", "valid": True, "value": 500},
            {**base, "kind": "rival", "arm": "neal-budget_filling", "valid": True, "value": 480}]
    c = plot.cells(rows)[(1000, 1.0)]
    assert c["matches_ref"] and c["beats_rival"]
    rows[2]["value"] = 499.9
    assert not plot.cells(rows)[(1000, 1.0)]["beats_rival"]                    # within the 0.05 % tie band
    rows[1]["valid"] = False
    c = plot.cells(rows)[(1000, 1.0)]
    assert not c["matches_ref"] and not c["beats_rival"]
