"""Offline tests: instances, the shared evaluator, each wrapper on tiny instances (each run well under 1 s),
the MQLib trap checks, and run.py end to end with a canned API answer (a fixed all-+1 assignment)."""
import importlib.util
import itertools
import os
import types
from pathlib import Path

import numpy as np
import pytest

import examplekit as bk
import fourcolors.qubo as fq
import instances as I
import plot
import run
import solvers as S

MQLIB = os.environ.get("MQLIB_BIN", "")
have_mqlib = pytest.mark.skipif(not (MQLIB and Path(MQLIB).is_file()), reason="no MQLib executable (set MQLIB_BIN)")


def gset_file(tmp_path, edges, n):
    p = tmp_path / "toy.txt"
    p.write_text(f"{n} {len(edges)}\n" + "".join(f"{a} {b} {w}\n" for a, b, w in edges))
    return p


def test_gset_reader_and_merge(tmp_path):
    p = gset_file(tmp_path, [(1, 2, 1), (2, 3, -1), (3, 1, 1), (2, 1, 1)], 4)      # (1,2) twice: summed
    inst = I.build(f"gset:path={p},ref=3")
    assert inst["n"] == 4 and inst["i"].tolist() == [0, 0, 1] and inst["c"].tolist() == [2, 1, -1]
    assert inst["reference"] == 3 and inst["sense"] == "max"


def test_generators():
    r = I.build("regular:n=50,d=3,seed=2")
    assert np.all(np.bincount(np.r_[r["i"], r["j"]], minlength=50) == 3)
    s = I.build("sparse:n=1000,deg=8,seed=1")
    assert len(s["i"]) == 4000 and np.all(s["i"] < s["j"])
    assert np.array_equal(s["i"], I.build("sparse:n=1000,deg=8,seed=1")["i"])
    e = I.build("ea3d:L=4,seed=1")
    assert e["n"] == 64 and len(e["i"]) == 192 and e["reference"] == pytest.approx(-1.7863 * 64)
    pm = I.build("regular:n=20,d=3,seed=1,w=pm1")
    assert set(pm["c"].tolist()) <= {-1, 1}


def test_objective_matches_brute_force_maxcut(tmp_path):
    inst = I.build("sparse:n=12,deg=3,seed=4,w=pm1")
    edges = list(zip(inst["i"].tolist(), inst["j"].tolist()))
    w = dict(zip(edges, inst["c"].tolist()))
    best, _ = bk.brute_force_maxcut(12, edges, w)
    got = max(I.objective(inst, np.array(b)) for b in itertools.product((-1, 1), repeat=12))
    assert got == best
    s = np.random.default_rng(0).choice([-1, 1], 12)
    assert I.objective(inst, s) == bk.cut_value(edges, s, w)


def test_spin_glass_objective_is_the_energy():
    inst = I.build("ea3d:L=3,seed=2")
    s = np.ones(27, dtype=int)
    assert I.objective(inst, s) == I.energy(inst, s) == int(inst["c"].sum())
    assert I.gap(inst, inst["reference"]) == 0


CANNED = '0.2,BURER2002,"/x/y.txt",3,0.205731,[0:3e-06;3:2.9e-05]\n\nSolution:\n1 -1 -1\n'
TRAP = '0.2,HH_,"/x/y.txt",0,0.000132,[0:7e-06]\n\nSolution:\n-1 -1 -1\n'


def test_mqlib_parser_and_trap_checks(tmp_path):
    inst = I.build(f"gset:path={gset_file(tmp_path, [(1, 2, 3), (2, 3, -2)], 3)}")
    p = S.parse_mqlib(CANNED)
    assert p["objective"] == 3 and p["s"].tolist() == [1, -1, -1] and p["history"][-1] == (3.0, 2.9e-05)
    S.check_mqlib(p, inst, hh=False)
    with pytest.raises(S.MQLibTrap, match="no heuristic"):
        S.check_mqlib(S.parse_mqlib(TRAP), inst, hh=True)
    wrong = dict(p, objective=5.0)
    with pytest.raises(S.MQLibTrap, match="printed objective"):
        S.check_mqlib(wrong, inst, hh=False)
    assert S.mqlib_root("/opt/MQLib/bin/MQLib") == Path("/opt/MQLib")


def test_hh_refused_without_hhdata(tmp_path):
    fake = tmp_path / "MQLib" / "bin" / "MQLib"
    fake.parent.mkdir(parents=True); fake.write_text("")
    inst = I.build("regular:n=10,d=3,seed=1")
    with pytest.raises(S.MQLibTrap, match="hhdata"):
        S.run_mqlib(inst, 0.1, 0, binary=fake, hh=True)


@have_mqlib
def test_mqlib_burer_live_on_a_tiny_instance():
    inst = I.build("regular:n=30,d=3,seed=1,w=pm1")
    out = S.run_mqlib(inst, 0.1, 1, binary=MQLIB, hh=False)
    assert out["heuristic"] == "BURER2002" and out["reported_energy"] == I.energy(inst, out["s"])


@pytest.mark.parametrize("name", ["neal", "neal-fill", "openjij", "openjij-fill"])
def test_annealers_on_a_tiny_instance(name):
    if name.startswith("openjij"):
        pytest.importorskip("openjij")
    inst = I.build("ea3d:L=3,seed=1")
    bqm = S.bqm_of(inst)
    cal = S.calibrate(name, bqm, 0.2, seed=0) if name.endswith("-fill") else None
    out = S.run_annealer(inst, 0.2, 3, name=name, bqm=bqm, calibration=cal)
    assert set(np.unique(out["s"]).tolist()) <= {-1, 1}
    assert out["reported_energy"] == I.energy(inst, out["s"])
    if cal:
        assert cal["num_sweeps"] >= 10 and out["num_sweeps"] == cal["num_sweeps"]
    else:
        assert out["num_sweeps"] is None                      # library defaults


def test_run_end_to_end_offline(tmp_path, monkeypatch):
    held = {}

    def fake_upload(model, client=None):
        held["m"] = model
        return "id"

    def fake_solve(input_id=None, time_limit_s=None, seed=None, client=None):
        m = held["m"]
        s = np.ones(m.n, dtype=int)                     # canned: a fixed answer, nothing solved
        return types.SimpleNamespace(status="finished", objective=m.energy(s), solution=s.tolist(),
                                     trajectory=[{"t_search_s": 0.01}],
                                     job=types.SimpleNamespace(id="j", billed_s=time_limit_s, run_s=time_limit_s,
                                                               wait_s=0.0))

    monkeypatch.setattr(fq, "upload", fake_upload)
    monkeypatch.setattr(fq, "solve", fake_solve)
    settings = ["4colors", "neal"] + (["openjij-fill"] if importlib.util.find_spec("openjij") else []) \
        + (["mqlib-burer"] if MQLIB and Path(MQLIB).is_file() else [])
    out = tmp_path / "o"
    rows = run.main(["--out", str(out), "--instance", "ea3d:L=3,seed=1", "regular:n=20,d=3,seed=1",
                     "--budgets", "0.1", "--seeds", "0", "1", "--solvers", *settings, "--mqlib", MQLIB],
                    client=object())
    solved = [r for r in rows if r.get("objective") is not None]
    assert len(solved) == 2 * 2 * len(settings)
    assert all(r["agrees"] for r in solved)
    assert {r["setting"] for r in solved} == set(settings)
    s = plot.main([str(out / "results.jsonl"), "--out", str(tmp_path / "f")])
    assert (tmp_path / "f" / "gap_by_budget.png").exists() and (tmp_path / "f" / "wall_vs_budget.png").exists()
    assert s["best_other"][0.1] is not None


def test_errors_are_rows_not_gaps(tmp_path):
    rows = run.main(["--out", str(tmp_path / "e"), "--instance", "regular:n=10,d=3,seed=1", "--budgets", "0.1",
                     "--seeds", "0", "--solvers", "mqlib-burer", "--mqlib", str(tmp_path / "missing" / "bin" / "MQLib")])
    assert len(rows) == 1 and "error" in rows[0]
