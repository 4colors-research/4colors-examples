import json

import numpy as np
from qiskit import transpile

import run
from qaoakit import ansatz as az, apikit as ak


def test_binding_check_on_a_qiskit_routed_circuit():
    """The check accounts for the layout: a routed circuit passes, a wrong binding fails."""
    qc, _ = az.build("hea-ring-6x2")
    out = transpile(qc, target=ak.qiskit_target(), optimization_level=3, seed_transpiler=3)
    assert out.layout is not None
    v = list(np.random.default_rng(0).uniform(-3, 3, len(qc.parameters)))
    assert ak.bound_states_match(qc, out, v)["ok"]
    assert ak.bound_states_match(qc, out, dict(zip(qc.parameters, v)))["ok"]
    assert not ak.bound_states_match(qc, out, v, compiled_values=v[::-1])["ok"]


def test_parameters_survive_both_routes(canned):
    qc, _ = az.build("qaoa-8-p2")
    out, rec = run.compile_pm(qc, 1, 1, None)
    assert list(out.parameters) == list(qc.parameters) and rec["target_how"] == "given"
    out2, rec2 = ak.compile_4colors(qc, budget_s=1, seed=1)
    assert rec2["parameters"] == [p.name for p in qc.parameters]


def test_end_to_end_offline(canned, tmp_path):
    run.main(["--out", str(tmp_path), "--ansatz", "hea-4x3,ma-6", "--budget", "0.01", "--bindings", "4",
              "--check", "2", "--text-demo", "hea-6x1", "--loop-ansatz", "qaoa-6-p1", "--loop-maxiter", "6",
              "--per-step-evals", "3"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    binds = [r for r in rows if r["kind"] == "bindings"]
    assert binds and all(r["checks_ok"] and len(r["two_qubit_gates_per_binding"]) == 1 for r in binds)
    assert all(r["compiled_n_parameters"] == r["n_parameters"] for r in binds)
    demo = next(r for r in rows if r["kind"] == "text_demo")
    assert demo["names_found"] and demo["bind_by_name_ok"]
    assert demo["bind_by_position_ok"] is False          # 12 names sort as t0, t1, t10, t11, t2, ...
    loops = [r for r in rows if r["kind"] == "loop"]
    assert len(loops) == 4
    once = [r for r in loops if r["mode"] == "once"]
    assert abs(once[0]["trace"][0]["energy"] - once[1]["trace"][0]["energy"]) < 1e-9
    per = [r for r in loops if r["mode"] == "per_step"]
    assert all(0 < r["evaluations"] <= 4 for r in per)          # COBYLA takes at least n + 2 evaluations
    # every arm walks the same path: equivalent circuits give the same energies
    for r in per:
        for a, b in zip(r["trace"], once[0]["trace"]):
            assert abs(a["energy"] - b["energy"]) < 1e-8
    # the instant comparison: 4colors with no budget and Qiskit's default transpile, for every ansatz
    comp = [r for r in rows if r["kind"] == "compile"]
    for spec in ("hea-4x3", "ma-6"):
        routes = {(r["arm"], r["route"]) for r in comp if r["ansatz"] == spec}
        assert {("4colors", "compile-instant"), ("qiskit-default", "transpile-default"),
                ("4colors", "generate_pass_manager"), ("4colors", "compile"), ("qiskit-o3", "transpile")} <= routes
    assert all(r["budget_s"] is None for r in comp if r["route"] in ("compile-instant", "transpile-default"))
    assert all(r.get("approximation_mode") == "exact" for r in comp if r["arm"] == "4colors")
    assert all(r.get("verification") is not None for r in comp if r["arm"] == "4colors")
    import plot

    plot.main(["--out", str(tmp_path)])
    assert (tmp_path / "gates.png").exists() and (tmp_path / "loop.png").exists()
    table = (tmp_path / "table.md").read_text()
    assert "instant: 4colors (no budget)" in table and "matched budget" in table and "untimed" in table


def test_selection_rule_per_panel():
    """Only strict wins are charted, per panel; ties (the hardware-efficient ansatze) are left out."""
    import plot

    def c(ansatz, arm, route, g):
        return {"kind": "compile", "ansatz": ansatz, "arm": arm, "route": route, "two_qubit_gates": g}

    comp = [c("hea", "4colors", "generate_pass_manager", 7), c("hea", "qiskit-o3", "transpile", 7),
            c("qaoa", "4colors", "generate_pass_manager", 69), c("qaoa", "qiskit-o3", "transpile", 88),
            c("hea", "4colors", "compile-instant", 7), c("hea", "qiskit-default", "transpile-default", 9),
            c("qaoa", "4colors", "compile-instant", 90), c("qaoa", "qiskit-default", "transpile-default", 88)]
    assert plot.select_ansatze(comp, "budget") == (["qaoa"], ["hea"])
    assert plot.select_ansatze(comp, "instant") == (["hea"], ["qaoa"])
