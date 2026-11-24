"""Offline tests for the biggest-circuits post (no API call)."""
import json

from conftest import load

import circuitkit as ck


def test_default_configs_span_the_range():
    """The default configs give 150-qubit inputs from about 500 to over 20,000 two-qubit gates (graphs only;
    building the 20k-gate circuits is left to the run)."""
    import examplekit

    run = load("run")
    sizes = []
    for density, p in run.configs("0.05:1,0.1:2,0.2:2,0.2:4,0.4:4,0.5:4"):
        edges = examplekit.gnp(150, density, seed=150)
        sizes.append(p * len(edges))
    assert sizes == sorted(sizes)                     # listed from smallest to largest input
    assert min(sizes) < 1000 and max(sizes) > 20000 and any(1800 < s < 2600 for s in sizes)


def test_build_counts_one_rzz_per_edge_per_layer():
    run = load("run")
    qc, circ = run.build(12, 0.5, 3)
    assert circ["input_2q"] == 3 * circ["edges"] == qc.count_ops()["rzz"]
    assert circ["qubits"] == 12 and circ["circuit_id"] == "qaoa_gnp0.5_12_p3"


def test_refusal_rows_and_skips(canned, tmp_path, monkeypatch):
    """A circuit refused as too large is a row with its code; its larger budgets become skip rows; a budget
    refusal does not stop the larger budgets."""
    import fourcolors
    from fourcolors.errors import InvalidRequest

    run = load("run")
    real = canned.compile

    def picky(circuit, target=None, time_budget_s=None, **kw):
        if circuit.num_qubits >= 8:
            raise InvalidRequest("The circuit has too many two-qubit gates", code="input_too_large",
                                 field="input.circuit")
        if time_budget_s < 0.02:
            raise InvalidRequest("time_budget_s is too short for this circuit", code="invalid_option",
                                 field="options.time_budget_s")
        return real(circuit, target, time_budget_s, **kw)

    monkeypatch.setattr(fourcolors.quantum.compiler, "compile", picky)
    run.main(["--out", str(tmp_path), "--sizes", "6,8", "--configs", "0.5:1", "--budgets", "0.01,0.03,0.05",
              "--seeds", "1", "--skip-qiskit"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    small = [r for r in rows if r.get("qubits") == 6 and r["kind"] in ("result", "skip")]
    assert [(r["kind"], r["refusal_code"]) for r in small] == [("result", "invalid_option"), ("result", None),
                                                               ("result", None)]
    big = [r for r in rows if r.get("qubits") == 8 and r["kind"] in ("result", "skip")]
    assert [(r["kind"], r["refusal_code"]) for r in big] == [("result", "input_too_large"), ("skip", None),
                                                             ("skip", None)]
    assert big[0]["refusal_field"] == "input.circuit" and big[0]["billed_s"] == 0.0


def test_run_and_plot_offline(canned, tmp_path):
    run, plot = load("run"), load("plot")
    run.main(["--out", str(tmp_path), "--sizes", "8", "--configs", "0.3:1,0.6:2", "--budgets", "0.01,0.02",
              "--seeds", "1", "--max-qiskit-trials", "1"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    q = ck.results(rows, arm="qiskit")
    assert q and all(r["peak_rss_mb"] > 0 for r in q)
    assert len(ck.results(rows, arm="4colors")) == 4
    plot.main([str(tmp_path), "--figs", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / "fig_size.png").exists()


def test_plan_limit_and_limits_row(canned, tmp_path):
    """Budgets above --plan-max-budget are skip rows (never sent, also not run on the Qiskit side); the limits
    row is the first row after meta and carries no account fields."""
    run = load("run")
    run.main(["--out", str(tmp_path), "--sizes", "6", "--configs", "0.5:1", "--budgets", "0.01,0.02,5",
              "--seeds", "1", "--plan-max-budget", "0.02", "--max-qiskit-trials", "1"])
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert rows[1]["kind"] == "limits"
    assert set(rows[1]) == {"kind", "post", "plan_max_budget", "limits", "note", "written_utc"}
    four = [r for r in rows if r.get("arm") == "4colors" and r["kind"] in ("result", "skip")]
    assert [(r["kind"], r["budget_s"]) for r in four] == [("result", 0.01), ("result", 0.02), ("skip", 5.0)]
    assert "plan limit" in four[-1]["note"]
    assert not [r for r in rows if r.get("arm") == "qiskit" and r["budget_s"] == 5.0]


def test_selection_rule_keeps_only_wins():
    """A cell is shown only when ours is lower and Qiskit's pick is within budget; ties, losses, over-budget
    and refused cells are left out with a reason."""
    plot = load("plot")

    def r(cid, arm, b, g, **kw):
        return ck.row(kind="result", circuit_id=cid, input_2q=100, arm=arm, budget_s=b, two_qubit_gates=g,
                      coupler_set="full" if arm == "qiskit" else "fakefez", **kw)

    rows = [r("win", "4colors", 10, 90), r("win", "qiskit", 10, 100, gates_on_failed_couplers=3),
            r("tie", "4colors", 10, 100), r("tie", "qiskit", 10, 100),
            r("loss", "4colors", 10, 120), r("loss", "qiskit", 10, 100),
            r("over", "4colors", 10, 50), r("over", "qiskit", 10, 100, over_budget=True),
            r("refused", "4colors", 10, None, refusal_code="input_too_large"), r("refused", "qiskit", 10, 100)]
    got = {c["circuit_id"]: (c["shown"], c["why"]) for c in plot.cells(rows)}
    assert got["win"][0] is True and plot.cells(rows)[0]["qiskit_full_on_dead"] in (3, None)
    assert [k for k, (shown, _) in got.items() if shown] == ["win"]
    assert "reported apart" in got["over"][1] and "refused" in got["refused"][1]
