import examplekit
import grouping as gp
import plot
import run


class Args:
    time_limit, seeds, k_steps, max_nnz = 1.0, [1], 3, 10 ** 9


def test_sweep_records_valid_and_invalid_answers(canned, tmp_path):
    terms = [("XI", 1.0), ("IX", 1.0), ("ZI", 1.0), ("IZ", 1.0)]      # XI,IX | ZI,IZ: 2 groups suffice
    edges = gp.conflict_edges([t[0] for t in terms], "gc")

    def answer(call):
        k = call["n"] // 4
        colors = [0, 0, 1, 1] if k >= 2 else None
        if colors is None:
            return [0] * call["n"]                                      # breaks the one-hot rows: invalid
        return [int(colors[i] == c) for i in range(4) for c in range(k)]

    canned.solution = answer
    out = examplekit.Results(tmp_path / "r.jsonl")
    best = run.coloring_sweep(out, "toy", terms, edges, 3, Args, None, {"max_constraints": 100})
    assert best == [[0, 1], [2, 3]]
    rows = out.rows()
    assert [r["k"] for r in rows] == [3, 2, 1] and [r["valid"] for r in rows] == [True, True, False]


def test_sweep_skips_models_over_the_plan_limit(canned, tmp_path):
    out = examplekit.Results(tmp_path / "r.jsonl")
    assert run.coloring_sweep(out, "toy", [("XI", 1.0)] * 5, [], 3, Args, None, {"max_constraints": 2}) is None
    assert out.rows()[0]["skipped"] == "max_constraints" and not canned.calls


def test_run_and_plot_offline(canned, tmp_path):
    run.main(["--out", str(tmp_path / "o"), "--instances", "hubbard_2x2", "--k-steps", "1", "--budget", "0.2",
              "--max-qiskit-trials", "1"])
    rows = examplekit.load_rows([tmp_path / "o"])
    kinds = {r["kind"] for r in rows}
    assert {"meta", "license", "instance", "grouping", "solve", "result", "priced"} <= kinds
    assert all(r["valid"] for r in rows if r["kind"] == "grouping")
    assert all(r["input_diagonalizes"] for r in rows if r["kind"] == "result")
    assert {r["compiler"] for r in rows if r["kind"] == "priced"} == {"4colors", "qiskit"}
    plot.main([str(tmp_path / "o"), "--figs", str(tmp_path / "f")])
    assert (tmp_path / "f" / "selection.md").exists() and (tmp_path / "f" / "groups.png").exists()
