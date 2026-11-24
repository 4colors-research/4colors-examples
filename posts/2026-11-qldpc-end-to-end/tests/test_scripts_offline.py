import numpy as np

import examplekit
import codes as cd
import plot
import run


def test_distance_rows_certify_canned_answers(canned, tmp_path):
    hx, hz, _ = cd.code("toric_3")
    L = cd.x_logicals(hx, hz)
    ker = cd.nullspace(hx)

    def answer(call):
        # a hand-picked Z logical for logical 0 (found by enumeration), with slack bits solved by hand; else zeros
        n_vars, rows, lrow = cd.distance_layout(hx, L[0])
        if call["n"] != n_vars:
            return [0] * call["n"]
        for bits in range(1, 2 ** len(ker)):
            e = (np.array([(bits >> i) & 1 for i in range(len(ker))]) @ ker) % 2
            if cd.certify(hx, L[0], e) and e.sum() == 3:
                break
        x = list(e) + [0] * (n_vars - 18)
        for sup, slack in rows + [lrow]:
            s = int(sum(e[j] for j in sup)) // 2
            for t, v in enumerate(slack):
                x[v] = (s >> t) & 1
        return x

    canned.solution = answer

    class A:
        max_logicals, seed = 1, 1

    out = examplekit.Results(tmp_path / "r.jsonl")
    best, reported = run.distance_rows(out, "toric_3", 1.0, A, None)
    assert best == 3 == reported
    r = [x for x in out.rows() if x["kind"] == "distance"][0]
    assert r["certified"] and r["model_feasible"] and r["weight"] == 3


def test_all_zero_answer_is_not_certified(canned, tmp_path):
    class A:
        max_logicals, seed = 1, 1

    out = examplekit.Results(tmp_path / "r.jsonl")
    assert run.distance_rows(out, "toric_3", 1.0, A, None)[0] is None
    assert out.rows()[0]["certified"] is False


def test_compile_and_plot_offline(canned, tmp_path, monkeypatch):
    monkeypatch.setitem(run.STAGES, "tiny", {"distance": ["toric_3"], "compile": ["toric_3"], "max_logicals": 1})
    run.main(["--out", str(tmp_path / "o"), "--stage", "tiny", "--budgets", "1", "--budget", "0.2",
              "--compile-seeds", "1", "--max-qiskit-trials", "1"])
    rows = examplekit.load_rows([tmp_path / "o"])
    res = [r for r in rows if r["kind"] == "result"]
    assert {(r["arm"], r["chip"]) for r in res} == {(a, c) for a in ("4colors", "qiskit") for c in ("tanner", "heavy_hex")}
    assert all(r["syndrome_check"] == "pass" for r in res)            # canned = the input round; Qiskit routed
    plot.main([str(tmp_path / "o"), "--figs", str(tmp_path / "f")])
    assert (tmp_path / "f" / "tanner.png").exists() and (tmp_path / "f" / "selection.md").exists()
