"""Offline tests for the edge post: the usability rule, the pair verdict, the edge, the PUB order, and
run.py compile + score end to end on canned results (the input circuit returned unchanged; no API call,
no hardware)."""
import json
import math

import pytest
from conftest import load

edge = load("edge")


def test_a_clear_answer_is_usable():
    s = edge.score_counts({"101": 120, "001": 3, "111": 2, "000": 1}, "101")
    assert s["shots"] == 126 and s["k"] == 120 and s["rank"] == 1 and s["runner_up"] == 3
    assert s["margin_z"] == pytest.approx((120 - 3) / math.sqrt(123))
    assert s["usable"]


def test_two_counts_against_one_is_not_an_answer():
    """At 50+ qubits most wrong outcomes are seen once: "most frequent" alone would call 2 vs 1 an answer."""
    s = edge.score_counts({"1" * 8: 2, "0" * 8: 1, "01" * 4: 1}, "1" * 8)
    assert s["rank"] == 1 and not s["usable"]


def test_a_tie_for_first_or_a_missing_answer_is_not_usable():
    assert edge.score_counts({"11": 50, "10": 50}, "11")["rank"] == 2
    assert not edge.score_counts({"11": 50, "10": 50}, "11")["usable"]
    s = edge.score_counts({"10": 7}, "11")
    assert s["k"] == 0 and s["rank"] is None and not s["usable"]


def test_majority_vote_is_a_separate_weaker_decoder():
    counts = {"110": 3, "100": 2, "111": 2, "010": 2}          # right answer "110" by per-bit vote
    assert edge.bitwise_majority(counts) == "110"
    assert not edge.score_counts(counts, "110")["usable"]       # but not usable as a single-shot answer


def test_pair_verdict_rules():
    ours = edge.score_counts({"1": 400, "0": 3600}, "1")
    rival = edge.score_counts({"1": 20, "0": 3980}, "1")
    v = edge.pair_verdict(ours, rival, 0.5, 0.5)
    assert v["verdict"] == "WIN" and v["ratio"] == pytest.approx(20.0)
    assert edge.pair_verdict(ours, rival, 3.2, 0.5)["verdict"] == "AA-FAIL"
    assert edge.pair_verdict(rival, ours, 0.5, 0.5)["verdict"] == "LOSS"
    same = edge.pair_verdict(ours, ours, 0.0, 0.0)
    assert same["verdict"] == "TIE" and same["z"] == 0.0


def test_zero_rival_counts_give_a_lower_bound_ratio():
    ours = edge.score_counts({"1": 100, "0": 3900}, "1")
    rival = edge.score_counts({"0": 4000}, "1")
    v = edge.pair_verdict(ours, rival, 0.0, 0.0)
    assert v["ratio"] == pytest.approx(200.0) and v["ratio_note"]


def test_edge_flip_needs_a_win_and_usable_vs_unusable():
    ours = edge.score_counts({"1" * 6: 300, "0" * 6: 4, "1" * 5 + "0": 3}, "1" * 6)
    rival = edge.score_counts({"1" * 6: 2, "0" * 6: 1, "01" * 3: 1, "10" * 3: 3996}, "1" * 6)
    v = edge.pair_verdict(ours, rival, 0.1, 0.1)
    assert ours["usable"] and not rival["usable"] and v["edge_flip"]


def test_edge_and_first_unusable():
    cells = [{"n": n, "arm": "a", "usable": u} for n, u in ((32, True), (48, True), (64, False), (80, False))]
    cells += [{"n": n, "arm": "b", "usable": u} for n, u in ((32, True), (48, False), (64, True), (80, False))]
    cells += [{"n": 32, "arm": "c", "usable": True, "aa_ok": False}]
    assert edge.edge(cells, "a") == {"arm": "a", "edge_n": 48, "monotone": True, "sizes": [32, 48, 64, 80]}
    assert edge.edge(cells, "b")["edge_n"] == 64 and not edge.edge(cells, "b")["monotone"]
    assert edge.edge(cells, "c")["edge_n"] is None
    assert edge.first_unusable(cells, "a") == 64 and edge.first_unusable(cells, "b") == 48


def test_shown_is_wins_only():
    pairs = [{"n": 1, "verdict": "WIN"}, {"n": 2, "verdict": "TIE"}, {"n": 3, "verdict": "LOSS"},
             {"n": 4, "verdict": "AA-FAIL"}]
    assert [p["n"] for p in edge.shown(pairs)] == [1]


def test_pub_order_is_fixed_and_a_permutation():
    keys = [f"k{i}" for i in range(20)]
    assert edge.pub_order(keys) == edge.pub_order(keys)
    assert sorted(edge.pub_order(keys)) == sorted(keys) and edge.pub_order(keys) != keys


def test_qpu_estimate():
    assert edge.qpu_estimate_s(28, 2000, s_per_shot=4e-4, overhead_s=5) == pytest.approx(27.4)


def test_compile_and_score_offline(canned, tmp_path):
    """run.py compile on FakeFez with canned results (the input circuit returned unchanged, nothing compiled), then score
    synthetic counts: ours clean, Qiskit noise only."""
    run = load("run")
    out = tmp_path / "r"
    run.main(["compile", "--out", str(out), "--sizes", "6,8", "--budget", "0.01", "--shots", "100"])
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    res = [r for r in rows if r["kind"] == "result"]
    assert {(r["circuit_id"], r["arm"]) for r in res} == {(f"bv_{n}", a) for n in (6, 8) for a in ("qiskit", "4colors")}
    assert all(r["two_qubit_gates"] >= 1 for r in res)
    assert all(r["noiseless_ok"] for r in rows if r["kind"] == "check")
    ours_calls = canned.calls
    assert len(ours_calls) == 2 and all(c["time_budget_s"] >= 1 for c in ours_calls)
    assert all(c["approximation_mode"] == "exact" for c in ours_calls)
    assert (out / "snapshot.json").exists()
    pubs = json.loads((out / "pubs.json").read_text())["pubs"]
    assert len(pubs) == 2 * 2 * run.REPS
    counts = {}
    for p in pubs:
        wrong = "".join("1" if c == "0" else "0" for c in p["answer"])
        counts[p["key"]] = {p["answer"]: 90, wrong: 10} if p["arm"] == "4colors" else {wrong: 99, p["answer"]: 1}
    (out / "counts.json").write_text(json.dumps(counts))
    run.main(["score", "--out", str(out), "--counts", str(out / "counts.json")])
    edges = json.loads((out / "edges.json").read_text())
    assert edges["edges"]["4colors"]["edge_n"] == 8 and edges["edges"]["qiskit"]["edge_n"] is None
    assert edges["shown"] == [6, 8]
    pairs = [json.loads(x) for x in (out / "pairs.jsonl").read_text().splitlines()]
    assert all(p["edge_flip"] for p in pairs)


def test_noiseless_check_passes_fails_and_declines():
    from qiskit import QuantumCircuit, transpile

    import circuitkit as ck

    run = load("run")
    qc, _, answer = run.bv_circuit(12)
    out = transpile(qc, target=ck.fez().target, optimization_level=1, seed_transpiler=1)   # 156-qubit ISA
    assert run.noiseless_ok(out, answer) is True
    wrong = ("0" if answer[0] == "1" else "1") + answer[1:]
    assert run.noiseless_ok(out, wrong) is False
    t = QuantumCircuit(1, 1)
    t.t(0)
    t.measure(0, 0)
    assert run.noiseless_ok(t, "0") is None                  # not Clifford: no check, never a pass
