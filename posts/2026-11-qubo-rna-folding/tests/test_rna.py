import itertools
import json
from fractions import Fraction

import pytest

import plot
import rna
import run


def test_stems_of_a_tiny_hairpin_by_hand():
    # GGGAAACCC: G0-C8, G1-C7, G2-C6 stack around the AAA loop; every sub-stem of length >= 2 listed by hand
    assert rna.stems("GGGAAACCC", min_stem=3) == [rna.Stem(0, 8, 3)]
    assert rna.stems("GGGAAACCC", min_stem=2) == [rna.Stem(0, 7, 2), rna.Stem(0, 8, 2), rna.Stem(0, 8, 3),
                                                  rna.Stem(1, 7, 2), rna.Stem(1, 8, 2)]
    # a loop shorter than min_loop is refused: GGGACCC has one unpaired base after three pairs
    assert rna.stems("GGGACCC", min_stem=3, min_loop=2) == []
    assert rna.stems("GGGACCC", min_stem=3, min_loop=1) == [rna.Stem(0, 6, 3)]
    # G-U wobble pairs count; T is read as U
    assert rna.stems("GGGCAAUUU") == [rna.Stem(0, 8, 3)]
    assert rna.stems("gggcaattt") == [rna.Stem(0, 8, 3)]
    with pytest.raises(ValueError):
        rna.clean("GGXAA")


def test_overlap_and_crossing():
    a, b = rna.Stem(0, 10, 2), rna.Stem(5, 15, 2)          # 0 < 5 < 10 < 15: a pseudoknot
    c, d = rna.Stem(0, 20, 2), rna.Stem(5, 10, 2)          # nested
    e = rna.Stem(1, 9, 2)                                  # shares base 1 with a
    assert rna.crossing(a, b) and rna.crossing(b, a)
    assert not rna.crossing(c, d) and not rna.overlap(c, d)
    assert rna.overlap(a, e) and not rna.crossing(a, e)


def _small_case():
    seq = rna.random_sequence(36, 7)
    st = rna.stems(seq)
    assert 4 <= len(st) <= 16, len(st)
    return seq, st


def test_model_energy_is_eq13_on_every_feasible_assignment():
    seq, st = _small_case()
    m = rna.build_model(seq, st)
    for bits in itertools.product((0, 1), repeat=len(st)):
        x = list(bits)
        if rna.compatible(st, x):
            assert m.energy(x) == rna.SCALE * rna.energy(st, x)


def test_eq13_by_hand_on_two_stems():
    st = [rna.Stem(0, 10, 2), rna.Stem(5, 15, 3)]          # mu = 3; crossing, so delta = c_P = 1/2
    assert rna.energy(st, [1, 0]) == 10 * (4 - 12 + 9) - 4
    assert rna.energy(st, [0, 1]) == -9
    assert rna.energy(st, [1, 1]) == 6 - 9 - Fraction(1, 2) * 2 * 3


@pytest.mark.parametrize("kind", ["base", "pairs"])
def test_rows_admit_exactly_the_non_overlapping_sets(kind):
    seq, st = _small_case()
    m = rna.build_model(seq, st, rows=kind)
    for bits in itertools.product((0, 1), repeat=len(st)):
        assert bool(m.feasible(list(bits))) == rna.compatible(st, bits)


def test_base_rows_are_fewer_than_pair_rows_here():
    seq, st = _small_case()
    assert len(rna.overlap_rows(st, len(seq), "base")) <= len(rna.overlap_rows(st, len(seq), "pairs"))


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_exact_reference_equals_brute_force(seed):
    st = rna.stems(rna.random_sequence(36, seed))
    if len(st) > 16:
        st = st[:16]
    e, x = rna.brute_force(st)
    ref = rna.exact_reference(st)
    assert ref["status"] == "exact"
    assert ref["energy"] == e
    assert rna.compatible(st, ref["x"]) and rna.energy(st, ref["x"]) == e


def test_exact_reference_gives_up_honestly():
    st = rna.stems(rna.random_sequence(80, 2))
    ref = rna.exact_reference(st, max_nodes=50)
    assert ref["status"] == "gave_up" and rna.compatible(st, ref["x"])


def test_dot_bracket_nested_and_pseudoknot():
    n = 16
    assert rna.dot_bracket(n, [rna.Stem(0, 15, 2), rna.Stem(4, 11, 2)]) == "((..((....))..))"
    s = rna.dot_bracket(n, [rna.Stem(0, 10, 2), rna.Stem(5, 15, 2)])
    assert s == "((...[[..))...]]"
    assert rna.has_pseudoknot([rna.Stem(0, 10, 2), rna.Stem(5, 15, 2)])


def test_fasta_reader(tmp_path):
    p = tmp_path / "x.fa"
    p.write_text(">one\nGGGAAA\nCCC\n>two\nacgt\n")
    assert rna.read_fasta(p) == [("one", "GGGAAACCC"), ("two", "ACGU")]


def test_run_plumbing_with_a_canned_answer(tmp_path, canned):
    seq, st = _small_case()
    ref = rna.exact_reference(st)
    canned.solution = ref["x"]                              # a fixed answer chosen by the test
    run.main(["--out", str(tmp_path), "--sequence", seq, "--seeds", "1", "2", "--time-limit", "1"])
    rows = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    inst = [r for r in rows if r["kind"] == "instance"]
    sol = [r for r in rows if r["kind"] == "solve"]
    assert len(inst) == 1 and inst[0]["reference_status"] == "exact" and len(sol) == 2
    for r in sol:
        assert r["feasible"] and r["model_feasible"] and r["matches_exact"]
        assert r["solver_objective"] == r["energy_python"] == r["energy_model"]
        assert r["engine_version"] == "canned" and r["service_constraints"]["feasible"]
    assert canned.calls[0]["m"] == inst[0]["n_rows"] > 0

    canned.solution = None                                  # all zeros: feasible, energy 0, no match
    run.main(["--out", str(tmp_path / "b"), "--sequence", seq, "--rows", "pairs"])
    rows = [json.loads(l) for l in (tmp_path / "b" / "results.jsonl").read_text().splitlines()]
    s = [r for r in rows if r["kind"] == "solve"][0]
    assert s["rows_kind"] == "pairs" and s["energy"] == 0 and s["matches_exact"] is (ref["energy"] == 0)


def test_plot_on_hand_made_rows(tmp_path):
    seq = "GGGAAACCCAAAGGGAAACCC"
    pairs = [[0, 8], [1, 7], [2, 6], [12, 20], [13, 19], [14, 18]]
    rows = [{"kind": "instance", "instance": "hand", "sequence": seq, "length": len(seq), "n_stems": 6,
             "reference_status": "exact", "reference_pairs": pairs},
            {"kind": "solve", "instance": "hand", "seed": 1, "pairs": pairs, "matches_exact": True, "energy": -18.0},
            {"kind": "instance", "instance": "miss", "sequence": seq, "length": len(seq), "n_stems": 6,
             "reference_status": "exact", "reference_pairs": pairs},
            {"kind": "solve", "instance": "miss", "seed": 1, "pairs": [], "matches_exact": False, "energy": 0.0}]
    p = tmp_path / "results.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    plot.main([str(p), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "arcs_hand.png").exists()
    assert not (tmp_path / "fig" / "arcs_miss.png").exists()       # not equal to the exact optimum: not drawn
    assert (tmp_path / "fig" / "size_ladder.png").exists()
