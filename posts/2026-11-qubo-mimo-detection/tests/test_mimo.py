import itertools
import json
import math

import numpy as np
import pytest

import mimo
import plot
import run


@pytest.mark.parametrize("mod", ["qpsk", "16qam"])
def test_ising_energy_plus_constant_is_the_residual_on_every_assignment(mod):
    inst = mimo.generate(2, 3, mod, 6.0, seed=4)          # 4 or 8 spins: every assignment
    quad, lin, const = mimo.ising_terms(inst)
    m = mimo.model(inst)
    for bits in itertools.product((-1, 1), repeat=inst.n_spins):
        s = np.array(bits)
        e = sum(v * s[i] * s[j] for (i, j), v in quad.items()) + float(np.dot(lin, s))
        assert e + const == pytest.approx(mimo.residual(inst, s), rel=1e-9, abs=1e-9)
        assert m.energy(list(bits)) + const == pytest.approx(mimo.residual(inst, s), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("mod,nt,nr,snr", [("qpsk", 4, 4, 0.0), ("qpsk", 3, 5, 8.0), ("16qam", 2, 2, 5.0),
                                           ("16qam", 2, 3, 15.0)])
def test_sphere_decoder_is_exact_ml(mod, nt, nr, snr):
    for seed in range(4):
        inst = mimo.generate(nt, nr, mod, snr, seed)
        best, _ = mimo.brute_force(inst)
        sd = mimo.sphere_decode(inst)
        assert sd["complete"] and sd["nodes"] > 0
        assert sd["residual"] == pytest.approx(best, rel=1e-9, abs=1e-9)
        assert mimo.residual(inst, sd["spins"]) == pytest.approx(best, rel=1e-9, abs=1e-9)


def test_sphere_decoder_node_cap_flags_an_incomplete_search():
    inst = mimo.generate(8, 8, "16qam", 0.0, 1)
    sd = mimo.sphere_decode(inst, max_nodes=20)
    assert not sd["complete"] and sd["nodes"] == 20


def test_mmse_and_zf_recover_symbols_on_a_noiseless_orthogonal_channel():
    k = 2
    nt = 4
    # a unitary complex channel (scaled DFT): orthogonal columns, no noise
    F = np.exp(-2j * np.pi * np.outer(np.arange(nt), np.arange(nt)) / nt) / math.sqrt(nt)
    bits = np.array([1, 0, 0, 1, 1, 1, 0, 0] * 2)
    s = mimo.gray_bits_to_spins(bits, k)
    xr = mimo.spins_to_amplitudes(s, k)
    H, y = mimo.real_form(F, F @ (xr[:nt] + 1j * xr[nt:]))
    inst = mimo.Instance(nt, nt, "16qam", 99.0, 0, H, y, bits, s, sigma2=1e-12)
    for det in (mimo.mmse, mimo.zero_forcing):
        sc = mimo.score(inst, det(inst))
        assert sc["bit_errors"] == 0 and sc["symbol_errors"] == 0 and sc["residual"] == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("k", [1, 2])
def test_bit_spin_amplitude_round_trips(k):
    for code in range(1 << (3 * k)):
        g = (code >> np.arange(3 * k)) & 1
        s = mimo.gray_bits_to_spins(g, k)
        assert set(np.unique(s)) <= {-1, 1}
        assert np.array_equal(mimo.spins_to_gray_bits(s, k), g)
        a = mimo.spins_to_amplitudes(s, k)
        assert set(a) <= set(mimo.levels(k))
        assert np.array_equal(mimo.amplitudes_to_spins(a, k), s)


def test_16qam_labels_are_gray_and_levels_are_natural_binary():
    k = 2
    order = []
    for a in mimo.levels(k):                      # -3, -1, 1, 3
        order.append(tuple(mimo.spins_to_gray_bits(mimo.amplitudes_to_spins(np.array([a]), k), k)))
    assert order == [(0, 0), (0, 1), (1, 1), (1, 0)]
    for u, v in zip(order, order[1:]):            # neighbors differ in exactly one bit
        assert sum(x != y for x, y in zip(u, v)) == 1
    assert mimo.symbol_energy(1) == 2 and mimo.symbol_energy(2) == 10


def test_generate_is_seeded_and_true_spins_have_low_residual_at_high_snr():
    a, b = mimo.generate(4, 4, "qpsk", 30, 7), mimo.generate(4, 4, "qpsk", 30, 7)
    assert np.array_equal(a.H, b.H) and np.array_equal(a.y, b.y)
    assert mimo.score(a, a.s_true)["bit_errors"] == 0
    assert mimo.residual(a, a.s_true) < 0.1 * mimo.residual(a, -a.s_true)


def test_packed_model_is_block_diagonal():
    insts = [mimo.generate(2, 2, "qpsk", 5, s) for s in (1, 2)]
    m, offs = mimo.packed_model(insts)
    assert offs == [0, 4, 8] and m.n == 8
    rng = np.random.default_rng(0)
    s = rng.choice([-1, 1], size=8)
    consts = [mimo.ising_terms(i)[2] for i in insts]
    want = sum(mimo.residual(i, s[o:o + 4]) for i, o in zip(insts, offs))
    assert m.energy(list(s)) + sum(consts) == pytest.approx(want)


def test_wilson_interval():
    lo, hi = mimo.wilson(0, 100)
    assert lo == 0 and 0 < hi < 0.05
    lo, hi = mimo.wilson(50, 100)
    assert lo < 0.5 < hi


def test_run_plumbing_with_a_canned_answer(canned, tmp_path):
    # the canned job answers the true transmitted spins of each frame (fixed by the test; nothing is solved)
    truth = {}

    def answer(call):
        return truth[call["n"]]

    inst = mimo.generate(4, 4, "qpsk", 8.0, 1)
    truth[inst.n_spins] = list(inst.s_true)
    canned.solution = answer
    run.main(["--out", str(tmp_path), "--nt", "4", "--mod", "qpsk", "--snr", "8", "--frames", "1",
              "--time-limit", "0.5", "--seeds", "3"])
    rows = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    arms = sorted(r["arm"] for r in rows)
    assert arms == ["api", "mmse", "sphere", "zf"]
    api = next(r for r in rows if r["arm"] == "api")
    assert api["bit_errors"] == 0 and api["engine_version"] == "canned" and api["seed"] == 3
    assert api["time_limit_s"] == 0.5 and canned.calls[0]["time_limit_s"] == 0.5
    assert api["energy_python"] == pytest.approx(api["solver_objective"])
    sd = next(r for r in rows if r["arm"] == "sphere")
    assert api["ml_residual"] == sd["residual"]
    assert api["is_ml"] == (api["residual"] <= sd["residual"] + 1e-6 * max(1, sd["residual"]))
    assert api["is_ml"] is True or api["residual"] > sd["residual"]
    if api["is_ml"]:
        assert api["t_to_ml_s"] is not None


def test_run_packed_splits_blocks(canned, tmp_path):
    insts = [mimo.generate(2, 2, "qpsk", 6.0, s) for s in (1, 2, 3)]
    canned.solution = lambda call: [v for i in insts[:call["n"] // 4] for v in i.s_true] if call["n"] == 12 \
        else list(insts[0].s_true)
    run.main(["--out", str(tmp_path), "--nt", "2", "--snr", "6", "--frames", "3", "--pack", "3",
              "--time-limit", "1", "--no-sphere"])
    rows = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    api = [r for r in rows if r["arm"] == "api"]
    assert len(canned.calls) == 1 and canned.calls[0]["n"] == 12
    assert [r["block"] for r in api] == [0, 1, 2] and all(r["bit_errors"] == 0 for r in api)
    assert all(r["pack"] == 3 and r["t_to_ml_s"] is None for r in api)


def _row(arm, snr, errors, n_bits=1000, **kw):
    return {"arm": arm, "mod": "qpsk", "nt": 16, "nr": 16, "snr_db": snr, "bit_errors": errors, "n_bits": n_bits,
            **kw}


def test_plot_applies_the_selection_rule(tmp_path):
    rows = [_row("mmse", 4, 150), _row("api", 4, 40, time_limit_s=1.0), _row("sphere", 4, 38, complete=True),
            _row("mmse", 8, 60), _row("api", 8, 58, time_limit_s=1.0), _row("sphere", 8, 50, complete=False),
            _row("api", 8, 1, time_limit_s=10.0)]
    rows += [{"arm": "sphere", "mod": "qpsk", "snr_db": 4, "nt": nt, "nodes": 10 ** (nt // 8), "complete": True,
              "wall_s": 1e-4 * 4 ** (nt // 8), "timed": True} for nt in (8, 16, 32)]
    rows += [{"arm": "api", "mod": "qpsk", "snr_db": 4, "nt": nt, "pack": 0, "block": 0,
              "t_to_ml_s": 0.002} for nt in (8, 16, 32)]
    f = tmp_path / "results.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    plot.main([str(f), "--out", str(tmp_path / "fig"), "--budget", "1"])
    sel = json.loads((tmp_path / "fig" / "selection.json").read_text())
    drawn = {(c["snr_db"]): c["drawn"] for c in sel["ber"]}
    assert drawn == {4: True, 8: False}                    # 58 vs 60 errors overlap: not drawn
    ttm = {c["nt"]: c["drawn"] for c in sel["time_to_ml"]}
    assert ttm == {8: False, 16: False, 32: True}          # sphere 0.4 and 1.6 ms beat 2 ms: not drawn
    for name in ("ber_qpsk_16x16.png", "time_to_ml_qpsk_snr4.png", "sphere_nodes.png", "couplings.png"):
        assert (tmp_path / "fig" / name).stat().st_size > 0
