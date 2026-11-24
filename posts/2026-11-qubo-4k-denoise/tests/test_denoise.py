"""Offline tests: MRF energy forms, the 4CQB file, the exact min cut against brute force, ICM, run.py."""
import itertools
import types

import numpy as np
import pytest

import examplekit as bk
import fourcolors
import fourcolors.qubo as fq
import mrf
import plot
import run


def tiny(h=3, w=4, eps=0.2, seed=0, stripes=None, beta=3, gamma=0):
    clean = mrf.synthetic_image(h, w, stripes)
    y = mrf.add_flip_noise(clean, eps, seed)
    return clean, y, mrf.build(y, lam=mrf.data_weight(eps), beta=beta, stripes=stripes, gamma=gamma)


def brute_force(m):
    n = m["h"] * m["w"]
    best = None
    for bits in itertools.product((0, 1), repeat=n):
        e = mrf.energy(m, np.array(bits))
        if best is None or e < best[0]:
            best = (e, bits)
    return best


def test_energy_forms_agree_and_file_round_trips(tmp_path):
    for stripes, gamma in ((None, 0), ((0, 3, 1, 4), 7)):
        _, y, m = tiny(stripes=stripes, gamma=gamma)
        path = tmp_path / "m.4cqb.zst"
        info = mrf.write_qubo(path, m)
        model = fourcolors.read_file(path)
        assert model.n == 12 and model.nnz == mrf.qubo_nnz(m) == info["nnz"]
        rng = np.random.default_rng(3)
        for _ in range(20):
            x = rng.integers(0, 2, 12)
            assert model.energy(x) == mrf.energy(m, x)


def test_stream_file_equals_client_writer(tmp_path):
    _, y, m = tiny(h=4, w=5, stripes=(1, 3, 0, 5), gamma=8)
    raw = tmp_path / "m.4cqb"
    mrf.write_qubo(raw, m, compress=False)
    model = fourcolors.read_file(raw)
    assert raw.read_bytes() == fq.write_4cqb(model)


def test_chunks_cover_every_pair_once():
    _, y, m = tiny(h=5, w=3)
    small = list(mrf.qubo_chunks(m, rows_per_chunk=1))
    big = list(mrf.qubo_chunks(m, rows_per_chunk=100))
    cat = lambda parts, k: np.concatenate([p[k] for p in parts])
    for k in range(3):
        assert np.array_equal(cat(small, k), cat(big, k))
    assert len(cat(big, 0)) == 5 * 2 + 4 * 3


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_min_cut_equals_brute_force(seed):
    _, y, m = tiny(h=3, w=4, eps=0.25, seed=seed, beta=2 + seed)
    x, e = mrf.exact_mincut(m)
    assert e == brute_force(m)[0] == mrf.energy(m, x)


def test_min_cut_refuses_the_stripe_variant():
    _, y, m = tiny(stripes=(0, 3, 0, 4), gamma=10)
    assert not m["submodular"]
    with pytest.raises(ValueError):
        mrf.exact_mincut(m)


def test_icm_never_raises_the_energy_and_stops():
    clean, y, m = tiny(h=12, w=16, eps=0.15, stripes=(2, 8, 4, 12), gamma=6)
    x, sweeps = mrf.icm(m, y)
    assert mrf.energy(m, x) <= mrf.energy(m, y)
    x2, _ = mrf.icm(m, x)                        # a local minimum stays put
    assert np.array_equal(x, x2)


def test_crop_is_its_own_model():
    clean, y, m = tiny(h=6, w=8, stripes=(1, 5, 2, 7), gamma=6)
    c = mrf.crop(m, y, 1, 4, 2, 6)
    assert (c["h"], c["w"]) == (3, 4) and c["stripes"] == (0, 3, 0, 4)


def test_denoising_helps_on_a_small_image():
    clean = mrf.synthetic_image(60, 80)
    y = mrf.add_flip_noise(clean, 0.1, 5)
    m = mrf.build(y, lam=mrf.data_weight(0.1), beta=10)
    x, e = mrf.exact_mincut(m)
    assert mrf.pixel_error(x, clean) < mrf.pixel_error(y, clean)


def test_sizes():
    assert mrf.parse_size("4k") == (2160, 3840) and 2160 * 3840 == 8_294_400
    assert mrf.parse_size("1440p") == (1440, 2560) and 1440 * 2560 == 3_686_400
    assert mrf.parse_size("640x360") == (360, 640)


class FakeClient:
    pass


def test_run_end_to_end_offline(tmp_path, monkeypatch):
    held = {}

    def fake_upload(path, client=None):
        held[path] = fourcolors.read_file(path)
        return path

    def fake_solve(input_id=None, time_limit_s=None, seed=None, client=None):
        model = held[input_id]
        x = np.zeros(model.n, dtype=int)            # the all-background answer, valid if poor
        job = types.SimpleNamespace(id="j", billed_s=time_limit_s, run_s=time_limit_s, wait_s=0)
        return types.SimpleNamespace(status="finished", objective=model.energy(x), solution=x.tolist(), job=job)

    monkeypatch.setattr(fourcolors.qubo, "upload", fake_upload)
    monkeypatch.setattr(fourcolors.qubo, "solve", fake_solve)
    out = tmp_path / "o"
    rows = run.main(["--out", str(out), "--size", "40x30", "--budgets", "1", "--exact", "crop", "--crop", "16x12",
                     "--preview-width", "20"], client=FakeClient())
    events = [r["event"] for r in rows]
    assert events.count("solve") == 2 and events.count("exact") == 1    # full + crop solves, exact on the crop
    assert all(r["agrees"] for r in rows if r["event"] == "solve")
    assert (out / "images").exists() and any((out / "images").glob("*-difference.png"))
    rows += run.main(["--out", str(tmp_path / "s"), "--size", "40x30", "--variant", "stripes", "--budgets", "1"],
                     client=FakeClient())
    t = plot.table(rows)
    kinds = {v["reference"] for k, v in t.items()}
    assert kinds == {"exact", "best found"}
    plot.main([str(out / "results.jsonl"), str(tmp_path / "s" / "results.jsonl"), "--out", str(tmp_path / "f")])
    assert (tmp_path / "f" / "denoise_gap.png").exists()


def test_pilot_exact_rows(tmp_path):
    import pilot_exact

    rows = pilot_exact.main(["--out", str(tmp_path), "--sizes", "24x16", "32x20"])
    assert [r["n"] for r in rows] == [384, 640]
    assert all(r["energy"] == r["energy_check"] and r["pixel_error"] <= r["noisy_pixel_error"] for r in rows)


def test_triptych_is_three_panels(tmp_path):
    from PIL import Image

    h, w = 30, 40
    clean = mrf.synthetic_image(h, w)
    noisy = mrf.add_flip_noise(clean, 0.1, 1)
    paths = mrf.save_panels(tmp_path, "t", clean, noisy, clean, preview_width=20)
    im = Image.open(paths["triptych"])
    assert im.size == (3 * 20 + 4 * 12, round(h * 20 / w) + 2 * 12)
