"""Offline tests: the generator, the streaming 4CQB writer, the evaluator and run.py with canned API answers (a fixed all-+1 assignment)."""
import itertools
import json
import types

import numpy as np
import pytest
import scipy.sparse as sp

import examplekit as bk
import fourcolors
import fourcolors.qubo as fq
import plot
import run
import spinglass as sg


def naive_ea3d(L, seed):
    """The same instance by explicit loops over sites and directions (an independent re-derivation)."""
    terms = {}
    for z in range(L):
        for y in range(L):
            for x in range(L):
                i = (z * L + y) * L + x
                nbrs = [(z * L + y) * L + (x + 1) % L, (z * L + (y + 1) % L) * L + x, (((z + 1) % L) * L + y) * L + x]
                for d, j in enumerate(nbrs):
                    J = int(bk.ea3d_coupling(seed, np.array([i]), d)[0])
                    terms[(min(i, j), max(i, j))] = -J
    return terms


@pytest.mark.parametrize("L", [3, 4, 5])
def test_lattice_counts_degree_and_couplings(L):
    r, c, v = bk.ea3d_arrays(L, seed=7)
    n = L ** 3
    assert len(r) == bk.ea3d_nnz(L) == 3 * n
    assert np.all(r < c) and c.max() < n
    assert set(np.unique(v).tolist()) <= {-1, 1}
    deg = np.bincount(np.concatenate([r, c]), minlength=n)
    assert np.all(deg == 6)                                   # every site has 6 neighbors
    pairs = r * n + c
    assert len(np.unique(pairs)) == len(pairs)                # no duplicate bond
    assert np.all(np.diff(pairs) > 0)                         # sorted by (row, col)


def test_matches_explicit_loops_including_wrap():
    L, seed = 4, 3
    r, c, v = bk.ea3d_arrays(L, seed)
    got = {(int(a), int(b)): int(w) for a, b, w in zip(r, c, v)}
    assert got == naive_ea3d(L, seed)
    # the periodic bond from x = L-1 back to x = 0 in row 0 is present
    assert (0, L - 1) in got and (0, L * (L - 1)) in got and (0, L * L * (L - 1)) in got


def test_deterministic_by_seed_and_independent_of_slab_size():
    L = 6
    a = bk.ea3d_arrays(L, 11)
    b = [np.concatenate(x) for x in zip(*bk.ea3d_chunks(L, 11, max_sites=36))]   # one plane per slab
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.array_equal(a[2], bk.ea3d_arrays(L, 11)[2])
    assert not np.array_equal(a[2], bk.ea3d_arrays(L, 12)[2])
    assert abs(a[2].mean()) < 0.2                             # about half +1, half -1


def test_stream_writer_is_byte_identical_to_the_client_writer(tmp_path):
    L, seed = 4, 5
    r, c, v = bk.ea3d_arrays(L, seed)
    n = L ** 3
    model = fq.to_model(sp.coo_matrix((v, (r, c)), shape=(n, n)), vartype="SPIN")
    raw = tmp_path / "x.4cqb"
    info = bk.write_4cqb_stream(raw, n=n, nnz=len(r), chunks=lambda: bk.ea3d_chunks(L, seed, max_sites=16),
                                vartype="SPIN", compress=False)
    assert raw.read_bytes() == fq.write_4cqb(model)
    assert info["bytes_raw"] == sg.size_report(L)["file_raw_gb"] * 1e9


def test_zstd_file_round_trips_through_read_file(tmp_path):
    L, seed = 3, 2
    path = tmp_path / "x.4cqb.zst"
    info = sg.write_instance(path, L, seed)
    m = fourcolors.read_file(path)
    assert (m.n, m.nnz, m.vartype) == (27, 81, "SPIN")
    r, c, v = bk.ea3d_arrays(L, seed)
    assert np.array_equal(m.row, r) and np.array_equal(m.col, c) and np.array_equal(m.val, v)
    # the id computed while writing is the one the client computes for the upload
    from fourcolors.optimizer import _inputs
    assert info["input_id"] == _inputs.file_model_id(path)
    _inputs.check_one_zstd_frame(path)


def test_writer_refuses_unsorted_terms(tmp_path):
    bad = lambda: iter([(np.array([1, 0]), np.array([2, 1]), np.array([1, 1]))])
    with pytest.raises(ValueError, match="sorted"):
        bk.write_4cqb_stream(tmp_path / "b", n=3, nnz=2, chunks=bad, compress=False)


def test_ising_energy_agrees_with_brute_force_enumeration():
    rng = np.random.default_rng(0)
    n = 10
    r, c = np.triu_indices(n, 1)
    keep = rng.random(len(r)) < 0.4
    r, c = r[keep], c[keep]
    v = rng.integers(-3, 4, len(r))
    h = rng.integers(-2, 3, n)
    for bits in itertools.product((-1, 1), repeat=n):
        s = np.array(bits)
        direct = sum(int(w) * bits[a] * bits[b] for a, b, w in zip(r, c, v)) + sum(int(h[i]) * bits[i] for i in range(n))
        assert bk.ising_energy(r, c, v, s, linear=h) == direct


def test_slab_energy_matches_the_model_energy():
    L, seed = 5, 9
    s = np.random.default_rng(1).choice([-1, 1], L ** 3)
    r, c, v = bk.ea3d_arrays(L, seed)
    m = fq.to_model(sp.coo_matrix((v, (r, c)), shape=(L ** 3,) * 2), vartype="SPIN")
    assert bk.ea3d_energy(L, seed, s) == m.energy(s)
    sc = sg.score(L, seed, s)
    assert sc["energy_per_spin"] == pytest.approx(m.energy(s) / L ** 3)


def test_size_report_for_the_post_sizes():
    assert sg.size_report(100)["n"] == 1_000_000
    assert sg.size_report(464)["n"] == 99_897_344
    assert sg.size_report(464)["file_raw_gb"] < 6         # 56 bytes per spin raw
    assert sg.size_report(464)["generator_mem_gb"] < 1


class FakeClient:
    def license(self):
        return {"solvers": [{"id": "optimizer.qubo", "model_limits": {"max_variables": None}}]}


def test_run_end_to_end_offline(tmp_path, monkeypatch):
    """upload() then solve(input_id=...) at two budgets, with canned answers for the two client calls."""
    held = {}

    def fake_upload(path, client=None):
        m = fourcolors.read_file(path)                        # the uploaded file must be a valid 4CQB file
        held["m"] = m
        return "id-test"

    def fake_solve(input_id=None, time_limit_s=None, seed=None, client=None):
        assert input_id == "id-test"
        m = held["m"]
        s = np.ones(m.n, dtype=int)                     # canned: a fixed answer, nothing solved
        job = types.SimpleNamespace(id="job-1", billed_s=time_limit_s, run_s=time_limit_s, wait_s=0.0)
        return types.SimpleNamespace(status="finished", reason=None, objective=m.energy(s), solution=s.tolist(),
                                     job=job, trajectory=[{"seq": 1, "t_search_s": 0.1, "objective": 0}])

    monkeypatch.setattr(fourcolors.qubo, "upload", fake_upload)
    monkeypatch.setattr(fourcolors.qubo, "solve", fake_solve)
    out = tmp_path / "out"
    rows = run.main(["--out", str(out), "--L", "4", "--budgets", "1", "2", "--solver-seeds", "0", "1",
                     "--save-solutions"], client=FakeClient())
    solves = [r for r in rows if r["event"] == "solve"]
    assert len(solves) == 4 and all(r["agrees"] for r in solves)
    assert not list((out / "scratch").glob("*.zst"))          # deleted after the upload
    assert len(list((out / "solutions").glob("*.npy"))) == 4
    t = plot.table(bk.Results(out / "results.jsonl").rows())
    assert set(t) == {(64, 1.0), (64, 2.0)}
    plot.main([str(out / "results.jsonl"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "gap_by_budget.png").exists()
    json.dumps(rows)


def test_plan_only_needs_no_client(tmp_path):
    rep = run.main(["--out", str(tmp_path), "--L", "293", "--plan-only"], client=None)
    assert rep["n"] == 293 ** 3
