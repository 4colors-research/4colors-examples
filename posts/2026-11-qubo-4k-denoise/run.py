"""Denoise a 1080p or 1440p binary image with one QUBO (4K: see the post), and score it.

    python run.py --out results/1440p --size 1440p --variant submodular --budgets 10 60
    python run.py --out results/1440p-stripes --size 1440p --variant stripes --budgets 10 60
    python run.py --out results/small --size 640x360 --budgets 5 --exact full     # a quick first try

Steps: draw a clean test image (or --image yours.png), flip each pixel with probability --noise, build the
MRF, write it as a 4CQB file, upload it once, solve it by id at each budget, re-score every answer here.
Rows in <out>/results.jsonl:
  event=solve     ours: energy (re-evaluated), pixel error rate, wall and billed seconds
  event=icm       the local baseline (iterated conditional modes from the noisy image)
  event=exact     the exact minimum by s-t min cut (submodular variant only), with its wall time and memory
  event=noisy     the noisy image itself, for scale
Every row carries `instance` (variant, size, noise, seed, region); `region` is "full" or "crop".
Images (before, after, difference, exact) go to <out>/images.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

import examplekit as bk
import fourcolors.qubo
import mrf


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--size", default="1080p", help="1080p, 1440p, 4k or WxH")
    p.add_argument("--image", default=None, help="your own image (public domain or yours), thresholded to 0/1")
    p.add_argument("--variant", choices=["submodular", "stripes"], default="submodular")
    p.add_argument("--noise", type=float, default=0.10, help="flip probability")
    p.add_argument("--noise-seed", type=int, default=1)
    p.add_argument("--beta", type=int, default=10, help="smoothing weight (integer)")
    p.add_argument("--gamma", type=int, default=None, help="stripe weight; default 2 * beta")
    p.add_argument("--budgets", type=float, nargs="+", default=[10, 60])
    p.add_argument("--solver-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--exact", choices=["auto", "full", "crop", "none"], default="auto",
                   help="exact reference: whole image, a crop solved on its own, or none; auto = full up to "
                        "--exact-max-pixels, else crop")
    p.add_argument("--exact-max-pixels", type=float, default=9e6)
    p.add_argument("--crop", default="1024x1024", help="crop size WxH for --exact crop (from the image center)")
    p.add_argument("--scratch", default=None)
    p.add_argument("--keep-file", action="store_true")
    p.add_argument("--preview-width", type=int, default=960, help="also write downscaled images (0: none)")
    return p.parse_args(argv)


def solve_region(a, res, client, m, clean, noisy, tag, base, images):
    """Upload one MRF and solve it at every budget and seed; returns the best answer found."""
    scratch = Path(a.scratch or Path(a.out) / "scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"{tag}.4cqb.zst"
    t0 = time.perf_counter()
    info = mrf.write_qubo(path, m)
    res.add(event="write", **base, file_bytes=info["bytes_on_disk"], write_s=round(time.perf_counter() - t0, 3),
            peak_rss_gb=round(bk.peak_rss_gb(), 3))
    t0 = time.perf_counter()
    input_id = fourcolors.qubo.upload(str(path), client=client)
    res.add(event="upload", **base, input_id=input_id, upload_wall_s=round(time.perf_counter() - t0, 3),
            matches_local_id=info["input_id"] in (None, input_id))
    if not a.keep_file:
        os.remove(path)
    best = None
    for T in a.budgets:
        for seed in a.solver_seeds:
            t0 = time.perf_counter()
            r = fourcolors.qubo.solve(input_id=input_id, time_limit_s=T, seed=seed, client=client)
            wall = time.perf_counter() - t0
            row = dict(event="solve", solver="4colors", **base, budget_s=T, solver_seed=seed, wall_s=round(wall, 3),
                       status=r.status, objective_reported=r.objective,
                       engine_version=(getattr(r, "engine", None) or {}).get("version"), client_version=bk.client_version())
            job = getattr(r, "job", None)
            for k in ("id", "billed_s", "run_s", "wait_s"):
                row["job_" + k] = getattr(job, k, None) if job is not None else None
            if r.solution is not None:
                x = np.asarray(r.solution, dtype=np.uint8).reshape(m["h"], m["w"])
                e = mrf.energy(m, x)
                row.update(energy=e, agrees=(e == r.objective), pixel_error=mrf.pixel_error(x, clean))
                if best is None or e < best[0]:
                    best = (e, x, T, seed)
            res.add(**row)
            print(f"  {tag} T={T:g}s seed={seed}: E={row.get('energy')} error={row.get('pixel_error')}")
    return best


def main(argv=None, client=None):
    a = parse(argv)
    out = Path(a.out)
    res = bk.Results(out / "results.jsonl")
    h, w = mrf.parse_size(a.size)
    stripes = mrf.stripe_rect(h, w) if a.variant == "stripes" else None
    gamma = 2 * a.beta if a.gamma is None else a.gamma
    clean = mrf.load_image(a.image, h, w) if a.image else mrf.synthetic_image(h, w, stripes)
    noisy = mrf.add_flip_noise(clean, a.noise, a.noise_seed)
    lam = mrf.data_weight(a.noise)
    m = mrf.build(noisy, lam=lam, beta=a.beta, stripes=stripes, gamma=gamma)
    name = f"{a.variant}-{w}x{h}-eps{a.noise:g}-s{a.noise_seed}"
    print(f"{name}: {h * w:,} pixels, {mrf.qubo_nnz(m):,} couplings; service memory about "
          f"{bk.service_memory_gb(h * w):.1f} GB")
    base = dict(instance=name, region="full", variant=a.variant, h=h, w=w, n=h * w, noise=a.noise, lam=lam,
                beta=a.beta, gamma=gamma if stripes else 0, submodular=m["submodular"])
    res.add(event="noisy", **base, energy=mrf.energy(m, noisy), pixel_error=mrf.pixel_error(noisy, clean))

    t0 = time.perf_counter()
    xi, sweeps = mrf.icm(m, noisy)
    res.add(event="icm", solver="icm", **base, wall_s=round(time.perf_counter() - t0, 3), sweeps=sweeps,
            energy=mrf.energy(m, xi), pixel_error=mrf.pixel_error(xi, clean))

    client = client if client is not None else bk.client()
    best = solve_region(a, res, client, m, clean, noisy, name, base, out / "images")

    exact_mode = a.exact
    if exact_mode == "auto":
        exact_mode = "full" if h * w <= a.exact_max_pixels else "crop"
    x_exact = None
    if m["submodular"] and exact_mode == "full":
        t0 = time.perf_counter()
        x_exact, e = mrf.exact_mincut(m)
        res.add(event="exact", solver="mincut", **base, energy=e, wall_s=round(time.perf_counter() - t0, 3),
                peak_rss_gb=round(bk.peak_rss_gb(), 3), pixel_error=mrf.pixel_error(x_exact, clean))
    elif m["submodular"] and exact_mode == "crop":
        # The whole image is too large for the exact reference here: the crop is its own, smaller model,
        # solved exactly and by us at the same budgets (extra API jobs), and scored on its own.
        cw, ch = (int(v) for v in a.crop.lower().split("x"))
        r0, c0 = max(0, (h - ch) // 2), max(0, (w - cw) // 2)
        r1, c1 = min(h, r0 + ch), min(w, c0 + cw)
        mc = mrf.crop(m, noisy, r0, r1, c0, c1)
        cbase = dict(base, region=f"crop {c1 - c0}x{r1 - r0} at ({r0},{c0})", n=mc["h"] * mc["w"])
        t0 = time.perf_counter()
        xc, e = mrf.exact_mincut(mc)
        res.add(event="exact", solver="mincut", **cbase, energy=e, wall_s=round(time.perf_counter() - t0, 3),
                peak_rss_gb=round(bk.peak_rss_gb(), 3), pixel_error=mrf.pixel_error(xc, clean[r0:r1, c0:c1]))
        solve_region(a, res, client, mc, clean[r0:r1, c0:c1], noisy[r0:r1, c0:c1], name + "-crop", cbase, None)
    if best is not None:
        mrf.save_panels(out / "images", name, clean, noisy, best[1], x_exact, a.preview_width)
    return res.rows()


if __name__ == "__main__":
    main()
