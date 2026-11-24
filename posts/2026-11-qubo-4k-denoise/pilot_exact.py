"""E12-0: does the exact reference scale? The exact minimum by s-t min cut only, at each size: no API call.

    python pilot_exact.py --out $RUN/out --sizes 1080p 1440p

One `exact` row per size in <out>/results.jsonl: pixels, arcs, energy, wall seconds and this process's peak memory.
Run it as a timed record before the main run, on one pinned core: the post quotes these seconds only from
that record. If a size does not finish inside the record's time limit, E12 uses --exact crop for it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import examplekit as bk  # noqa: E402
import mrf  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--sizes", nargs="+", default=["1080p", "1440p"])
    p.add_argument("--noise", type=float, default=0.10)
    p.add_argument("--noise-seed", type=int, default=1)
    p.add_argument("--beta", type=int, default=10)
    a = p.parse_args(argv)
    res = bk.Results(Path(a.out) / "results.jsonl")
    for size in a.sizes:
        h, w = mrf.parse_size(size)
        clean = mrf.synthetic_image(h, w)
        y = mrf.add_flip_noise(clean, a.noise, a.noise_seed)
        lam = mrf.data_weight(a.noise)
        m = mrf.build(y, lam=lam, beta=a.beta)
        t0 = time.perf_counter()
        x, e = mrf.exact_mincut(m)
        wall = time.perf_counter() - t0
        row = res.add(event="exact", solver="mincut", size=size, h=h, w=w, n=h * w, couplings=mrf.qubo_nnz(m),
                      noise=a.noise, noise_seed=a.noise_seed, lam=lam, beta=a.beta, energy=e,
                      energy_check=mrf.energy(m, x), wall_s=round(wall, 3), peak_rss_gb=round(bk.peak_rss_gb(), 3),
                      pixel_error=mrf.pixel_error(x, clean), noisy_pixel_error=mrf.pixel_error(y, clean))
        print(f"{size}: {h * w:,} pixels, exact energy {e}, {wall:.1f} s, peak {row['peak_rss_gb']} GB")
    return res.rows()


if __name__ == "__main__":
    main()
