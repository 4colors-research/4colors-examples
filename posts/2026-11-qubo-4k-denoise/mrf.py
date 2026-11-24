"""Binary image denoising as one QUBO: a Markov random field on the pixel grid.

Labels x_p in {0, 1}. With a noisy binary image y (each pixel flipped with probability eps):

    E(x) = sum_p D_p(x_p) + sum_{p~q} w_pq [x_p != x_q]

- Data term: D_p(l) = lam [y_p != l], lam ~ round(scale * log((1 - eps) / eps)) (an integer).
- Smoothing prior: w_pq = beta >= 0 on every right and down neighbor pair (submodular).
- Stripe prior (non-submodular variant): inside a rectangle, horizontal pairs get w = beta - gamma < 0,
  which rewards neighbors that differ (a vertical-stripe texture). Graph cuts no longer apply there.

As a QUBO (x in {0,1}): [x_p != x_q] = x_p + x_q - 2 x_p x_q, so
    linear_p = D_p(1) - D_p(0) + sum_q w_pq,   quad_pq = -2 w_pq,   offset = sum_p D_p(0).
All values are integers, so the file is int64 and energies compare exactly.

Exact reference for the submodular case: one s-t minimum cut (Greig, Porteous and Seheult 1989), here with
scipy's max-flow (Dinic) on a sparse graph, so no pure-Python graph is built.
"""
from __future__ import annotations

import math

import numpy as np

SIZES = {"1080p": (1080, 1920), "1440p": (1440, 2560), "4k": (2160, 3840)}

FOUR_OUTER = [(0.58, 0.05), (0.75, 0.05), (0.75, 0.62), (0.90, 0.62), (0.90, 0.76), (0.75, 0.76),
              (0.75, 0.95), (0.58, 0.95), (0.58, 0.76), (0.10, 0.76), (0.10, 0.63)]
FOUR_HOLE = [(0.58, 0.27), (0.58, 0.62), (0.30, 0.62)]


def parse_size(text: str):
    if text.lower() in SIZES:
        return SIZES[text.lower()]
    w, h = text.lower().split("x")
    return int(h), int(w)


def stripe_rect(h: int, w: int):
    """The stripe region used by the non-submodular variant: a band in the right part of the image."""
    return (h // 8, h // 2, (3 * w) // 5, (19 * w) // 20)


def synthetic_image(h: int, w: int, stripes=None) -> np.ndarray:
    """A clean binary test image (uint8 0/1): the "4" glyph, discs and a ring, drawn with PIL.
    With stripes=(r0, r1, c0, c1), that rectangle holds vertical stripes of period 2 (the texture the
    stripe prior expects). No fonts and no downloads."""
    from PIL import Image, ImageDraw

    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    s = min(h, w)
    ox = w // 20
    d.polygon([(ox + x * s, y * h) for x, y in FOUR_OUTER], fill=255)
    d.polygon([(ox + x * s, y * h) for x, y in FOUR_HOLE], fill=0)
    for k in range(5):                                        # a row of discs of growing size
        r = s * (0.015 + 0.01 * k)
        cx, cy = w * (0.62 + 0.07 * k), h * 0.75
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    cx, cy, r = w * 0.48, h * 0.30, s * 0.12                  # a ring
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    d.ellipse([cx - 0.6 * r, cy - 0.6 * r, cx + 0.6 * r, cy + 0.6 * r], fill=0)
    x = (np.asarray(img) >= 128).astype(np.uint8)
    if stripes is not None:
        r0, r1, c0, c1 = stripes
        x[r0:r1, c0:c1] = (np.arange(c0, c1) % 2).astype(np.uint8)[None, :]
    return x


def load_image(path, h: int, w: int) -> np.ndarray:
    """Your own picture, resized to h x w and thresholded at mid-gray (1 = bright)."""
    from PIL import Image

    img = Image.open(path).convert("L").resize((w, h), Image.LANCZOS)
    return (np.asarray(img) >= 128).astype(np.uint8)


def add_flip_noise(clean: np.ndarray, eps: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (clean ^ (rng.random(clean.shape) < eps)).astype(np.uint8)


def data_weight(eps: float, scale: float = 10.0) -> int:
    return max(1, int(round(scale * math.log((1 - eps) / eps))))


def build(y: np.ndarray, *, lam: int, beta: int, stripes=None, gamma: int = 0) -> dict:
    """The MRF for noisy image y: a = D(1) - D(0) per pixel, weights wr (right pairs) and wd (down pairs)."""
    y = np.asarray(y, dtype=np.int64)
    h, w = y.shape
    a = lam * (1 - 2 * y)                       # D(1) - D(0): +lam where y = 0, -lam where y = 1
    offset = int(lam * y.sum())                  # sum of D(0): lam where y = 1
    wr = np.full((h, w - 1), beta, dtype=np.int64)
    wd = np.full((h - 1, w), beta, dtype=np.int64)
    if stripes is not None:
        r0, r1, c0, c1 = stripes
        wr[r0:r1, c0:c1 - 1] -= gamma            # horizontal pairs inside the rectangle
    return {"h": h, "w": w, "a": a, "wr": wr, "wd": wd, "offset": offset, "lam": lam, "beta": beta,
            "gamma": gamma, "stripes": stripes,
            "submodular": bool((wr >= 0).all() and (wd >= 0).all())}


def crop(m: dict, y: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> dict:
    """The MRF of the crop y[r0:r1, c0:c1] as its own model (its boundary is free)."""
    st = m["stripes"]
    if st is not None:
        st = (max(st[0], r0) - r0, min(st[1], r1) - r0, max(st[2], c0) - c0, min(st[3], c1) - c0)
        st = st if st[0] < st[1] and st[2] < st[3] else None
    return build(y[r0:r1, c0:c1], lam=m["lam"], beta=m["beta"], stripes=st, gamma=m["gamma"])


def energy(m: dict, x) -> int:
    """E(x) from the grid form (the same value as the QUBO form; tests check both)."""
    x = np.asarray(x, dtype=np.int64).reshape(m["h"], m["w"])
    e = m["offset"] + int((m["a"] * x).sum())
    e += int((m["wr"] * (x[:, 1:] != x[:, :-1])).sum()) + int((m["wd"] * (x[1:, :] != x[:-1, :])).sum())
    return e


# ---- the QUBO, written in row blocks --------------------------------------------------------------

def qubo_linear(m: dict) -> np.ndarray:
    lin = m["a"].copy()
    lin[:, :-1] += m["wr"]; lin[:, 1:] += m["wr"]
    lin[:-1, :] += m["wd"]; lin[1:, :] += m["wd"]
    return lin.ravel()


def qubo_nnz(m: dict) -> int:
    return int(np.count_nonzero(m["wr"]) + np.count_nonzero(m["wd"]))


def qubo_chunks(m: dict, rows_per_chunk: int = 256):
    """(row, col, val) of the quadratic terms, sorted by (row, col): for pixel p, (p, p+1) then (p, p+w)."""
    h, w = m["h"], m["w"]
    for r0 in range(0, h, rows_per_chunk):
        r1 = min(h, r0 + rows_per_chunk)
        p = (np.arange(r0, r1)[:, None] * w + np.arange(w)[None, :]).astype(np.int64)
        col = np.stack([p + 1, p + w], axis=2)
        val = np.zeros((r1 - r0, w, 2), dtype=np.int64)
        val[:, :-1, 0] = -2 * m["wr"][r0:r1]
        val[: min(r1, h - 1) - r0, :, 1] = -2 * m["wd"][r0:min(r1, h - 1)]
        row = np.broadcast_to(p[:, :, None], col.shape)
        keep = val != 0
        yield row[keep], col[keep], val[keep]


def write_qubo(path, m: dict, compress: bool = True) -> dict:
    import examplekit as bk

    return bk.write_4cqb_stream(path, n=m["h"] * m["w"], nnz=qubo_nnz(m), chunks=lambda: qubo_chunks(m),
                                linear=qubo_linear(m), offset=m["offset"], vartype="BINARY", compress=compress)


# ---- exact reference (submodular only) and a simple local baseline --------------------------------

def exact_mincut(m: dict):
    """The exact minimizer by one s-t minimum cut: (x, E(x)). Pixels on the sink side get label 1."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    if not m["submodular"]:
        raise ValueError("min cut needs every pairwise weight >= 0 (a submodular energy)")
    h, w = m["h"], m["w"]
    n = h * w
    s, t = n, n + 1
    a = m["a"].ravel()
    p = np.arange(n, dtype=np.int64).reshape(h, w)
    src, dst, cap = [], [], []
    pos = a > 0                                     # cost a x_p: the s->p arc is cut when p is on the sink side
    src.append(np.full(pos.sum(), s)); dst.append(np.flatnonzero(pos)); cap.append(a[pos])
    neg = a < 0                                     # cost a x_p = a + |a| (1 - x_p): p->t arc
    src.append(np.flatnonzero(neg)); dst.append(np.full(neg.sum(), t)); cap.append(-a[neg])
    for u, v, wt in ((p[:, :-1], p[:, 1:], m["wr"]), (p[:-1, :], p[1:, :], m["wd"])):
        k = wt.ravel() > 0
        u, v, wt = u.ravel()[k], v.ravel()[k], wt.ravel()[k]
        src += [u, v]; dst += [v, u]; cap += [wt, wt]
    src, dst, cap = np.concatenate(src), np.concatenate(dst), np.concatenate(cap)
    if cap.size and cap.max() >= 2 ** 31:
        raise ValueError("capacities must fit int32 for scipy's max-flow")
    C = sp.csr_array((cap.astype(np.int32), (src.astype(np.int32), dst.astype(np.int32))), shape=(n + 2, n + 2))
    flow = maximum_flow(C, s, t, method="dinic")
    R = (C - flow.flow).tocsr()                     # residual capacities (reverse arcs carry the flow back)
    R.data[R.data < 0] = 0
    R.eliminate_zeros()
    reach = breadth_first_order(R, s, directed=True, return_predecessors=False)
    x = np.ones(n + 2, dtype=np.uint8)
    x[reach] = 0                                    # the source side keeps label 0
    x = x[:n]
    e = energy(m, x)
    expected = int(flow.flow_value) + m["offset"] + int(a[neg].sum())
    if e != expected:
        raise AssertionError(f"min-cut energy {e} != flow value bound {expected}")
    return x.reshape(h, w), e


def icm(m: dict, x0, max_sweeps: int = 100):
    """Iterated conditional modes on the two checkerboard colors (pixels of one color have no common
    pair, so each half-sweep only lowers the energy). Stops at a local minimum. Returns (x, sweeps)."""
    x = np.asarray(x0, dtype=np.int64).reshape(m["h"], m["w"]).copy()
    wr, wd, a = m["wr"], m["wd"], m["a"]
    color = (np.add.outer(np.arange(m["h"]), np.arange(m["w"])) % 2).astype(bool)
    for sweep in range(1, max_sweeps + 1):
        changed = 0
        for c in (False, True):
            g = 1 - 2 * x                            # what each neighbor adds to E(x_p = 1) - E(x_p = 0)
            d = a.copy()
            d[:, :-1] += wr * g[:, 1:]; d[:, 1:] += wr * g[:, :-1]
            d[:-1, :] += wd * g[1:, :]; d[1:, :] += wd * g[:-1, :]
            new = np.where(d < 0, 1, np.where(d > 0, 0, x))
            mask = (color == c) & (new != x)
            x[mask] = new[mask]; changed += int(mask.sum())
        if not changed:
            return x.astype(np.uint8), sweep
    return x.astype(np.uint8), max_sweeps


def pixel_error(x, clean) -> float:
    return float(np.mean(np.asarray(x).reshape(np.shape(clean)) != clean))


def save_panels(folder, tag, clean, noisy, ours, exact=None, preview_width: int = 0):
    """PNG files: noisy (before), ours (after), difference to the clean image, and the exact answer.
    preview_width > 0 also writes downscaled copies for the post."""
    from pathlib import Path
    from PIL import Image

    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    out = {}
    imgs = {"before": noisy, "after": ours}
    if exact is not None:
        imgs["exact"] = exact
    for name, im in imgs.items():
        out[name] = Image.fromarray(((1 - np.asarray(im)) * 255).astype(np.uint8))   # ink on white
    diff = np.full(clean.shape + (3,), 245, dtype=np.uint8)
    wrong = np.asarray(ours) != clean
    diff[wrong] = (235, 104, 52)                                                    # wrong pixels in orange
    out["difference"] = Image.fromarray(diff)
    paths = {}
    for name, im in out.items():
        paths[name] = folder / f"{tag}-{name}.png"
        im.save(paths[name])
        if preview_width:
            hh = max(1, round(im.height * preview_width / im.width))
            im.resize((preview_width, hh), Image.BOX).save(folder / f"{tag}-{name}-preview.png")
    paths["triptych"] = triptych(folder / f"{tag}-triptych.png", out["before"], out["after"], out["difference"],
                                 panel_width=preview_width or out["before"].width)
    return paths


def triptych(path, before, after, difference, *, panel_width: int, gap: int = 12):
    """The post's hero: noisy input | our answer | pixels that differ from the clean image, side by side on a
    light background, each panel downscaled (box filter) to `panel_width`."""
    from PIL import Image

    hh = max(1, round(before.height * panel_width / before.width))
    panels = [im.convert("RGB").resize((panel_width, hh), Image.BOX) for im in (before, after, difference)]
    sheet = Image.new("RGB", (3 * panel_width + 4 * gap, hh + 2 * gap), (252, 252, 251))
    for k, im in enumerate(panels):
        sheet.paste(im, (gap + k * (panel_width + gap), gap))
    sheet.save(path)
    return path
