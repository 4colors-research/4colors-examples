"""Review previews of the figures: one HTML page per cell (two arms side by side: heat maps in a row, timelines
stacked on one shared x scale) and, optionally, a PNG screenshot of it with headless Chrome.

    python make_preview.py inspect/                                  # Qiskit O3 vs ours (no budget), every cell
    python make_preview.py inspect/ --arms tket_tk2 ours_T1 --cells bv-n48 --png

Reads the SVGs inspect_cell.py wrote into <inspect dir>/<device>/<cell>/ and writes preview-<a>-<b>.html beside them
(and preview-<cell>-<device>[-<pair>].png into the inspect dir with --png). Offline rendering of existing files: no
compiler, no records written. Previews are for layout review; the post's figures are built from the recorded run's
report.json by whoever builds them, with the same scales.
"""
import argparse
import glob
import json
import os
import subprocess

CSS = """body{margin:0;background:#fcfcfb;font-family:Inter,system-ui,sans-serif;color:#0b0b0b}
.wrap{padding:16px;display:flex;flex-direction:column;gap:12px;width:max-content}
h1{font-size:17px;font-weight:600;margin:0}
p{font-size:12.5px;color:#52514e;margin:0;max-width:880px;line-height:1.45}
.row{display:flex;gap:16px;align-items:flex-start}
.col{display:flex;flex-direction:column;gap:12px;align-items:flex-start}
img{display:block;border:1px solid #e4e3df}"""


def page(cell, a, b, rep):
    sc = rep.get("figure_scale", {})
    note = (f"Same scales in both panels: heat-map color and width up to {sc.get('heat_vmax')} gates per coupler; "
            f"timelines at {sc.get('px_per_layer')} px per two-qubit layer and {sc.get('px_per_qubit')} px per qubit, "
            f"plot sized for the longest arm of this cell ({sc.get('n_layers')} layers) and the most qubits any arm "
            f"uses ({sc.get('n_rows')}), so every panel of the cell is directly comparable.")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{cell} {rep['device']}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>{cell} on {rep['device']}</h1><p>{note}</p>
<div class="row"><img src="heat-{a}.svg"><img src="heat-{b}.svg"></div>
<div class="col"><img src="timeline-{b}.svg"><img src="timeline-{a}.svg"></div>
</div></body></html>"""


def shoot(html, png):
    cmd = ["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-sandbox",
           "--force-device-scale-factor=1", f"--screenshot={png}", "--window-size=1900,2400",
           "file://" + os.path.abspath(html)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    from PIL import Image, ImageChops                       # crop the empty window around the page content
    im = Image.open(png).convert("RGB")
    bg = Image.new("RGB", im.size, im.getpixel((im.width - 1, im.height - 1)))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        im.crop((0, 0, min(im.width, box[2] + 16), min(im.height, box[3] + 16))).save(png)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inspect_dir")
    ap.add_argument("--arms", nargs=2, default=["qiskit_o3", "ours_inst"])
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--devices", nargs="*")
    ap.add_argument("--png", action="store_true", help="also screenshot each page (needs google-chrome and PIL)")
    a = ap.parse_args(argv)
    out = []
    for rj in sorted(glob.glob(os.path.join(a.inspect_dir, "*", "*", "report.json"))):
        rep = json.load(open(rj))
        cell, dev = rep["circuit"], rep["device"]
        if (a.cells and cell not in a.cells) or (a.devices and dev not in a.devices):
            continue
        if not all(arm in rep["arms"] for arm in a.arms):
            continue
        html = os.path.join(os.path.dirname(rj), f"preview-{a.arms[0]}-{a.arms[1]}.html")
        with open(html, "w") as f:
            f.write(page(cell, a.arms[0], a.arms[1], rep))
        out.append(html)
        if a.png:
            suffix = "" if a.arms == ["qiskit_o3", "ours_inst"] else f"-{a.arms[0]}"
            png = os.path.join(a.inspect_dir, f"preview-{cell}-{dev}{suffix}.png")
            shoot(html, png)
            out.append(png)
    print("\n".join(out))
    return out


if __name__ == "__main__":
    main()
