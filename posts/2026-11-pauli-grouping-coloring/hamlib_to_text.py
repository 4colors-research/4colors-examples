"""Turn HamLib HDF5 files into the OpenFermion-text files run.py reads (one Hamiltonian per .txt file).

    pip install h5py
    python hamlib_to_text.py path/to/hamlib_file.hdf5 --out hamlib_txt [--match "JW"] [--max-terms 1500]

HamLib (Sawaya et al., Quantum 8, 1559, 2024) stores each Hamiltonian as a string dataset in OpenFermion
QubitOperator form. This script walks every dataset, keeps those whose path contains --match, parses them with
grouping.parse_openfermion (so a bad file fails here, not mid-run), drops those with more than --max-terms terms
(the coloring model grows as conflicts x K; see the post), and writes <safe dataset path>.txt. Check the
dataset's license and the file layout on the HamLib site before downloading [CHECK].
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouping as gp


def safe_name(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.strip("/"))[:120]


def keep(text: str, max_terms: int):
    terms = gp.parse_openfermion(text)
    return terms if len(terms) <= max_terms else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hdf5", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--match", default="")
    p.add_argument("--max-terms", type=int, default=1500)
    a = p.parse_args(argv)
    import h5py

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.hdf5:
        with h5py.File(f, "r") as h:
            names = []
            h.visit(lambda n: names.append(n) if isinstance(h[n], h5py.Dataset) else None)
            for n in names:
                if a.match not in n:
                    continue
                raw = h[n][()]
                text = raw.decode() if isinstance(raw, bytes) else str(raw)
                terms = keep(text, a.max_terms)
                if terms is None:
                    print(f"skip {n}: more than {a.max_terms} terms")
                    continue
                (out / f"{safe_name(n)}.txt").write_text(text)
                print(f"wrote {n}: {len(terms)} terms, {len(terms[0][0])} qubits")


if __name__ == "__main__":
    main()
