"""Select k features with the mRMR QUBO and an exact "k features" row; score against plain-Python references.

    python run.py --out runs/fs --dataset german:data/german.data-numeric \
        --dataset csv:taiwan-bankruptcy:"Bankrupt?":data/data.csv --k 5 8 10 --alpha 0.5 0.7 0.9 --seeds 1 2 3 --time-limit 1
    python run.py --out runs/smoke --dataset synthetic:2000:20:1 --k 5 --alpha 0.7 --seeds 1

Datasets: `german:<path>` (UCI Statlog German Credit, german.data-numeric; named german-credit),
`csv:<name>:<target column>:<path>`,
`synthetic:<rows>:<d>:<seed>`. Files are read from disk; this script downloads nothing.

Rows: `instance` (sizes, file sha256), `reference` (exact enumeration when d <= --exact-max, greedy on the QUBO
objective, Peng's greedy mRMR; each scored on the same objective), `solve` (our answer, objective recomputed,
rows checked). With --auc, every answer also gets a cross-validated logistic-regression AUC (context only).
"""
from __future__ import annotations

import hashlib
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors
import fourcolors.qubo

import examplekit
import features as fs


def load(spec: str):
    kind, _, rest = spec.partition(":")
    if kind == "german":
        X, y, names = fs.load_german_numeric(rest)
        return "german-credit", X, y, names, sha256(rest)
    if kind == "csv":
        name, target, path = rest.split(":", 2)
        X, y, names = fs.load_csv(path, target)
        return name, X, y, names, sha256(path)
    if kind == "synthetic":
        rows, d, seed = (int(v) for v in rest.split(":"))
        X, y, names = fs.synthetic(rows, d, seed)
        return f"synthetic-{rows}x{d}-s{seed}", X, y, names, None
    raise SystemExit(f"unknown dataset spec {spec!r}")


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p = examplekit.base_parser(__doc__, time_limit=1.0)
    p.add_argument("--dataset", action="append", required=True)
    p.add_argument("--k", type=int, nargs="+", default=[5, 8, 10])
    p.add_argument("--alpha", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--exact-max", type=int, default=3_000_000, help="enumerate when C(d, k) <= this")
    p.add_argument("--auc", action="store_true", help="add a 5-fold logistic-regression AUC per answer (context)")
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()

    for spec in a.dataset:
        name, X, y, names, digest = load(spec)
        rel, red = fs.relevance_redundancy(X, y, a.bins)
        d = len(rel)
        out.add(kind="instance", dataset=name, n_rows=int(len(y)), d=d, positives=int(y.sum()), sha256=digest,
                bins=a.bins, feature_names=names)
        aucs = {}

        def context(x):
            if not a.auc:
                return {}
            key = tuple(fs.chosen(x))
            if key not in aucs:
                aucs[key] = fs.cv_auc(X, y, key)
            return {"cv_auc": aucs[key]}

        for k in a.k:
            for alpha in a.alpha:
                common = dict(dataset=name, d=d, k=k, alpha=alpha)
                refs = {"greedy_qubo": fs.greedy_qubo(rel, red, alpha, k), "greedy_mrmr": fs.greedy_mrmr(rel, red, k)}
                if comb(d, k) <= a.exact_max:
                    refs["exact"] = fs.exact_best(rel, red, alpha, k)[1]
                for method, x in refs.items():
                    out.add(kind="reference", method=method, **common, energy_python=fs.energy(rel, red, alpha, x),
                            features=fs.chosen(x), **context(x))
                m = fs.model(rel, red, alpha, k)
                for seed in a.seeds:
                    try:
                        r = fourcolors.qubo.solve(m, time_limit_s=a.time_limit, seed=seed, client=client)
                    except fourcolors.errors.FourColorsError as exc:
                        out.add(kind="solve", **common, seed=seed, time_limit_s=a.time_limit,
                                error_code=getattr(exc, "code", None), error=str(exc))
                        continue
                    x = r.solution
                    ok = x is not None and bool(m.feasible(x))
                    row = out.add(kind="solve", **common, seed=seed, time_limit_s=a.time_limit,
                                  energy_python=None if x is None else fs.energy(rel, red, alpha, x),
                                  model_feasible=None if x is None else ok,
                                  service_feasible=(r.constraints or {}).get("feasible"),
                                  features=None if x is None else fs.chosen(x),
                                  **(context(x) if ok else {}), **examplekit.result_fields(r))
                    print(f"{name} k={k} alpha={alpha} seed={seed} E={row['energy_python']} feasible={ok}")


if __name__ == "__main__":
    main()
