"""Turn the study's rows into the post's numbers: per-cell medians, the per-seed separation, the time-to-target
ladder, the first-answer times, and the cells the selection rule admits to the charts.

    python analyze.py $RUN/out/results.jsonl [$RUN_LADDER/out/results.jsonl] --out $RUN/analysis

Definitions (fixed before any data are seen):
- gap: (objective - reference) / |reference|, worse is positive, from the energy re-evaluated here. The 3D and 2D
  spin glasses use the published ground-state density (families.REFERENCES); the 3-regular graphs use the best
  cut found by any arm on that instance in these files ("best found, may be ours"), recomputed here.
- cell: (family, budget, setting). Its value is the median over seeds per instance, then the median over instances.
- best rival: the rival setting with the lowest cell value at that (family, budget).
- win: ours is lower than the best rival by more than the tie band (0.05 percentage points of gap), with 10 seeds.
  `separated` says whether every seed of ours beat every seed of that rival on every instance.
- time to target: for each instance, the target is our median gap at --target-budget (10 s); a rival setting reaches
  it at the smallest budget (main run plus ladder) where its median gap over seeds is at or below the target.
  The bracket (largest budget that missed, first budget that reached) is what is reported, never a point.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys as _sys                           # noqa: E402
from pathlib import Path as _Path             # noqa: E402
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import examplekit as bk

TIE_BAND = 0.0005


def load(paths):
    return bk.load_rows(paths)


def rescore_best_found(rows):
    """Gap of every solve row of a family without a published reference, against the best found per instance."""
    best = {}
    for r in rows:
        if r.get("kind") == "solve" and r.get("reference") is None and r.get("objective") is not None:
            k = r["instance"]
            best[k] = r["objective"] if k not in best else (max(best[k], r["objective"]) if r.get("family") == "regular"
                                                              else min(best[k], r["objective"]))
    for r in rows:
        if r.get("kind") == "solve" and r.get("reference") is None and r.get("objective") is not None:
            ref = best[r["instance"]]
            r["gap"] = (ref - r["objective"]) / abs(ref) if r.get("family") == "regular" else \
                (r["objective"] - ref) / abs(ref)
            r["reference_used"] = ref
    return best


def cells(rows):
    """{(family, budget, setting): {instance: [gap per seed]}} over solve rows with an answer."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("kind") == "solve" and r.get("gap") is not None:
            out[(r["family"], float(r["budget_s"]), r["setting"])][r["instance"]].append(r["gap"])
    return out


def cell_value(per_instance):
    return bk.median([bk.median(g) for g in per_instance.values()])


def table(rows):
    c = cells(rows)
    fams = sorted({k[0] for k in c})
    budgets = sorted({k[1] for k in c})
    out = []
    for fam in fams:
        for T in budgets:
            ours = c.get((fam, T, "api"))
            rivals = {k[2]: v for k, v in c.items() if k[0] == fam and k[1] == T and k[2] != "api"}
            if not ours or not rivals:
                continue
            vals = {s: cell_value(v) for s, v in rivals.items()}
            best = min(vals, key=vals.get)
            o = cell_value(ours)
            rv = rivals[best]
            sep = all(max(ours[i]) < min(rv[i]) for i in ours if i in rv)
            seeds = min(len(g) for g in ours.values())
            overran = sum(1 for r in rows if r.get("kind") == "solve" and r["family"] == fam and
                          float(r["budget_s"]) == T and r["setting"] == best and r.get("overran"))
            out.append({"family": fam, "budget_s": T, "ours": o, "best_rival": best, "rival": vals[best],
                        "ratio": vals[best] / o if o else None, "separated": sep, "seeds": seeds,
                        "rival_overran_runs": overran,
                        "win": bool(o < vals[best] - TIE_BAND and seeds >= 10),
                        "all_rivals": vals})
    return out


def ttt(rows, target_budget=10.0):
    """Per (family, rival setting): the bracket of budgets in which the rival reaches our target_budget gap."""
    c = cells(rows)
    out = []
    for (fam, T, setting), ours in c.items():
        if setting != "api" or T != target_budget:
            continue
        targets = {i: bk.median(g) for i, g in ours.items()}
        for rs in sorted({k[2] for k in c if k[0] == fam and k[2] != "api"}):
            budgets = sorted(k[1] for k in c if k[0] == fam and k[2] == rs)
            per_inst = []
            for inst, tgt in targets.items():
                lo, hi = None, None
                for b in budgets:
                    g = c[(fam, b, rs)].get(inst)
                    if not g:
                        continue
                    if bk.median(g) <= tgt:
                        hi = b
                        break
                    lo = b
                per_inst.append({"instance": inst, "target_gap": tgt, "missed_at": lo, "reached_at": hi})
            out.append({"family": fam, "setting": rs, "target_budget_s": target_budget, "instances": per_inst,
                        "factor_low": None if any(p["reached_at"] is None for p in per_inst) or not per_inst else
                        min((p["missed_at"] or 0) for p in per_inst) / target_budget,
                        "factor_high": None if any(p["reached_at"] is None for p in per_inst) or not per_inst else
                        max(p["reached_at"] for p in per_inst) / target_budget,
                        "never_reached": sum(p["reached_at"] is None for p in per_inst),
                        "largest_budget_tried": max(budgets) if budgets else None})
    return out


def first_answer(rows, budget=1.0):
    """At the smallest budget: our time to the first answer received vs each rival's clock, and no-answer counts."""
    by = defaultdict(lambda: {"times": [], "no_answer": 0, "runs": 0})
    for r in rows:
        if r.get("kind") not in ("solve", "error") or float(r.get("budget_s", -1)) != budget:
            continue
        k = (r["family"], r["setting"])
        by[k]["runs"] += 1
        if r.get("kind") == "error" or r.get("gap") is None:
            by[k]["no_answer"] += 1
            continue
        t = r.get("client_first_answer_s") if r["setting"] == "api" else r.get("clock_s")
        by[k]["times"].append(t)
    return [{"family": f, "setting": s, "median_s": bk.median(v["times"]), "max_s": max(v["times"], default=None),
             "no_answer": v["no_answer"], "runs": v["runs"]} for (f, s), v in sorted(by.items())]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--target-budget", type=float, default=10.0)
    a = p.parse_args(argv)
    rows = load(a.results)
    best = rescore_best_found(rows)
    rep = {"table": table(rows), "ttt": ttt(rows, a.target_budget), "first_answer": first_answer(rows),
           "best_found": best, "tie_band": TIE_BAND,
           "engine_versions": sorted({str(r.get("engine_version")) for r in rows if r.get("setting") == "api"
                                      and r.get("kind") == "solve"}),
           "threads": sorted({str(r.get("threads")) for r in rows if r.get("setting") == "api"
                              and r.get("kind") == "solve"})}
    rep["shown"] = [(t["family"], t["budget_s"]) for t in rep["table"] if t["win"]]
    rep["ran"] = [(t["family"], t["budget_s"]) for t in rep["table"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(rep, indent=1, default=str))
    for t in rep["table"]:
        print(f"{t['family']:8s} {t['budget_s']:5g}s ours {100 * t['ours']:.3f}%  {t['best_rival']} "
              f"{100 * t['rival']:.3f}%  x{t['ratio'] or 0:.2f}  separated={t['separated']}  "
              f"{'WIN' if t['win'] else 'not shown'}")
    print(f"shown {len(rep['shown'])} of {len(rep['ran'])} cells")
    return rep


if __name__ == "__main__":
    main()
