"""Give it a second, and when longer pays: the runs behind the post (see the post).

    python run.py --out DIR --arm qiskit-long   [--circuits ids] [--budgets 1,3,10,60,300,600] [--workers 4 --cpus 4,5,6,7]
    python run.py --out DIR --arm qiskit-reseed [--circuits ids] [--workers 4 --cpus 4,5,6,7]
    python run.py ... --arm qiskit-long --qubits-initially-zero 1               # sensitivity E6 only
    python run.py ... --arm qiskit-long --workers 1 --threads 4 --cpus 4,5,6,7  # sensitivity E7 only
    python run.py --out DIR --arm 4colors --qiskit QDIR [--circuits ids] [--stream]

One results.jsonl per run, rows of these kinds:

* "circuit": id, family, qubits, input two-qubit gates, sha256 of the QASM and QPY files every arm reads;
* qiskit-long: one `qiskit_long.run` per (circuit, seed, T), N single-thread workers, the clock started after
  warm-up; a "result" row per selection key ("2q", "esp") with the scorer's numbers on the saved circuit, the
  run's wall, the winning member, the best single member in hindsight, and the measured approximation; plus
  "incumbent" rows (the best-so-far curve);
* qiskit-reseed: one run per seed at the largest T with only the `o3` member (plain O3, new seeds on the same N
  cores); a "result" row for every budget from the prefix of that run, and a "qiskit-default" row: the first O3
  trial, which is what most users run;
* 4colors: `compile(qc, target=from_qiskit(FakeFez()), time_budget_s=T_cell, seed=s, ...)` on the public API,
  with T_cell = min(T, Qiskit-long's measured wall for that cell and seed) and the approximation Qiskit's answer
  had (section 8.3), checked by the service (verify="wait": the row carries the verification label and
  distance) and scored by the same scorer; with --stream, one submit() per circuit at the largest budget,
  seed 1, every incumbent as an "incumbent" row.

The 4colors arm needs the Qiskit-long record first (--qiskit QDIR): it reads T_cell and the approximation from it.
Sensitivity rows (E6: qubits_initially_zero true; E7: threads > 1) never set T_cell or the approximation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import qiskit_long as ql  # noqa: E402  (first: it sets the one-thread environment before Qiskit loads)

import examplekit  # noqa: E402
import circuits as cl  # noqa: E402
import circuitkit as ck  # noqa: E402
import score as sc  # noqa: E402

POST = "compiler-longer-compiles"
BUDGETS = (1.0, 3.0, 10.0, 60.0, 300.0, 600.0)
#: seeds per budget (section 7): three seeds up to 60 s, two at 300 and 600 s.
SEEDS = {1.0: (1, 2, 3), 3.0: (1, 2, 3), 10.0: (1, 2, 3), 60.0: (1, 2, 3), 300.0: (1, 2), 600.0: (1, 2)}
PLAN_FLOOR_S = 1.0            # the client accepts time_budget_s from 1 to 600 s [OD-9: confirm for the standard plan]
EXACT_D = 1e-9                # a measured approximation at or below this is exact


def seeds_for(budget: float, override: Optional[Sequence[int]]) -> Sequence[int]:
    return tuple(override) if override else SEEDS.get(float(budget), (1,))


def run_dir(out: str, arm: str, cid: str, seed: int, budget: float) -> Path:
    return Path(out, arm, cid, f"s{seed}-T{budget:g}")


def read_jsonl(path) -> List[dict]:
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def best_single_member(trials: List[dict], key: str) -> Optional[dict]:
    """The best that any one member reached within T on its own (section 8.4, FAIRNESS A3): hindsight, so an
    upper bound on any static choice of one Qiskit configuration."""
    fn = ql.SELECT_KEYS[key]
    per: Dict[str, dict] = {}
    for r in trials:
        if r.get("ok") and r.get("counted"):
            if r["member"] not in per or fn(r) < fn(per[r["member"]]):
                per[r["member"]] = r
    return min(per.values(), key=fn) if per else None


def best_prefix(trials: List[dict], key: str, budget: float) -> Optional[dict]:
    """The best trial that ended by `budget` (a prefix of a budget-agnostic run)."""
    fn = ql.SELECT_KEYS[key]
    ok = [r for r in trials if r.get("ok") and r["t_end"] <= budget]
    return min(ok, key=lambda r: (fn(r), r["t_end"])) if ok else None


def load_qpy(path):
    from qiskit import qpy

    with open(path, "rb") as fh:
        return qpy.load(fh)[0]


def scored(qc_in, qc_out, target, lay=None) -> dict:
    out = sc.score(qc_out, target)
    out["approximation"] = sc.approximation(qc_in, qc_out, lay)
    out["approximation_measured"] = out["approximation"] is not None
    return out


# ---- the Qiskit arms -----------------------------------------------------------------------------------

def qiskit_long_rows(qc, circ, qpy_path, *, out, seed, budget, workers, cpus, members, target, backend,
                     arm="qiskit-long", run=ql.run, threads=1, qubits_initially_zero=False):
    d = run_dir(out, arm, circ["circuit_id"], seed, budget)
    rec = run({"qpy": str(qpy_path), "id": circ["circuit_id"], "backend": backend, "budget": budget,
               "workers": workers, "threads": threads, "cpus": cpus, "seed": seed, "members": members,
               "approximation_degree": 1.0, "qubits_initially_zero": qubits_initially_zero, "out": str(d)})
    trials = read_jsonl(d / "trials.jsonl")
    rows = []
    for key in ql.SELECT_KEYS:
        fin = rec["final"].get(key)
        base = dict(kind="result", post=POST, **circ, arm=arm, select=key, seed=seed, budget_s=budget,
                    wall_s=rec["wall_s"], workers=workers, threads=threads,
                    qubits_initially_zero=qubits_initially_zero, over_budget=fin is None, trials=rec["trials"],
                    trials_counted=rec["trials_counted"], failures=rec["failures"], members=rec["members"],
                    versions=rec["versions"], engine_version=ck.qiskit_version(), run_dir=str(d))
        if fin is None:
            rows.append(ck.row(**base, status="over_budget"))
            continue
        qc_out = load_qpy(d / f"best-{key}-{circ['circuit_id']}.qpy")
        hind = best_single_member(trials, key)
        rows.append(ck.row(**base, status="finished", member=fin["member"], k=fin["k"], qiskit_seed=fin["seed"],
                           t_found_s=fin["t_end"], **scored(qc, qc_out, target, fin.get("layouts")),
                           best_member=hind["member"] if hind else None,
                           best_member_2q=hind["two_qubit_gates"] if hind else None,
                           best_member_esp=hind["estimated_esp"] if hind else None))
        for t, member, k, s, g2, d2, dep, esp in rec["traj"][key]:
            rows.append(dict(kind="incumbent", post=POST, **circ, arm=arm, select=key, seed=seed, budget_s=budget,
                             t_s=t, member=member, k=k, qiskit_seed=s, two_qubit_gates=g2, two_qubit_depth=d2,
                             depth=dep, estimated_esp=esp))
    share: Dict[str, float] = {}
    for r in trials:                       # where the core-seconds went, per member (Figure 4)
        share[r["member"]] = share.get(r["member"], 0.0) + max(0.0, min(r["t_end"], budget) - r["t_start"])
    rows.append(dict(kind="time_share", post=POST, **circ, arm=arm, seed=seed, budget_s=budget, core_s=share))
    return rows


def qiskit_reseed_rows(qc, circ, qpy_path, *, out, seed, budgets, workers, cpus, target, backend, run=ql.run,
                       threads=1, qubits_initially_zero=False):
    """Plain O3 reseeded on the same N cores: one run at max(budgets), every shorter budget from its prefix."""
    top = max(budgets)
    d = run_dir(out, "qiskit-reseed", circ["circuit_id"], seed, top)
    rec = run({"qpy": str(qpy_path), "id": circ["circuit_id"], "backend": backend, "budget": top,
               "workers": workers, "threads": threads, "cpus": cpus, "seed": seed, "members": "o3",
               "approximation_degree": 1.0, "qubits_initially_zero": qubits_initially_zero, "out": str(d)})
    trials = read_jsonl(d / "trials.jsonl")
    rows = []
    first = min((r for r in trials if r.get("ok") and r["worker"] == 0), key=lambda r: r["t_start"], default=None)
    if first is not None:
        rows.append(ck.row(kind="result", post=POST, **circ, arm="qiskit-default", select="2q", seed=seed,
                           budget_s=None, wall_s=first["t_end"] - first["t_start"], status="finished",
                           engine_version=ck.qiskit_version(), qiskit_seed=first["seed"],
                           **{k: first[k] for k in ("two_qubit_gates", "two_qubit_depth", "depth", "estimated_esp")}))
    for b in budgets:
        for key in ql.SELECT_KEYS:
            pick = best_prefix(trials, key, b)
            base = dict(kind="result", post=POST, **circ, arm="qiskit-reseed", select=key, seed=seed, budget_s=b,
                        workers=workers, threads=threads, qubits_initially_zero=qubits_initially_zero,
                        engine_version=ck.qiskit_version(), run_dir=str(d),
                        trials_in_budget=sum(1 for r in trials if r.get("ok") and r["t_end"] <= b))
            if pick is None:
                rows.append(ck.row(**base, status="over_budget", over_budget=True))
                continue
            rows.append(ck.row(**base, status="finished", over_budget=False, wall_s=pick["t_end"],
                               qiskit_seed=pick["seed"], metrics_from="driver trial record (tests check it equals "
                               "the scorer)", **{k: pick[k] for k in ("two_qubit_gates", "two_qubit_depth", "depth",
                                                                     "estimated_esp")}))
    return rows


# ---- the 4colors arm -------------------------------------------------------------------------------------

def is_sensitivity(row: dict) -> bool:
    """E6 (qubits_initially_zero) and E7 (threads > 1) rows: reported apart, never the main Qiskit arm."""
    return bool(row.get("qubits_initially_zero")) or int(row.get("threads") or 1) > 1


def cell_settings(qrows: List[dict], cid: str, seed: int, budget: float) -> dict:
    """T_cell and the approximation option for one 4colors compile, from the Qiskit-long rows of that cell."""
    mine = [r for r in qrows if r.get("kind") == "result" and r.get("arm") == "qiskit-long"
            and r.get("circuit_id") == cid and r.get("seed") == seed and r.get("budget_s") == budget
            and not is_sensitivity(r)]
    if not mine:
        return {"skip": "no qiskit-long record for this cell"}
    wall = min(r["wall_s"] for r in mine)
    t_cell = min(float(budget), float(wall))
    floor = t_cell < PLAN_FLOOR_S
    ds = [r.get("approximation") for r in mine if r.get("status") == "finished"]
    measured = [d for d in ds if d is not None]
    if not ds:
        opts, d_cell = {"approximation_mode": "exact"}, None
    elif len(measured) < len(ds):        # too big to check as an operator: Qiskit ran at approximation_degree=1.0
        opts, d_cell = {"approximation_mode": "exact"}, None
    else:
        d_cell = min(measured)
        opts = {"approximation_mode": "exact"} if d_cell <= EXACT_D else {"max_approximation_error": d_cell}
    return {"t_cell": max(t_cell, PLAN_FLOOR_S), "budget_floor_applied": floor, "d_cell": d_cell,
            "d_measured": bool(ds) and len(measured) == len(ds), "options": opts,
            "qiskit_over_budget": all(r.get("over_budget") for r in mine)}


def fourcolors_rows(qc, circ, *, qrows, seed, budget, target, esp_target, client=None, verify="wait"):
    cs = cell_settings(qrows, circ["circuit_id"], seed, budget)
    if "skip" in cs:
        return [ck.row(kind="skipped", post=POST, **circ, arm="4colors", seed=seed, budget_s=budget,
                       status=cs["skip"])]
    res, f = ck.compile_4colors(qc, target=target, time_budget_s=cs["t_cell"], seed=seed, client=client,
                                verify=verify, keep_circuit=True, **cs["options"])
    out = f.pop("circuit", None)
    extra = scored(qc, out, esp_target) if out is not None else {}
    shown = {k: v for k, v in cs.items() if k != "options"}
    # select="2q": the compile's own objective (the default, two_qubit); the same answer is used on the ESP chart
    return [ck.row(kind="result", post=POST, **circ, arm="4colors", select="2q", coupler_set=ck.HEAVY_HEX_DEVICE, seed=seed,
                   budget_s=budget, options=cs["options"], **shown, **{**f, **extra})]


def fourcolors_stream(qc, circ, *, qrows, budget, target, client=None):
    cs = cell_settings(qrows, circ["circuit_id"], 1, budget)
    if "skip" in cs:
        return []
    return [{**r, "t_cell": cs["t_cell"], "d_cell": cs["d_cell"]}
            for r in ck.stream_rows(qc, post=POST, circuit=circ, target=target, budget_s=cs["t_cell"], seed=1,
                                    client=client, **cs["options"])]


# ---- main ------------------------------------------------------------------------------------------------

def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--arm", required=True, choices=("qiskit-long", "qiskit-reseed", "4colors"))
    p.add_argument("--circuits", default="all", help="comma list of ids (default: the post's twelve)")
    p.add_argument("--budgets", type=ck.floats, default=list(BUDGETS))
    p.add_argument("--seeds", type=ck.ints, default=None, help="override the per-budget seeds (section 7)")
    p.add_argument("--workers", type=int, default=4, help="Qiskit: N single-thread worker processes")
    p.add_argument("--cpus", default="", help="Qiskit: one cpu per worker thread, comma list (pinned)")
    p.add_argument("--threads", type=int, default=1,
                   help="Qiskit: Rayon threads per worker (default 1; --workers 1 --threads N is sensitivity E7)")
    p.add_argument("--qubits-initially-zero", type=int, default=0, choices=(0, 1),
                   help="Qiskit: 1 only for sensitivity E6 (main runs: 0, answers are checked as operators)")
    p.add_argument("--members", default="all", help="qiskit-long members (default: the frozen portfolio)")
    p.add_argument("--backend", default="FakeFez")
    p.add_argument("--qiskit", nargs="*", default=[], help="4colors arm: the Qiskit-long record(s) to match")
    p.add_argument("--stream", action="store_true", help="4colors arm: one incumbent stream per circuit")
    p.add_argument("--verify", default="wait", choices=("wait", "background", "skip"),
                   help="4colors arm: the service's check of each answer (default wait: the row carries its label)")
    a = p.parse_args(argv)
    if a.threads < 1:
        p.error("--threads must be at least 1")
    if a.arm == "4colors" and (a.threads != 1 or a.qubits_initially_zero):
        p.error("--threads and --qubits-initially-zero apply to the Qiskit arms only")
    cpus = [c for c in a.cpus.split(",") if c]
    if cpus and len(cpus) < a.workers * a.threads:
        p.error("--cpus needs one cpu per worker thread (workers x threads)")
    return a


def main(argv=None, *, run=ql.run, client=None):
    args = parse(argv)
    ids = list(cl.CIRCUITS) if args.circuits == "all" else [x for x in args.circuits.split(",") if x]
    res = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    res.add(**ck.meta_row(POST, args))
    files = {r["circuit_id"]: r for r in cl.export(os.path.join(args.out, "circuits"), ids)}
    cpus = [int(c) for c in args.cpus.split(",") if c] or None
    qiskit_target = ql.load_target(args.backend)
    if args.arm == "4colors":
        qrows = examplekit.load_rows(args.qiskit)
        target = ck.heavy_hex_target()
        client = client if client is not None else examplekit.client()
    for cid in ids:
        meta = files[cid]
        qc = load_qpy(os.path.join(args.out, "circuits", meta["qpy"]))
        circ = ck.circuit_fields(qc, cid, meta["family"])
        res.add(kind="circuit", post=POST, **meta)
        print(f"{cid}: {circ['qubits']} qubits, {circ['input_2q']} two-qubit gates in", flush=True)
        qpy_path = os.path.join(args.out, "circuits", meta["qpy"])
        if args.arm == "qiskit-reseed":
            for seed in args.seeds or (1, 2):
                for r in qiskit_reseed_rows(qc, circ, qpy_path, out=args.out, seed=seed, budgets=args.budgets,
                                            workers=args.workers, cpus=cpus, target=qiskit_target,
                                            backend=args.backend, run=run, threads=args.threads,
                                            qubits_initially_zero=bool(args.qubits_initially_zero)):
                    res.add(**r)
            continue
        for budget in args.budgets:
            for seed in seeds_for(budget, args.seeds):
                if args.arm == "qiskit-long":
                    rows = qiskit_long_rows(qc, circ, qpy_path, out=args.out, seed=seed, budget=budget,
                                            workers=args.workers, cpus=cpus, members=args.members,
                                            target=qiskit_target, backend=args.backend, run=run,
                                            threads=args.threads,
                                            qubits_initially_zero=bool(args.qubits_initially_zero))
                else:
                    rows = fourcolors_rows(qc, circ, qrows=qrows, seed=seed, budget=budget, target=target,
                                           esp_target=qiskit_target, client=client, verify=args.verify)
                for r in rows:
                    res.add(**r)
                    if r.get("kind") == "result":
                        print(f"  {r['arm']} {r.get('select') or ''} seed {seed} {budget:g}s: "
                              f"{r.get('two_qubit_gates')} {r.get('status') or ''}", flush=True)
        if args.arm == "4colors" and args.stream:
            for r in fourcolors_stream(qc, circ, qrows=qrows, budget=max(args.budgets), target=target, client=client):
                res.add(**r)
    print(f"wrote {res.path}")


if __name__ == "__main__":
    main()
