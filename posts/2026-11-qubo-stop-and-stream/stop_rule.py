"""Stop paying for idle seconds: stop a job once it stops improving.

    python stop_rule.py --out runs/stop --n 100000 --time-limit 60 --patience 5 --seeds 1 2

Every --poll seconds the script peeks at the best answer so far with `job.best()`. If the cut has not improved by
more than --min-gain (relative) for --patience seconds, it calls `job.stop()`, which keeps the best answer. Ctrl-C
does the same. The final row records `billed_s`, the search seconds the job is billed for, beside its time limit.
The patience clock is the client's wall clock (untimed); `billed_s` and the trajectory are the service's.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors.qubo

import examplekit
import maxcut_model


def run_with_patience(job, *, patience: float, min_gain: float, poll: float, clock=time.monotonic,
                      sleep=time.sleep, on_poll=None):
    """(final result, why it ended): "no_progress", "interrupted" or "ended" (the job ended by itself).

    Works in energy terms (lower is better), so it fits any minimization model."""
    best_energy, last_gain_at = None, clock()
    try:
        while True:
            b = job.best()
            now = clock()
            if b is not None and b.status != "running":           # the job has ended: this is its final result
                return b, "ended"
            if b is not None and b.objective is not None:
                if best_energy is None or b.objective < best_energy - min_gain * abs(best_energy):
                    best_energy, last_gain_at = b.objective, now
                if on_poll is not None:
                    on_poll(now, b)
            if best_energy is not None and now - last_gain_at >= patience:
                return job.stop(), "no_progress"
            sleep(poll)
    except KeyboardInterrupt:
        print(f"\nstopping job {job.id}; the best so far is kept", file=sys.stderr)
        return job.stop(), "interrupted"


def main(argv=None):
    p = examplekit.base_parser(__doc__, time_limit=60.0, seeds=(1, 2))
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--instance-seed", type=int, default=5)
    p.add_argument("--patience", type=float, default=5.0, help="seconds without a gain before stopping")
    p.add_argument("--min-gain", type=float, default=0.0, help="relative gain that counts as progress")
    p.add_argument("--poll", type=float, default=1.0)
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()
    edges, m = maxcut_model.build(a.n, seed=a.instance_seed)
    t0 = time.monotonic()
    for seed in a.seeds:
        common = {"arm": f"patience_{a.patience:g}s", "n": a.n, "n_edges": len(edges), "seed": seed,
                  "time_limit_s": a.time_limit, "patience_s": a.patience, "min_gain": a.min_gain,
                  "instance_seed": a.instance_seed}
        job = fourcolors.qubo.submit(m, time_limit_s=a.time_limit, seed=seed, client=client)
        print(f"job {job.id} submitted")

        def on_poll(now, b):
            out.add(kind="poll", wall_s_untimed=round(now - t0, 3), solver_objective=b.objective, job_id=job.id,
                    **common)

        final, why = run_with_patience(job, patience=a.patience, min_gain=a.min_gain, poll=a.poll, on_poll=on_poll)
        x = final.solution
        value = None if x is None else maxcut_model.cut(edges, x)
        out.add(kind="final", why=why, cut=value, energy_python=None if x is None else m.energy(x),
                stopped=final.stopped, **common, **examplekit.result_fields(final))
        print(f"{why}: status {final.status}, cut {value}, billed {final.billed_s} s of a {a.time_limit:g} s limit")


if __name__ == "__main__":
    main()
