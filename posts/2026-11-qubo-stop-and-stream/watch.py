"""Submit a job, watch each better answer arrive, and keep the best on Ctrl-C.

    python watch.py --out runs/watch --n 100000 --time-limit 60 --seeds 1
    python watch.py --out runs/watch --n 100000 --time-limit 60 --live-plot      # a window that redraws

`fourcolors.qubo.submit` returns at once; `for best in job.incumbents()` yields each new best (status "running"),
then the final result, always last. Press Ctrl-C at any point: the client stops the job on the server and the loop
yields the best so far as a "stopped" result. You are billed for search time only (`billed_s`).
Client-side wall times printed and recorded here are untimed (they include network and queueing).
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


def watch(job, edges, out, common, *, clock=time.monotonic, on_point=None):
    """Iterate the job's incumbents; record each; return the final result."""
    t0 = clock()
    final = None
    for best in job.incumbents():
        value = None if best.solution is None else maxcut_model.cut(edges, best.solution)
        wall = clock() - t0
        if best.status == "running":
            out.add(kind="incumbent", wall_s_untimed=round(wall, 3), cut=value, solver_objective=best.objective,
                    job_id=job.id, **common)
            print(f"{wall:7.1f} s  cut {value}")
            if on_point is not None:
                on_point(wall, value)
        else:
            final = best
    return final


def main(argv=None):
    p = examplekit.base_parser(__doc__, time_limit=60.0, seeds=(1,))
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--instance-seed", type=int, default=5)
    p.add_argument("--live-plot", action="store_true", help="redraw a matplotlib window as answers arrive")
    a = p.parse_args(argv)
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = examplekit.client()
    edges, m = maxcut_model.build(a.n, seed=a.instance_seed)

    on_point = None
    if a.live_plot:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], marker="o")
        ax.set_xlabel("wall time (s)"); ax.set_ylabel("cut")
        xs, ys = [], []

        def on_point(t, v):
            xs.append(t); ys.append(v)
            line.set_data(xs, ys); ax.relim(); ax.autoscale_view(); plt.pause(0.01)

    for seed in a.seeds:
        common = {"arm": "watch", "n": a.n, "n_edges": len(edges), "seed": seed, "time_limit_s": a.time_limit,
                  "instance_seed": a.instance_seed}
        job = fourcolors.qubo.submit(m, time_limit_s=a.time_limit, seed=seed, client=client)
        print(f"job {job.id} submitted")
        final = watch(job, edges, out, common, on_point=on_point)
        x = final.solution
        out.add(kind="final", cut=None if x is None else maxcut_model.cut(edges, x),
                energy_python=None if x is None else m.energy(x), stopped=final.stopped, **common,
                **examplekit.result_fields(final))
        print(f"final: status {final.status}, cut {None if x is None else maxcut_model.cut(edges, x)}, "
              f"billed {final.billed_s} s of a {a.time_limit:g} s limit")


if __name__ == "__main__":
    main()
