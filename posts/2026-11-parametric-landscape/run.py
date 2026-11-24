"""The QAOA landscape before and after noise: one compiled p = 1 circuit per compiler, bound on a (gamma, beta) grid.

For each 3-regular graph the parametric p = 1 QAOA circuit (with measurements) is compiled once by the 4colors
Quantum Compiler and once by Qiskit at optimization level 3 (matched budget), then bound at every grid point and
simulated. The 4colors arm gets the chip's working couplers as a description, no calibration; Qiskit gets the
device as published (FakeFez's full 176-coupler map with its calibration as-is: the rival gets the pure
calibration). The noise model below is uniform, so neither arm's score depends on the calibration:
  - ideal: the input circuit, noiseless (exact statevector);
  - noisy, per compiler: depolarizing error e2 on every CZ, e2/10 on SX and X, symmetric readout error e_ro,
    averaged over `--trajectories` noisy statevector trajectories per point.
A noiseless run of each compiled circuit at a few grid points must equal the ideal value (`check_ok`).

Halving the work: for a graph whose every vertex has odd degree (3-regular), the ideal p = 1 landscape is symmetric,
f(pi - gamma, beta) = f(gamma, beta): time reversal gives f(gamma, beta) = f(-gamma, -beta), and exp(-i pi C) = Z^n
(odd degrees) gives f(gamma + pi, beta) = f(gamma, -beta). On the grid gamma_i = pi i / k, row i > k/2 equals row
k - i, so only rows 0..k/2 are simulated and the rest are mirrored. For the noisy compiled circuits the symmetry is
not exact gate by gate (sx conjugates to sxdg), so on the first seed of each (graph, compiler) --mirror-checks
mirrored rows are also simulated directly (`mirror_check_max_abs`) and one row is repeated with another simulator
seed (`repeat_check_max_abs`, the trajectory sampling scatter it is compared with). --no-mirror simulates every row.

Usage:
    python run.py --out runs/p2 [--n 12,16,20] [--grid 32] [--budget 10] [--seeds 1] [--e2 3e-3]
                  [--trajectories 100] [--max-active 24] [--threads 1]
Rows go to OUT/results.jsonl: one per (graph, arm, condition), with the whole grid of expected-cut ratios.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import transpile

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
import examplekit as bk  # noqa: E402
from qaoakit import ansatz as az, apikit as ak, core  # noqa: E402


def grid_axes(k):
    """gamma in [0, pi), beta in [0, pi/2): one period of the p = 1 landscape of an unweighted graph."""
    return [math.pi * i / k for i in range(k)], [math.pi / 2 * j / k for j in range(k)]


def mirror_plan(k):
    """(rows to simulate, {row: the row it is read from}) for gamma_i = pi i / k: row i <= k/2 is its own source,
    row i > k/2 is read from row k - i (f(pi - gamma, beta) = f(gamma, beta))."""
    half = k // 2
    return list(range(half + 1)), {i: (i if i <= half else k - i) for i in range(k)}


def mirrored(rows_by_index, k):
    """The full grid from the simulated rows {i: row}, by mirror_plan."""
    _, src = mirror_plan(k)
    return [list(rows_by_index[src[i]]) for i in range(k)]


def check_rows(k, n_checks):
    """Mirrored rows to simulate directly as a check: spread over the mirrored half, never row k/2 (its own mirror)."""
    _, src = mirror_plan(k)
    cand = [i for i in range(k) if src[i] != i]
    if not cand or n_checks <= 0:
        return []
    step = max(1, len(cand) // n_checks)
    return cand[::step][:n_checks]


def one_qubit_pulses(qc):
    """sx and x gates: the single-qubit gates the noise model charges (rz is virtual and noiseless)."""
    ops = qc.count_ops()
    return int(ops.get("sx", 0) + ops.get("x", 0))


def ideal_grid(inst, gammas, betas):
    n, cost = inst["nodes"], core.cost_vector(inst)
    return [[float(np.abs(core.qaoa_statevector(n, cost, [g, b])) ** 2 @ cost) for b in betas] for g in gammas]


def noisy_grid(compiled, params, inst, gammas, betas, e2, e_ro, trajectories, seed, threads):
    """Expected cut at every grid point, from the compiled circuit bound there (Aer parameter binds, one run per
    gamma row). Returns (grid, active qubits)."""
    n, cost = inst["nodes"], core.cost_vector(inst)
    qc, meas = core.compact(compiled, n)               # idle qubits dropped; parameters kept
    qc.save_probabilities(qubits=meas)
    sim = core._sim(core.noise_model(e2) if e2 > 0 else None, threads)
    tq = transpile(qc, sim, optimization_level=0)
    g_par, b_par = params
    grid = []
    for g in gammas:
        res = sim.run(tq, parameter_binds=[{g_par: [g] * len(betas), b_par: list(betas)}],
                      shots=trajectories if e2 > 0 else 1, seed_simulator=seed).result()
        grid.append([float(core.readout(res.data(k)["probabilities"], n, e_ro) @ cost) for k in range(len(betas))])
    return grid, qc.num_qubits


def landscape(out, th, inst, gammas, betas, a, seed, mirror, checks):
    """The noisy grid (mirrored or full) and the check fields."""
    k = len(gammas)
    sim = lambda gs, sd: noisy_grid(out, th, inst, gs, betas, a.e2, a.e_ro, a.trajectories, sd, a.threads)
    if not mirror:
        grid, active = sim(gammas, seed)
        return grid, active, {"mirrored": False, "rows_simulated": k}
    rows, _ = mirror_plan(k)
    part, active = sim([gammas[i] for i in rows], seed)
    grid = mirrored(dict(zip(rows, part)), k)
    extra = {"mirrored": True, "rows_simulated": len(rows)}
    if checks:
        direct, _ = sim([gammas[i] for i in checks], seed)
        extra["mirror_check_rows"] = checks
        extra["mirror_check_max_abs"] = float(max(abs(x - y) for i, row in zip(checks, direct)
                                                  for x, y in zip(row, grid[i])))
        again, _ = sim([gammas[rows[len(rows) // 2]]], seed + 1000)
        extra["repeat_check_max_abs"] = float(max(abs(x - y) for x, y in zip(again[0], part[len(rows) // 2])))
        extra["rows_simulated"] += len(checks) + 1
    return grid, active, extra


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", default="12,16,20")
    ap.add_argument("--graph-seed", type=int, default=1)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--seeds", default="1", help="compiler seeds; one compile per compiler and seed")
    ap.add_argument("--e2", type=float, default=3e-3)
    ap.add_argument("--e-ro", type=float, default=0.01)
    ap.add_argument("--trajectories", type=int, default=100)
    ap.add_argument("--max-active", type=int, default=24)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-mirror", action="store_true", help="simulate every gamma row (no symmetry)")
    ap.add_argument("--mirror-checks", type=int, default=2,
                    help="mirrored rows simulated directly on the first seed of each (graph, compiler)")
    a = ap.parse_args(argv)
    res = bk.Results(Path(a.out) / "results.jsonl")
    client = bk.client()
    gammas, betas = grid_axes(a.grid)
    for n in [int(x) for x in a.n.split(",")]:
        inst = az.reg3_instance(n, a.graph_seed)
        cost = core.cost_vector(inst)
        opt = float(cost.max())                         # brute force over all 2^n cuts: a certificate
        base = dict(instance=inst["id"], n=n, edges=inst["edges"], variant="std", p=1, exact_optimum=opt,
                    optimum_certificate="brute-force", random_ratio=core.random_ratio(cost), gammas=gammas,
                    betas=betas, grid=a.grid)
        t = time.perf_counter()
        ideal = ideal_grid(inst, gammas, betas)
        _, src = mirror_plan(a.grid)
        sym = max(abs(ideal[i][j] - ideal[src[i]][j]) for i in range(a.grid) for j in range(a.grid)) / opt
        res.add(kind="landscape", arm="ideal", condition="ideal", ratio_grid=(np.array(ideal) / opt).tolist(),
                ideal_symmetry_max_abs=float(sym), sim_s=round(time.perf_counter() - t, 2), **base)
        qc, th = core.build(inst, "std-p1")
        qc = az.to_input(qc)
        qc.measure_all()                                # same logical circuit, parametric, for both compilers
        seeds = [int(s) for s in a.seeds.split(",")]
        for seed in seeds:
            for arm in ("4colors", "qiskit-o3"):
                if arm == "4colors":
                    out, rec = ak.compile_4colors(qc, budget_s=a.budget, seed=seed, client=client,
                                                  approximation_mode="exact")
                else:
                    out, rec = ak.compile_qiskit(qc, budget_s=a.budget, seed=seed)
                row = dict(base, **rec)
                if out is None:
                    res.add(kind="landscape", condition="refused", **row)
                    continue
                active = core.compact(out, n)[0].num_qubits
                if active > a.max_active:
                    res.add(kind="landscape", condition="not simulated", active_qubits=active, **row)
                    continue
                # noiseless check at three grid points: the compiled circuit must give the ideal value
                pick = [(1, 1), (a.grid // 3, a.grid // 2), (a.grid - 2, a.grid // 4)]
                chk, _ = noisy_grid(out, th, inst, [gammas[i] for i, _ in pick], [betas[j] for _, j in pick], 0.0,
                                    0.0, 1, 1, a.threads)
                ok = all(abs(chk[k][k] - ideal[i][j]) < 1e-6 * opt for k, (i, j) in enumerate(pick))
                t = time.perf_counter()
                checks = check_rows(a.grid, a.mirror_checks) if seed == seeds[0] else []
                grid, active, extra = landscape(out, th, inst, gammas, betas, a, seed, not a.no_mirror, checks)
                n1 = one_qubit_pulses(out)
                for key in ("mirror_check_max_abs", "repeat_check_max_abs"):     # in units of the ratio, like the grid
                    if key in extra:
                        extra[key] /= opt
                res.add(kind="landscape", condition="noisy", e2=a.e2, e1=a.e2 / 10, e_ro=a.e_ro,
                        trajectories=a.trajectories, check_ok=ok, active_qubits=active, one_qubit_pulses=n1,
                        ratio_grid=(np.array(grid) / opt).tolist(), sim_s=round(time.perf_counter() - t, 2),
                        **extra, **row)
                print(inst["id"], arm, seed, rec.get("two_qubit_gates"), "check_ok", ok,
                      "mirror check", extra.get("mirror_check_max_abs"), flush=True)


if __name__ == "__main__":
    main()
