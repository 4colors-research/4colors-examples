"""MIMO detection through the 4colors Optimizer for QUBO, with MMSE, zero-forcing and a sphere decoder beside it.

    python run.py --out runs/try --nt 8 --mod qpsk --snr 10 --frames 2 --time-limit 1        # a first try
    python run.py --out runs/q32 --nt 32 --mod qpsk --snr 4 8 12 --frames 10 --time-limit 1 --seeds 1 2 3
    python run.py --out runs/b64 --nt 64 --mod qpsk --snr 8 --frames 200 --pack 50 --time-limit 1 10

Each channel use (a "frame") is a seeded i.i.d. Rayleigh channel. For every frame the script writes one row per
arm to <out>/results.jsonl:

  arm=mmse, arm=zf   linear detectors sliced to the constellation (local, cheap)
  arm=sphere         Schnorr-Euchner sphere decoder: exact ML when `complete`, with its node count
  arm=api            the 4colors answer, one job per frame (or one job per --pack frames, block-diagonal)

Every answer is scored by the same function (`mimo.score`): bit errors on the Gray-coded bits, symbol errors and
the residual ||y - Hx||^2. API rows also carry `is_ml` (residual equal to the sphere decoder's, when it finished)
and `t_to_ml_s` (search seconds until the service's best reached the ML residual, from its trajectory).
Local `wall_s` values are only quotable from a timed record (`--timed` marks the rows; the record says timed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))       # examplekit
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fourcolors.qubo

import examplekit
import mimo


def _tol(x: float) -> float:
    return 1e-6 * max(1.0, abs(x))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="directory for results.jsonl")
    p.add_argument("--seeds", type=int, nargs="+", default=[1], help="solver seeds (default %(default)s)")
    p.add_argument("--time-limit", dest="time_limits", type=float, nargs="+", default=[1.0],
                   help="seconds per job; several values give one job per value (default 1)")
    p.add_argument("--nt", type=int, default=16, help="users (transmit streams)")
    p.add_argument("--nr", type=int, default=None, help="receive antennas (default: = nt, a square system)")
    p.add_argument("--mod", default="qpsk", choices=sorted(mimo.MODS))
    p.add_argument("--snr", type=float, nargs="+", default=[8.0], help="SNR values in dB")
    p.add_argument("--frames", type=int, default=10, help="channel uses per SNR")
    p.add_argument("--frame-seed", type=int, default=1, help="seed of the first frame")
    p.add_argument("--pack", type=int, default=0, help="frames per job, block-diagonal (0: one job per frame)")
    p.add_argument("--sphere-max-nodes", type=int, default=2_000_000)
    p.add_argument("--no-sphere", action="store_true", help="skip the sphere decoder (large Nt, low SNR)")
    p.add_argument("--no-api", action="store_true", help="baselines only (for the timed local record)")
    p.add_argument("--timed", action="store_true", help="mark local wall times as from a timed record")
    a = p.parse_args(argv)
    nr = a.nr or a.nt
    if nr < a.nt:
        raise SystemExit("this post uses Nr >= Nt (the sphere decoder's QR form needs it)")
    out = examplekit.Results(Path(a.out) / "results.jsonl")
    client = None if a.no_api else examplekit.client()

    for snr in a.snr:
        insts = [mimo.generate(a.nt, nr, a.mod, snr, a.frame_seed + f) for f in range(a.frames)]
        ml = {}
        for f, inst in enumerate(insts):
            common = _common(inst, f)
            for arm, fn in (("mmse", mimo.mmse), ("zf", mimo.zero_forcing)):
                s = fn(inst)
                out.add(arm=arm, **common, **mimo.score(inst, s), bits_hat=mimo.bits_str(s, inst.k))
            if not a.no_sphere:
                sd = mimo.sphere_decode(inst, max_nodes=a.sphere_max_nodes)
                ml[f] = mimo.residual(inst, sd["spins"]) if sd["complete"] else None   # the one scorer
                out.add(arm="sphere", **common, **mimo.score(inst, sd["spins"]), bits_hat=mimo.bits_str(sd["spins"], inst.k),
                        nodes=sd["nodes"], complete=sd["complete"], max_nodes=a.sphere_max_nodes,
                        wall_s=sd["wall_s"], timed=bool(a.timed))
        if a.no_api:
            continue
        groups = [list(range(len(insts)))[i:i + a.pack] for i in range(0, len(insts), a.pack)] if a.pack \
            else [[f] for f in range(len(insts))]
        for T in a.time_limits:
            for seed in a.seeds:
                for g in groups:
                    _solve_group(out, client, [insts[f] for f in g], g, ml, T, seed, packed=bool(a.pack))
    return 0


def _common(inst: mimo.Instance, frame: int) -> dict:
    return {"inst": inst.name, "nt": inst.nt, "nr": inst.nr, "mod": inst.mod, "snr_db": inst.snr_db,
            "frame": frame, "frame_seed": inst.seed, "n_spins": inst.n_spins}


def _solve_group(out, client, insts, frames, ml, T, seed, *, packed: bool):
    if packed:
        m, offsets = mimo.packed_model(insts)
    else:
        m, offsets = mimo.model(insts[0]), [0, insts[0].n_spins]
    consts = [mimo.ising_terms(inst)[2] for inst in insts]
    r = fourcolors.qubo.solve(m, time_limit_s=T, seed=seed, client=client)
    fields = examplekit.result_fields(r)
    refs = [ml.get(f) for f in frames]
    all_ref = all(v is not None for v in refs)
    # time to ML quality, from the service's trajectory (model units: the residual minus the constants)
    target = sum(refs) - sum(consts) if all_ref else None
    t_ml = None if target is None else examplekit.time_to_target(fields["trajectory"], target + _tol(target))
    sol = r.solution
    for b, (inst, f) in enumerate(zip(insts, frames)):
        row = {"arm": "api", **_common(inst, f), "time_limit_s": T, "seed": seed,
               "pack": len(insts) if packed else 0, "block": b}
        if sol is None:
            out.add(**row, **fields, answered=False)
            continue
        s = sol[offsets[b]:offsets[b + 1]]
        sc = mimo.score(inst, s)
        ref = refs[b]
        out.add(**row, **fields, answered=True, **sc, bits_hat=mimo.bits_str(s, inst.k),
                energy_python=float(m.energy(list(sol))) if b == 0 else None,
                ml_residual=ref, is_ml=None if ref is None else sc["residual"] <= ref + _tol(ref),
                t_to_ml_s=t_ml, t_to_ml_scope="job" if packed else "frame")
    return r


if __name__ == "__main__":
    sys.exit(main())
