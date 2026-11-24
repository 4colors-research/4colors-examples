"""API-side helpers shared by the parametric and QAOA blog-post examples (posts/*-parametric-*, posts/*-qaoa-*).

The heavy-hex device (a description for 4colors, FakeFez's published Target for Qiskit), both compilers at a
matched budget with one scoring rule, the bound-circuit equivalence check, the QUBO optimizer call. Nothing here
answers in place of either service: the tests use fixed canned results (each post's tests/canned.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

# ---- target, gate counts -------------------------------------------------------------------------
# Every post compiles for one IBM Heron-class heavy-hex device, Qiskit's FakeFez (156 qubits, 176 couplers, 7 of them
# published at error 1.0). The two arms get different views of it:
# * 4colors: the 169 working couplers (the 176 without the 7 at error 1.0; saved in qaoakit/targets.json under
#   TARGET_NAME) as a description dict (`device_target()`), no calibration. The service has no named targets
#   (targets are descriptions).
# * Qiskit: FakeFez's published Target as shipped (`qiskit_target()`: all 176 couplers, its calibration as-is,
#   the error-1.0 couplers included). Qiskit gets the published calibration as it is, never our cut.
# Both arms' circuits are scored by the same rules: the post's own uniform noise model, and `device_score` (gates
# on the 7 error-1.0 couplers, and the estimated success probability from the published calibration, where a gate
# on an error-1.0 coupler gives 0).

HERE = Path(__file__).resolve().parent             # qaoakit/ (targets.json sits here)
TARGET_NAME = "heavy_hex_156_cz"                   # the key in targets.json and the `target` label of 4colors rows
TARGET = TARGET_NAME                               # kept for callers that pass the label
DEVICE_BASIS = ["cz", "rz", "sx", "x"]
QISKIT_TARGET_LABEL = "fakefez"                    # the `target` label of Qiskit rows (circuitkit.HEAVY_HEX_DEVICE)
QISKIT_TARGET_VIEW = "published map and calibration as-is"


def _ck():
    """shared/circuitkit.py (FakeFez, its failed couplers, the ESP scorer), shared with the compiler posts."""
    import sys

    blog = str(HERE.parent)                        # shared/
    if blog not in sys.path:
        sys.path.append(blog)
    import circuitkit

    return circuitkit


def target_couplers(name: str = TARGET) -> List[List[int]]:
    """The working couplers in targets.json (the 4colors arm's map)."""
    return json.loads((HERE / "targets.json").read_text())[name]["couplers"]


def device_target(name: str = TARGET_NAME) -> dict:
    """The 4colors arm's map as the service's description (format_version 1: n_qubits, cz, the working couplers),
    sent as `target=` to `fourcolors.quantum.compiler.compile`. No calibration."""
    couplers = sorted(tuple(sorted(int(x) for x in e)) for e in target_couplers(name))
    return {"format_version": 1, "n_qubits": 1 + max(max(e) for e in couplers), "two_qubit_gate": "cz",
            "couplers": [list(e) for e in couplers]}


def qiskit_target():
    """Qiskit's target: FakeFez's published `Target` as shipped (all 176 couplers, the 7 at error 1.0 included,
    every error rate and duration as published). The object itself, not a copy (circuitkit.fez().target)."""
    return _ck().fez().target


def coupling_map():
    """The CouplingMap of Qiskit's target (FakeFez as published: 176 couplers, both directions)."""
    return qiskit_target().build_coupling_map()


def failed_couplers():
    """The couplers FakeFez publishes at error 1.0 (circuitkit.failed_couplers on the published target)."""
    return _ck().failed_couplers(qiskit_target())


def device_score(qc) -> dict:
    """One scoring rule for both arms' compiled circuits (physical qubits of FakeFez): gates on the couplers
    published at error 1.0, and the estimated success probability from the published calibration
    (circuitkit.estimated_esp: the product of 1 - error over every gate and measurement; a gate on an error-1.0
    coupler gives 0). An estimate from a snapshot, not a measured fidelity."""
    if qc is None:
        return {}
    ck = _ck()
    return dict(gates_on_failed_couplers=ck.circuit_metrics(qc, failed_couplers())["gates_on_failed_couplers"],
                estimated_esp=ck.estimated_esp(qc, qiskit_target()))


def _client_version():
    try:
        import fourcolors
    except ImportError:
        return None
    return getattr(fourcolors, "__version__", None)


def gate_stats(qc) -> dict:
    """Two-qubit gates, two-qubit depth and depth (barriers and measurements excluded)."""
    if qc is None:
        return {}
    two = lambda ins: len(ins.qubits) == 2 and ins.operation.name not in ("barrier", "measure")
    return dict(two_qubit_gates=sum(1 for ins in qc.data if two(ins)), two_qubit_depth=qc.depth(two),
                depth=qc.depth(lambda ins: ins.operation.name not in ("barrier", "measure")),
                n_parameters=len(qc.parameters))


# ---- the two compilers -----------------------------------------------------------------------------

def compile_4colors(circuit, *, budget_s: float, seed: int, target=None, client=None, verify=None,
                    approximation_mode="exact", wait_verification_s: Optional[float] = None):
    """One compile through `fourcolors.quantum.compiler.compile`. Returns (Qiskit circuit or None, record).

    `approximation_mode` is sent as given, "exact" unless the caller asks otherwise (never left to the service's
    default, so the record says what was asked: like-for-like with an exact Qiskit arm, rule D2).

    A refusal is recorded (its `code`), never dropped. The record has the job id and `billed_s`, the
    verification label and measured distance, and `res.parameters` (the names to bind).
    """
    from fourcolors.errors import FourColorsError, WaitTimeout
    from fourcolors.quantum import compiler

    if target is None or isinstance(target, str):  # a label from targets.json: send its description
        label, target = (target or TARGET_NAME), device_target(target or TARGET_NAME)
    else:
        label = "custom"
    rec = dict(arm="4colors", budget_s=budget_s, seed=seed, target=label,
               approximation_mode=approximation_mode)
    t = time.perf_counter()
    try:
        res = compiler.compile(circuit, target=target, time_budget_s=budget_s, seed=seed, verify=verify,
                               approximation_mode=approximation_mode, client=client)
    except FourColorsError as e:
        rec.update(refusal_code=getattr(e, "code", None), error=type(e).__name__, message=str(e)[:500],
                   wall_s=round(time.perf_counter() - t, 3))
        return None, rec
    if wait_verification_s and res.verification.label == "pending":
        try:
            res.wait_verification(timeout=wait_verification_s)
        except WaitTimeout:
            pass                                   # still pending: recorded as such
    qc = res.to_qiskit() if res.circuit is not None else None
    rec.update(gate_stats(qc), **(device_score(qc) if label != "custom" else {}), status=res.status, reason=res.reason, verification=res.verification.label,
               distance=res.verification.distance, parameters=res.parameters,
               job_id=res.job.id if res.job is not None else None, billed_s=res.billed_s,
               engine=res.engine, engine_version=(res.engine or {}).get("version"),
               client_version=_client_version(), wall_s=round(time.perf_counter() - t, 3))
    return qc, rec


def compile_qiskit(circuit, *, budget_s: float, seed: int, target=None, level: int = 3):
    """Qiskit at `optimization_level=level` on FakeFez's published target (`qiskit_target()`, unless `target` is
    given) with a matched budget: transpile with seeds seed*1000, seed*1000+1, ... until `budget_s` seconds of wall
    time are used (at least one try); keep the fewest two-qubit gates (then the lowest two-qubit depth). The record
    has `device_score` (the same rule as the 4colors arm's). Returns (circuit, record)."""
    import qiskit
    from qiskit import transpile

    target = qiskit_target() if target is None else target
    t0, best, tries, first = time.perf_counter(), None, 0, None
    while tries == 0 or time.perf_counter() - t0 < budget_s:
        c = transpile(circuit, target=target, optimization_level=level, seed_transpiler=seed * 1000 + tries)
        s = gate_stats(c)
        first = s["two_qubit_gates"] if first is None else first
        key = (s["two_qubit_gates"], s["two_qubit_depth"])
        if best is None or key < best[0]:
            best = (key, c, s, seed * 1000 + tries)
        tries += 1
    wall = round(time.perf_counter() - t0, 3)
    rec = dict(arm=f"qiskit-o{level}", budget_s=budget_s, seed=seed, target=QISKIT_TARGET_LABEL,
               target_view=QISKIT_TARGET_VIEW, qiskit_couplers=len(_ck().device_couplers(target)), tries=tries,
               best_seed_transpiler=best[3], first_try_two_qubit_gates=first, qiskit=qiskit.__version__,
               wall_s=wall, **best[2], **device_score(best[1]))
    return best[1], rec


# ---- bound compiled circuit == bound input (noiseless statevector) -------------------------------

def strip_final_ops(qc):
    """A copy without measurements and barriers (the state before readout)."""
    out = qc.copy_empty_like()
    for ins in qc.data:
        if ins.operation.name not in ("measure", "barrier"):
            out.append(ins.operation, ins.qubits, ins.clbits)
    return out


def reduce_active(bound, layout, n_logical):
    """A bound compiled circuit cut down to the qubits it uses: returns (circuit on the active qubits, position of
    each logical input qubit in it at the end). `layout` is the compiled circuit's (routing permutes qubits)."""
    from qiskit import QuantumCircuit

    bound = strip_final_ops(bound)
    final = layout.final_index_layout() if layout is not None else list(range(n_logical))
    active = sorted({bound.find_bit(q).index for ins in bound.data for q in ins.qubits} | set(final[:n_logical]))
    pos = {p: k for k, p in enumerate(active)}
    red = QuantumCircuit(len(active), global_phase=bound.global_phase)
    for ins in bound.data:
        red.append(ins.operation, [pos[bound.find_bit(q).index] for q in ins.qubits])
    return red, [pos[final[k]] for k in range(n_logical)]


def statevector(qc):
    """Aer statevector (single thread) of a circuit without measurements."""
    import numpy as np
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="statevector", max_parallel_threads=1)
    c = qc.copy()
    c.save_statevector()
    return np.asarray(sim.run(transpile(c, sim, optimization_level=0)).result().get_statevector())


def bound_states_match(input_qc, compiled_qc, values, compiled_values=None, *, max_qubits: int = 24,
                       tol: float = 1e-8):
    """Does `compiled_qc` bound with `compiled_values` (default `values`) prepare the state of `input_qc` bound
    with `values`? The compiled circuit's `layout` says where each input qubit ends (routing permutes them);
    every other qubit it touches must return to |0>. Returns dict(ok, fidelity, active_qubits), or ok=None when
    more than `max_qubits` qubits are active (not simulated)."""
    import numpy as np
    from qiskit import QuantumCircuit

    bound = compiled_qc.assign_parameters(values if compiled_values is None else compiled_values)
    red, where = reduce_active(bound, compiled_qc.layout, input_qc.num_qubits)
    if red.num_qubits > max_qubits:
        return dict(ok=None, fidelity=None, active_qubits=red.num_qubits)
    exp = QuantumCircuit(red.num_qubits)
    exp.compose(strip_final_ops(input_qc.assign_parameters(values)), where, inplace=True)
    fid = float(abs(np.vdot(statevector(red), statevector(exp))) ** 2)
    return dict(ok=bool(fid > 1 - tol), fidelity=fid, active_qubits=red.num_qubits)


# ---- the QUBO optimizer ----------------------------------------------------------------------------

def maxcut_matrix(n: int, edges) -> "np.ndarray":
    """Upper-triangular Q with x^T Q x = -cut(x); `edges` are (a, b, w) triples."""
    import numpy as np

    Q = np.zeros((n, n))
    for a, b, w in edges:
        i, j = min(a, b), max(a, b)
        Q[i, j] += 2 * w
        Q[i, i] -= w
        Q[j, j] -= w
    return Q


def solve_qubo(model, *, time_limit_s: float, seed: int, warm_start=None, client=None, **kw):
    """One `fourcolors.qubo.solve`. Returns (result or None, record): objective, time to best (the search time
    of the first trajectory point at the final objective), billed seconds, job id, refusal code."""
    import fourcolors.qubo
    from fourcolors.errors import FourColorsError

    rec = dict(arm="4colors-qubo", time_limit_s=time_limit_s, seed=seed, warm_start=warm_start is not None)
    t = time.perf_counter()
    try:
        r = fourcolors.qubo.solve(model, time_limit_s=time_limit_s, seed=seed, warm_start=warm_start,
                                  client=client, **kw)
    except FourColorsError as e:
        rec.update(refusal_code=getattr(e, "code", None), error=type(e).__name__, message=str(e)[:500],
                   wall_s=round(time.perf_counter() - t, 3))
        return None, rec
    ttb = next((p.get("t_search_s") for p in r.trajectory if p.get("objective") == r.objective), None)
    rec.update(objective=r.objective, status=r.status, reason=r.reason, time_to_best_s=ttb,
               trajectory=[[p.get("t_search_s"), p.get("objective")] for p in r.trajectory][:200],
               constraints=r.constraints, job_id=r.job.id if r.job is not None else None, billed_s=r.billed_s,
               wall_s=round(time.perf_counter() - t, 3))
    return r, rec


# ---- plots ----------------------------------------------------------------------------------------
# One color per arm everywhere (validated categorical slots 1-3 of the blog palette; text stays in ink).

ARM_COLORS = {"4colors": "#2a78d6", "qiskit-o3": "#eb6834", "4colors-qubo": "#1baf7a", "ideal": "#52514e"}
ARM_LABELS = {"4colors": "4colors Quantum Compiler", "qiskit-o3": "Qiskit O3",
              "4colors-qubo": "4colors Optimizer for QUBO", "ideal": "ideal (noiseless)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def axes_style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure(*a, **k):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(*a, **k)


def read_jsonl(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


# ---- max-3-cut as a QUBO -------------------------------------------------------------------------------

def max3cut_qubo(n, edges, penalty=None):
    """One-hot max-3-cut, variable 3v + c = "node v has color c". Returns (upper-triangular Q, offset) with
    E(x) = offset + x^T Q x = -cut(x) on one-hot assignments; with `penalty` P the one-hot rows are added as
    P * sum_v (sum_c x_vc - 1)^2, so E is minus the cut plus P per violation unit."""
    import numpy as np

    Q = np.zeros((3 * n, 3 * n))
    offset = -float(sum(w for _, _, w in edges))
    for a, b, w in edges:
        for c in range(3):
            i, j = sorted((3 * a + c, 3 * b + c))
            Q[i, j] += w
    if penalty:
        for v in range(n):
            for c in range(3):
                Q[3 * v + c, 3 * v + c] -= penalty
                for d in range(c + 1, 3):
                    Q[3 * v + c, 3 * v + d] += 2 * penalty
            offset += penalty
    return Q, offset


def max3cut_model(n, edges):
    """The same objective with exact one-hot rows (x_v0 + x_v1 + x_v2 = 1 for every node) instead of a penalty."""
    import fourcolors.qubo

    Q, offset = max3cut_qubo(n, edges)
    m = fourcolors.qubo.to_model(Q, offset=offset)
    for v in range(n):
        m.add_constraint({3 * v + c: 1 for c in range(3)}, lo=1, hi=1)
    return m


def max3cut_optimum(n, edges):
    """Exact maximum 3-cut by enumerating all 3^n colorings (node 0 fixed to color 0); fine to n = 13."""
    import itertools

    best = -1.0
    for rest in itertools.product(range(3), repeat=n - 1):
        col = (0,) + rest
        best = max(best, sum(w for a, b, w in edges if col[a] != col[b]))
    return float(best)
