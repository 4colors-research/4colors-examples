"""Qiskit-long: Qiskit 2.6 given a time budget T and N cores, spent on stronger configurations, not only new seeds.

    python qiskit_long.py --qpy IN.qpy --id CID --backend FakeFez --budget 600 --workers 4 --cpus 4,5,6,7 \
        --seed 1 --out OUTDIR [--members all] [--approximation-degree 1.0] [--qubits-initially-zero 0] [--threads 1]

What it does (fixed before any run):

- N worker processes, one per core (pinned with --cpus), every thread pool at 1 (Rayon, OpenMP, BLAS, Qiskit's own
  process pool), so N cores means N cores.
- Every worker imports Qiskit, builds the Target and runs every member once on a tiny circuit before the clock
  starts (untimed warm-up, a courtesy to Qiskit). The clock starts when all are ready.
- Each worker walks the portfolio (MEMBERS below; the start is rotated by worker index so the N workers start on
  different members). A member's trial is one full transpile with its own seed. Members with an effort knob
  double their effort (more SABRE layout and swap trials, more forward-backward iterations, deeper lookahead,
  longer optimization loops, bigger VF2PostLayout search) whenever the doubled trial is predicted to fit into a
  third of the time left, so a long budget buys stronger configurations, not just more seeds.
- A trial is started only if its predicted time fits before the deadline (the job knows T up front and does not
  start work it cannot finish). Trials still running at T are killed and never counted.
- The answer is the best circuit among trials finished by T, by the pre-registered key: "2q" = (two-qubit gates,
  two-qubit depth, depth), or "esp" = (-estimated success probability, two-qubit gates, ...). Both incumbents and
  their best-so-far trajectories are kept. These metrics only steer Qiskit's own selection; reported numbers come
  from the one scorer, from the saved circuits.

Outputs in OUTDIR: run.json (one record: settings, versions, cpus, trajectories, finals), trials.jsonl (every
trial, including failures and skips), best-2q-<cid>.qpy and best-esp-<cid>.qpy (with layouts in run.json).

Qiskit 2.6.0 API facts this file relies on (read from the installed source, fixed before any run):
generate_preset_pass_manager(optimization_level, target=, approximation_degree=, seed_transpiler=,
layout_method=, routing_method=, translation_method=, hls_config=, initial_layout=, qubits_initially_zero=);
SabreLayout(coupling_map, seed, max_iterations, swap_trials, layout_trials, skip_routing);
SabreSwap(coupling_map, heuristic in {"basic", "lookahead", "decay"}, seed, trials);
LookaheadSwap(coupling_map, search_depth, search_width); VF2PostLayout(target, seed, call_limit, max_trials,
strict_direction); MinimumPoint(property_set_list, prefix, backtrack_depth); SwapStrategy.from_line(line);
Commuting2qGateRouter(swap_strategy, edge_coloring); FindCommutingPauliEvolutions(); HLSConfig(...,
plugin_selection="all", plugin_evaluation_fn=); high_level_synthesis_plugin_names(op).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

#: Every thread pool at one thread: N workers = N cores. Set before Qiskit is imported (workers are spawned, so
#: they import this module, and with it these settings, before they import Qiskit).
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "RAYON_NUM_THREADS": "1", "QISKIT_PARALLEL": "FALSE", "QISKIT_NUM_PROCS": "1"}
for _k, _v in THREAD_ENV.items():
    os.environ[_k] = _v
os.environ.pop("QISKIT_SABRE_ALL_THREADS", None)


def worker_env(threads: int = 1) -> Dict[str, str]:
    """The thread environment of one worker. threads = 1 (every main run): every pool at one thread. threads = N
    (sensitivity E7 only, FAIRNESS claim 6): one process with N Rayon threads for Qiskit's own parallel SABRE
    trials; every other pool stays at one."""
    if threads < 1:
        raise ValueError("threads must be at least 1")
    return {**THREAD_ENV, "RAYON_NUM_THREADS": str(threads)}


def worker_cpus(cpus: Optional[Sequence[int]], w: int, threads: int = 1) -> Optional[set]:
    """The cpus worker w is pinned to: `threads` consecutive entries of --cpus (one per thread), or None."""
    if not cpus:
        return None
    mine = list(cpus)[w * threads:(w + 1) * threads]
    if len(mine) < threads:
        raise ValueError(f"--cpus needs workers x threads = {(w + 1) * threads} cpus, got {len(cpus)}")
    return set(mine)

NON_GATES = frozenset({"barrier", "delay", "measure", "reset"})
FAILED_ERROR = 0.999            # circuitkit.FAILED_ERROR: an element with error >= this gives ESP 0
BASE_TRIALS = 20                # Qiskit O3's SABRE layout_trials / swap_trials (builtin_plugins._get_trial_count(20))
BASE_ITERS = 4                  # Qiskit O3's SabreLayout max_iterations
K_MAX = 64                      # effort cap: 1280 layout trials, 10 forward-backward iterations
ESCALATE_FRACTION = 1 / 3       # double effort only if the doubled trial fits in this share of the time left


# ---- metrics (selection only; the one scorer re-measures the saved circuits) --------------------------------

def is_two_qubit(inst) -> bool:
    return inst.operation.num_qubits == 2 and inst.operation.name not in NON_GATES


def metrics(qc, target=None) -> Dict[str, Optional[float]]:
    """circuitkit.circuit_metrics and circuitkit.estimated_esp, restated so the driver has no client import."""
    n2 = sum(1 for inst in qc.data if is_two_qubit(inst))
    out = {"two_qubit_gates": n2, "two_qubit_depth": qc.depth(is_two_qubit) if n2 else 0,
           "depth": qc.depth(lambda inst: inst.operation.name not in NON_GATES), "estimated_esp": None}
    if target is not None:
        out["estimated_esp"] = estimated_esp(qc, target)
    return out


def estimated_esp(qc, target) -> Optional[float]:
    logp, seen = 0.0, 0
    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(qc.find_bit(q).index for q in inst.qubits)
        try:
            props = target[name].get(qargs) if name in target.operation_names else None
        except (KeyError, TypeError):
            props = None
        if props is None or props.error is None:
            continue
        seen += 1
        e = min(max(float(props.error), 0.0), 1.0)
        if e >= FAILED_ERROR:
            return 0.0
        logp += math.log1p(-e)
    return math.exp(logp) if seen else None


SELECT_KEYS: Dict[str, Callable[[dict], tuple]] = {
    "2q": lambda m: (m["two_qubit_gates"], m["two_qubit_depth"], m["depth"]),
    "esp": lambda m: (-(m["estimated_esp"] or 0.0), m["two_qubit_gates"], m["two_qubit_depth"], m["depth"]),
}


# ---- targets ----------------------------------------------------------------------------------------------

def load_target(spec: str):
    """'FakeFez' (or any qiskit_ibm_runtime.fake_provider name): the backend's Target with its published
    calibration as-is (the device as Qiskit ships it, never a cut of ours).
    'generic:<n>:<line|ring|heavyhex>:<seed>' is a small GenericBackendV2 for offline tests only."""
    if spec.startswith("generic:"):
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit.transpiler import CouplingMap

        _, n, topo, seed = spec.split(":")
        n, seed = int(n), int(seed)
        cmap = {"line": CouplingMap.from_line, "ring": CouplingMap.from_ring}.get(topo)
        cmap = cmap(n) if cmap else CouplingMap.from_heavy_hex(n)
        return GenericBackendV2(cmap.size(), basis_gates=["cz", "rz", "sx", "x"], coupling_map=cmap,
                                seed=seed).target
    from qiskit_ibm_runtime import fake_provider

    return getattr(fake_provider, spec)().target


# ---- pass-manager surgery -----------------------------------------------------------------------------------

def map_tasks(task, fn, loop_hook=None):
    """Rebuild a flow-controller tree. Each pass p becomes fn(p) (None drops it); the task list of every
    DoWhileController is passed through loop_hook(tasks) -> tasks after its children are mapped."""
    from qiskit.passmanager.flow_controllers import ConditionalController, DoWhileController, FlowControllerLinear

    subs = getattr(task, "tasks", None)
    if subs is None:
        return fn(task)
    kept = [x for x in (map_tasks(t, fn, loop_hook) for t in subs) if x is not None]
    if isinstance(task, DoWhileController):
        if loop_hook is not None:
            kept = loop_hook(kept)
        return DoWhileController(kept, do_while=task.do_while)
    if isinstance(task, ConditionalController):
        return ConditionalController(kept, condition=task.condition)
    return FlowControllerLinear(kept)


def map_stage(stage, fn, loop_hook=None):
    from qiskit.transpiler import PassManager

    if stage is None:
        return None
    new = PassManager()
    new.append(map_tasks(stage.to_flow_controller(), fn, loop_hook))
    return new


def map_all_stages(pm, fn, loop_hook=None):
    for name in ("init", "layout", "routing", "translation", "optimization"):
        setattr(pm, name, map_stage(getattr(pm, name), fn, loop_hook))
    return pm


def log2i(k: int) -> int:
    return int(round(math.log2(max(1, k))))


@dataclass
class Ctx:
    """What every member needs: the Target, the approximation setting, the input's state assumption."""
    target: object
    approximation_degree: Optional[float] = 1.0
    qubits_initially_zero: bool = False

    def preset(self, seed: int, **kw):
        from qiskit.transpiler import generate_preset_pass_manager

        return generate_preset_pass_manager(optimization_level=3, target=self.target, seed_transpiler=seed,
                                            approximation_degree=self.approximation_degree,
                                            qubits_initially_zero=self.qubits_initially_zero, **kw)


def sabre_fn(ctx: Ctx, k: int, *, layout_skip_routing: Optional[bool] = None, heuristic: Optional[str] = None,
             dense: bool = False, vf2post_scale: Optional[int] = None):
    """A pass map for the preset O3 pipeline at effort k: SabreLayout with 20k layout and swap trials and
    4 + log2(k) forward-backward iterations (or DenseLayout in its place), routing-stage SabreSwap with 20k trials
    (and the given heuristic), VF2PostLayout with call_limit and max_trials times min(k, 8)."""
    from qiskit.transpiler.passes import DenseLayout, SabreLayout, SabreSwap, VF2PostLayout

    trials = BASE_TRIALS * k
    post = vf2post_scale if vf2post_scale is not None else min(k, 8)

    def fn(p):
        if isinstance(p, SabreLayout):
            if dense:
                return DenseLayout(target=ctx.target)
            skip = p.skip_routing if layout_skip_routing is None else layout_skip_routing
            return SabreLayout(ctx.target, seed=p.seed, max_iterations=BASE_ITERS + log2i(k), swap_trials=trials,
                               layout_trials=trials, skip_routing=skip)
        if isinstance(p, SabreSwap):
            return SabreSwap(ctx.target, heuristic=heuristic or p.heuristic, seed=p.seed, trials=trials)
        if isinstance(p, VF2PostLayout) and post > 1:
            return VF2PostLayout(target=ctx.target, seed=p.seed,
                                 call_limit=None if p.call_limit is None else p.call_limit * post,
                                 max_trials=None if not p.max_trials else p.max_trials * post,
                                 strict_direction=p.strict_direction)
        return p
    return fn


def augmented_loop(ctx: Ctx, k: int):
    """O3's optimization loop made stronger: CommutativeOptimization (2.6's merged commutative cancellation),
    matrix-based CommutativeInverseCancellation and Optimize1qGatesSimpleCommutation (run to completion) join the
    loop body, and the loop's MinimumPoint stop waits 5k non-improving rounds instead of 5."""
    from qiskit.transpiler.passes import (CommutativeInverseCancellation, CommutativeOptimization, MinimumPoint,
                                          Optimize1qGatesSimpleCommutation, TwoQubitPeepholeOptimization)

    ad = 1.0 if ctx.approximation_degree is None else ctx.approximation_degree

    def fn(p):
        if isinstance(p, MinimumPoint):
            prefix = p.backtrack_name[: -len("_minimum_point_state")]
            return MinimumPoint(p.property_set_list, prefix, backtrack_depth=5 * k)
        return p

    def hook(tasks):
        if not any(isinstance(t, TwoQubitPeepholeOptimization) for t in tasks):
            return tasks                 # not the optimization loop (the init stage has no DoWhile; be safe)
        extra = [CommutativeOptimization(approximation_degree=ad),
                 CommutativeInverseCancellation(matrix_based=True, max_qubits=4),
                 Optimize1qGatesSimpleCommutation(target=ctx.target, run_to_completion=True)]
        i = next(j for j, t in enumerate(tasks) if isinstance(t, TwoQubitPeepholeOptimization)) + 1
        return tasks[:i] + extra + tasks[i:]
    return fn, hook


def hls_all_config():
    """HLSConfig trying every installed synthesis plugin for every high-level object kind and keeping the one
    with the fewest two-qubit gates (plugin_selection="all"). Only matters when the input holds such objects."""
    from qiskit.transpiler.passes import HLSConfig, high_level_synthesis_plugin_names

    kinds = {}
    for op in ("mcx", "mcmt", "clifford", "linear_function", "permutation", "PauliEvolution", "qft",
               "ModularAdder", "HalfAdder", "FullAdder", "Multiplier", "IntComp", "WeightedSum"):
        names = high_level_synthesis_plugin_names(op)
        if names:
            kinds[op] = list(names)
    return HLSConfig(plugin_selection="all",
                     plugin_evaluation_fn=lambda qc: sum(1 for i in qc.data if is_two_qubit(i)), **kinds)


# ---- the swap-strategy member (Weidenfeller et al., Quantum 6, 870 (2022)) -----------------------------------

def lift_rzz_blocks(qc):
    """The same circuit with each maximal run of consecutive two-qubit ZZ rotations (gate name 'rzz', one
    numeric angle) as one PauliEvolutionGate of commuting ZZ terms (rzz(t) = exp(-i t/2 ZZ)), so Qiskit's
    swap-strategy router can see the commuting block. Returns (circuit, number of blocks with >= 2 terms)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    out.global_phase = qc.global_phase
    block: List[tuple] = []
    blocks = 0

    def flush():
        nonlocal block, blocks
        if len(block) >= 2:
            op = SparsePauliOp.from_sparse_list([("ZZ", [a, b], t / 2) for a, b, t in block], num_qubits=qc.num_qubits)
            out.append(PauliEvolutionGate(op, time=1.0), out.qubits)
            blocks += 1
        else:
            for a, b, t in block:
                out.rzz(t, a, b)
        block = []

    for inst in qc.data:
        op = inst.operation
        if op.name == "rzz" and op.num_qubits == 2 and len(op.params) == 1:
            try:
                theta = float(op.params[0])
            except TypeError:
                theta = None
            if theta is not None:
                a, b = (qc.find_bit(q).index for q in inst.qubits)
                block.append((a, b, theta))
                continue
        flush()
        out.append(op, [out.qubits[qc.find_bit(q).index] for q in inst.qubits],
                   [out.clbits[qc.find_bit(c).index] for c in inst.clbits])
    flush()
    return out, blocks


def choose_line(target, n: int, seed: int, call_limit=(30_000_000, 100_000)) -> Optional[List[int]]:
    """A path of n physical qubits picked by Qiskit's own VF2Layout (a subgraph isomorphism of the path graph,
    scored by the target's published error rates); seed -1 is Qiskit's default labelling, other seeds shuffle it."""
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler.passes import VF2Layout

    path = QuantumCircuit(n)
    for i in range(n - 1):
        path.cz(i, i + 1)
    vf = VF2Layout(target=target, seed=seed, call_limit=call_limit, max_trials=2_500)
    vf.run(circuit_to_dag(path))
    layout = vf.property_set["layout"]
    if layout is None:
        return None
    return [layout[path.qubits[i]] for i in range(n)]


def swapstrat_pm(ctx: Ctx, seed: int, line: List[int]):
    """Line swap strategy for the commuting ZZ blocks (Commuting2qGateRouter with an alternating edge coloring),
    then O3 on the chosen line with no further routing ("none": the strategy's swaps stay; ElidePermutations is
    only added when routing is not "none", so it cannot undo them)."""
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (Commuting2qGateRouter,
                                                                            FindCommutingPauliEvolutions, SwapStrategy)

    n = len(line)
    strat = SwapStrategy.from_line(list(range(n)))
    coloring = {e: (i + 1) % 2 for i in range(n - 1) for e in ((i, i + 1), (i + 1, i))}
    pm = ctx.preset(seed, initial_layout=line, routing_method="none")
    pm.pre_init = PassManager([FindCommutingPauliEvolutions(), Commuting2qGateRouter(strat, coloring)])
    return pm


# ---- the portfolio --------------------------------------------------------------------------------------------

@dataclass
class Member:
    name: str
    scales: bool                     # whether effort k changes the configuration
    cost_prior: float                # rough trial cost relative to plain O3 at k = 1 (scheduling only)
    build: Callable                  # build(ctx, k, seed, prep) -> object with .run(circuit)
    applies: Callable = lambda prep: True
    needs: str = ""                  # an optional package, if any


def _o3(ctx, k, seed, prep):
    return ctx.preset(seed)


def _sabre(ctx, k, seed, prep):
    return map_all_stages(ctx.preset(seed), sabre_fn(ctx, k))


def _sabre_lookahead(ctx, k, seed, prep):
    # routing_method "sabre" keeps the preset's own routing stage; SabreLayout then only picks the layout and the
    # routing-stage SabreSwap routes with the lookahead heuristic (no decay term)
    return map_all_stages(ctx.preset(seed, routing_method="sabre"),
                          sabre_fn(ctx, k, layout_skip_routing=True, heuristic="lookahead"))


def _dense(ctx, k, seed, prep):
    # VF2Layout first (a perfect layout if one exists), DenseLayout in SabreLayout's place, then SabreSwap and
    # VF2PostLayout from the default stages (an explicit layout_method="dense" would switch VF2PostLayout off)
    return map_all_stages(ctx.preset(seed), sabre_fn(ctx, k, dense=True))


def _lookahead_swap(ctx, k, seed, prep):
    from qiskit.transpiler.passes import LookaheadSwap

    base = sabre_fn(ctx, k)

    def fn(p):
        if isinstance(p, LookaheadSwap):
            return LookaheadSwap(ctx.target, search_depth=min(5 + log2i(k), 8), search_width=min(6 + log2i(k), 9))
        return base(p)
    return map_all_stages(ctx.preset(seed, routing_method="lookahead"), fn)


def _basic_swap(ctx, k, seed, prep):
    return map_all_stages(ctx.preset(seed, routing_method="basic"), sabre_fn(ctx, 1))


def _synth_translation(ctx, k, seed, prep):
    return map_all_stages(ctx.preset(seed, translation_method="synthesis"), sabre_fn(ctx, k))


def _kitchen_sink(ctx, k, seed, prep):
    base = sabre_fn(ctx, k)
    loop_fn, hook = augmented_loop(ctx, k)
    return map_all_stages(ctx.preset(seed, hls_config=hls_all_config()), lambda p: loop_fn(base(p)), hook)


def _swapstrat(ctx, k, seed, prep):
    line = choose_line(ctx.target, prep["qubits"], seed=-1 if seed % 4 == 0 else seed)
    if line is None:
        raise RuntimeError("no line of that length in the coupling graph")
    pm = swapstrat_pm(ctx, seed, line)

    class _Run:                       # the member compiles the lifted circuit (same operator) instead of the input
        def run(self, _qc):
            return pm.run(prep["lifted"])
    return _Run()


def _ai(ctx, k, seed, prep):
    from qiskit_ibm_transpiler import generate_ai_pass_manager

    cmap = ctx.target.build_coupling_map()
    return generate_ai_pass_manager(
        optimization_level=3, ai_optimization_level=3, coupling_map=cmap, ai_layout_mode="optimize",
        include_ai_synthesis=True,
        qiskit_transpile_options={"target": ctx.target, "seed_transpiler": seed,
                                  "approximation_degree": ctx.approximation_degree,
                                  "qubits_initially_zero": ctx.qubits_initially_zero})


MEMBERS: List[Member] = [
    Member("o3", False, 1.0, _o3),
    Member("o3-sabre-k", True, 1.2, _sabre),
    Member("o3-sabre-lookahead-k", True, 1.2, _sabre_lookahead),
    Member("o3-dense-k", True, 1.0, _dense),
    Member("o3-lookaheadswap-k", True, 8.0, _lookahead_swap),
    Member("o3-basicswap", False, 1.0, _basic_swap),
    Member("o3-synthtrans-k", True, 1.3, _synth_translation),
    Member("o3-kitchen-sink-k", True, 2.0, _kitchen_sink),
    Member("o3-swapstrat-line", False, 1.0, _swapstrat, applies=lambda prep: prep["blocks"] > 0),
    Member("o3-ai-local", False, 3.0, _ai, needs="qiskit_ibm_transpiler"),
]


def available_members(names: str) -> List[Member]:
    import importlib.util

    chosen = MEMBERS if names == "all" else [m for m in MEMBERS if m.name in names.split(",")]
    return [m for m in chosen if not m.needs or importlib.util.find_spec(m.needs) is not None]


def prepare(qc) -> dict:
    lifted, blocks = lift_rzz_blocks(qc)
    return {"qubits": qc.num_qubits, "lifted": lifted, "blocks": blocks}


# ---- the scheduler (pure, so it is testable without clocks) ----------------------------------------------------

@dataclass
class MemberState:
    k: int = 1
    last_dt: Optional[float] = None   # seconds of the last trial at effort k
    trials: int = 0
    dead: bool = False                # raised an error: not retried


@dataclass
class Scheduler:
    members: List[Member]
    rotate: int = 0
    states: Dict[str, MemberState] = field(default_factory=dict)
    ref_dt: Optional[float] = None    # first measured trial / its member's cost prior
    _i: int = 0

    def __post_init__(self):
        n = len(self.members)
        if n:
            r = self.rotate % n
            self.members = self.members[r:] + self.members[:r]
        self.states = {m.name: MemberState() for m in self.members}

    def predict(self, m: Member, k: int) -> Optional[float]:
        st = self.states[m.name]
        if st.last_dt is not None:
            return st.last_dt * (k / st.k if m.scales else 1.0)
        if self.ref_dt is not None:
            return self.ref_dt * m.cost_prior * (k if m.scales else 1)
        return None                   # nothing measured yet: the first trial always starts

    def next(self, remaining: float):
        """(member, k) for the next trial, or None if nothing is predicted to finish in `remaining` seconds."""
        n = len(self.members)
        for step in range(n):
            m = self.members[(self._i + step) % n]
            st = self.states[m.name]
            if st.dead:
                continue
            k = st.k
            if m.scales and st.last_dt is not None and k < K_MAX:
                p2 = self.predict(m, 2 * k)
                if p2 is not None and p2 <= remaining * ESCALATE_FRACTION:
                    k = 2 * k
            p = self.predict(m, k)
            if p is None or p <= remaining:
                self._i = (self._i + step + 1) % n
                return m, k
        return None

    def done(self, m: Member, k: int, dt: float, ok: bool):
        st = self.states[m.name]
        st.trials += 1
        if not ok:
            st.dead = True
            return
        st.k, st.last_dt = k, dt
        if self.ref_dt is None:
            self.ref_dt = dt / m.cost_prior


# ---- workers ---------------------------------------------------------------------------------------------------

def _qpy_bytes(qc) -> bytes:
    from qiskit import qpy

    buf = io.BytesIO()
    qpy.dump(qc, buf)
    return buf.getvalue()


def _layouts(qc):
    lay = qc.layout
    if lay is None:
        return None
    return {"initial": lay.initial_index_layout(filter_ancillas=True),
            "final": lay.final_index_layout(filter_ancillas=True)}


def _tiny(n: int):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.rzz(0.3, 0, n - 1)
    qc.rzz(0.2, 0, 1)
    return qc


def worker(w: int, args: dict, ready, start, outq):
    """One core (or `threads` cores, E7): warm up, wait for the common start, then run trials until nothing fits
    before the deadline."""
    threads = int(args.get("threads", 1))
    os.environ.update(worker_env(threads))      # before the first transpile builds Rayon's pool
    pin = worker_cpus(args.get("cpus"), w, threads)
    if pin:
        os.sched_setaffinity(0, pin)
    from qiskit import qpy

    with open(args["qpy"], "rb") as fh:
        qc = qpy.load(fh)[0]
    ctx = Ctx(load_target(args["backend"]), args["approximation_degree"], args["qubits_initially_zero"])
    prep = prepare(qc)
    members = [m for m in available_members(args["members"]) if m.applies(prep)]
    tiny = _tiny(min(4, qc.num_qubits))
    tprep = prepare(tiny)
    for m in members:                 # warm-up, untimed: every member's code path once on a tiny circuit
        try:
            m.build(ctx, 1, 0, tprep).run(tiny)
        except Exception:             # a member that fails here fails again later and is recorded then
            pass
    ready.put(w)
    t0 = start.get()
    deadline = t0 + args["budget"]
    sched = Scheduler(members, rotate=w)
    best: Dict[str, Optional[tuple]] = {key: None for key in SELECT_KEYS}
    trial = 0
    while True:
        now = time.monotonic()
        pick = sched.next(deadline - now)
        if pick is None:
            outq.put(("idle", w, now - t0))
            return
        m, k = pick
        seed = args["seed"] * 1_000_000 + w * 10_000 + trial
        trial += 1
        ts = time.monotonic()
        try:
            out = m.build(ctx, k, seed, prep).run(qc)
            ok, err = True, None
        except Exception as e:  # recorded, never hidden
            out, ok, err = None, False, f"{type(e).__name__}: {e}"[:300]
        te = time.monotonic()
        sched.done(m, k, te - ts, ok)
        rec = {"worker": w, "member": m.name, "k": k, "seed": seed, "t_start": ts - t0, "t_end": te - t0,
               "ok": ok, "error": err}
        if ok:
            rec.update(metrics(out, ctx.target))
            improved = []
            for key, fn in SELECT_KEYS.items():
                score = fn(rec)
                if best[key] is None or score < best[key]:
                    best[key] = score
                    improved.append(key)
            if improved and te <= deadline:
                outq.put(("circ", w, rec, improved, _qpy_bytes(out), _layouts(out)))
                continue
        outq.put(("trial", w, rec))


# ---- the parent ------------------------------------------------------------------------------------------------

def run(args: dict) -> dict:
    import multiprocessing as mp
    import queue

    import qiskit

    ctx_mp = mp.get_context("spawn")
    ready, outq = ctx_mp.Queue(), ctx_mp.Queue()
    starts = [ctx_mp.Queue() for _ in range(args["workers"])]
    procs = [ctx_mp.Process(target=worker, args=(w, args, ready, starts[w], outq), daemon=True)
             for w in range(args["workers"])]
    for p in procs:
        p.start()
    for _ in procs:
        ready.get(timeout=args.get("warmup_timeout", 1800))
    t0 = time.monotonic()
    for s in starts:
        s.put(t0)
    deadline = t0 + args["budget"]
    trials: List[dict] = []
    best: Dict[str, Optional[dict]] = {key: None for key in SELECT_KEYS}
    traj: Dict[str, List[list]] = {key: [] for key in SELECT_KEYS}
    idle = 0
    Path(args["out"]).mkdir(parents=True, exist_ok=True)

    def take(msg):
        nonlocal idle
        kind = msg[0]
        if kind == "idle":
            idle += 1
            return
        rec = msg[2]
        rec["counted"] = rec["t_end"] <= args["budget"]
        trials.append(rec)
        if kind != "circ" or not rec["counted"]:
            return
        _, _, _, improved, blob, layouts = msg
        for key in improved:
            fn = SELECT_KEYS[key]
            if best[key] is None or (fn(rec), rec["t_end"]) < (fn(best[key]), best[key]["t_end"]):
                best[key] = {**rec, "layouts": layouts}
                Path(args["out"], f"best-{key}-{args['id']}.qpy").write_bytes(blob)
        for key in SELECT_KEYS:       # the best-so-far curves, rebuilt from every counted trial in time order
            pts, cur = [], None
            for r in sorted((r for r in trials if r["ok"] and r["counted"]), key=lambda r: r["t_end"]):
                s = SELECT_KEYS[key](r)
                if cur is None or s < cur:
                    cur = s
                    pts.append([round(r["t_end"], 4), r["member"], r["k"], r["seed"], r["two_qubit_gates"],
                                r["two_qubit_depth"], r["depth"], r["estimated_esp"]])
            traj[key] = pts

    while time.monotonic() < deadline and idle < len(procs):
        try:
            take(outq.get(timeout=max(0.01, min(1.0, deadline - time.monotonic()))))
        except queue.Empty:
            pass
    for p in procs:                   # trials still running at T are killed and never counted
        p.terminate()
    grace = time.monotonic() + 2.0    # drain messages already sent; take() drops any with t_end > T
    while time.monotonic() < grace:
        try:
            take(outq.get(timeout=0.1))
        except queue.Empty:
            break
    for p in procs:
        p.join(timeout=5)
    wall = time.monotonic() - t0
    with open(Path(args["out"], "trials.jsonl"), "a") as fh:
        for r in trials:
            fh.write(json.dumps({"id": args["id"], **r}) + "\n")
    rec = {"arm": "qiskit-long", "id": args["id"], "budget_s": args["budget"], "workers": args["workers"],
           "threads": int(args.get("threads", 1)),
           "cpus": args.get("cpus"), "seed": args["seed"], "backend": args["backend"],
           "approximation_degree": args["approximation_degree"],
           "qubits_initially_zero": args["qubits_initially_zero"],
           "members": [m.name for m in available_members(args["members"])], "wall_s": round(wall, 3),
           "trials": len(trials), "trials_counted": sum(1 for r in trials if r["counted"] and r["ok"]),
           "failures": sum(1 for r in trials if not r["ok"]),
           "final": {k: v for k, v in best.items()}, "traj": traj,
           "versions": _versions(), "thread_env": worker_env(int(args.get("threads", 1))), "pre_start_warmup": True,
           "over_budget": best["2q"] is None}
    with open(Path(args["out"], "run.json"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _versions():
    import importlib.metadata as md

    out = {}
    for pkg in ("qiskit", "qiskit-ibm-runtime", "qiskit-ibm-transpiler", "rustworkx", "numpy", "scipy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--qpy")
    src.add_argument("--qasm", help="OpenQASM 3 (converted to QPY in OUTDIR before the workers start)")
    p.add_argument("--id", required=True)
    p.add_argument("--backend", default="FakeFez")
    p.add_argument("--budget", type=float, required=True, help="T in seconds of wall time")
    p.add_argument("--workers", type=int, default=4, help="N cores = N single-threaded worker processes")
    p.add_argument("--cpus", default="", help="comma list, one cpu per worker thread (pinned)")
    p.add_argument("--threads", type=int, default=1,
                   help="Rayon threads per worker (default 1; N with --workers 1 is sensitivity E7)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--members", default="all")
    p.add_argument("--approximation-degree", type=float, default=1.0)
    p.add_argument("--qubits-initially-zero", type=int, default=0, choices=(0, 1))
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cpus = [int(c) for c in a.cpus.split(",") if c != ""]
    if a.threads < 1:
        p.error("--threads must be at least 1")
    if cpus and len(cpus) < a.workers * a.threads:
        p.error("--cpus needs one cpu per worker thread (workers x threads)")
    qpy_path = a.qpy
    if a.qasm:
        import qiskit.qasm3 as q3
        from qiskit import qpy

        Path(a.out).mkdir(parents=True, exist_ok=True)
        qpy_path = str(Path(a.out, f"in-{a.id}.qpy"))
        with open(qpy_path, "wb") as fh:
            qpy.dump(q3.loads(Path(a.qasm).read_text()), fh)
    rec = run({"qpy": qpy_path, "id": a.id, "backend": a.backend, "budget": a.budget, "workers": a.workers,
               "threads": a.threads, "cpus": cpus or None, "seed": a.seed, "members": a.members,
               "approximation_degree": a.approximation_degree, "qubits_initially_zero": bool(a.qubits_initially_zero),
               "out": a.out})
    print(json.dumps({k: v for k, v in rec.items() if k != "traj"}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
