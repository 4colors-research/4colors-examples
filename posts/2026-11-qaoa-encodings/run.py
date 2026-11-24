"""One problem, three encodings, two machines: max-3-cut as QAOA circuits and as QUBO models.

QAOA side (p = 1, 2; angles optimized noiselessly; compiled by Qiskit O3 and by 4colors; scored under noise):
  onehot-penalty  3 qubits per node (x_vc = node v has color c), the one-hot rows as a penalty P in the phase
                  layer, standard X mixer: any bit string can come out, and infeasible ones score 0
  onehot-xy       3 qubits per node, W-state start and an XY mixer that keeps every node one-hot by construction
  binary          2 qubits per node, color = min(b0 + 2 b1, 2) (the code 11 also means color 2): every bit string is a
                  coloring, but the cost has 3- and 4-qubit terms
Optimizer side (`fourcolors.qubo.solve` at short budgets): the penalty QUBO, and the one-hot model with exact rows
(`add_constraint`). The binary encoding is not quadratic (it has 4-body terms), so it is not a QUBO without extra
variables: recorded as such.

Scoring (a choice, stated in the post): quality is the max-3-cut of the decoded coloring over the optimum; an
infeasible one-hot string scores 0. The angles of every encoding are tuned on that score (noiseless expectation),
not on the penalty Hamiltonian; the penalty only shapes the phase layer.

The binary code is biased: color 2 has two codes (10 and 11), so the uniform start state gives every node color 2
with probability 1/2 and colors 0 and 1 with 1/4 each. Each `qaoa` row records the ratio of that encoding's own
start state (`start_ratio`: the expected score before any QAOA layer) beside the ratio after it, so each encoding
is measured against its own baseline, not against a uniform random coloring.

Larger graphs (`--count-graphs`, random 3-regular, max-3-cut): compiled by 4colors for gate counts only (`count`
rows; no angles, no simulation: a 72-qubit state does not fit in memory). The optimizer also gets a larger
3-regular graph (`--optimizer-graphs`), whose optimum is every edge cut: a connected graph of maximum degree 3 that
is not K4 can be colored with 3 colors (Brooks' theorem), so the certificate needs no enumeration.

Usage:
    python run.py --out runs/enc [--only m3c] [--encodings onehot-penalty,onehot-xy,binary] [--ps 1,2]
                  [--budget 5] [--seeds 1,2,3,4,5] [--e2 3e-3] [--qubo-limits 0.01,1]
                  [--count-graphs 12,24] [--count-seeds 1,2,3] [--optimizer-graphs 200] [--arms 4colors,qiskit-o3]
`--arms 4colors` and `--arms qiskit-o3` split the run into an API record and a timed local record (the angles are
tuned with a fixed seed, so both records use the same angles).
"""
import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]
import examplekit as bk  # noqa: E402
from qaoakit import ansatz as az, apikit as ak, core  # noqa: E402

ENCODINGS = ["onehot-penalty", "onehot-xy", "binary"]
BINARY_COLOR_PROBS = (0.25, 0.25, 0.5)      # colors 0, 1, 2 of one node under uniform bits (codes 00, 01, 10 and 11)
QUBITS_PER_NODE = {"onehot-penalty": 3, "onehot-xy": 3, "binary": 2}


# ---- instances ---------------------------------------------------------------------------------------------

def reg3_max3cut(n, seed=1):
    """A random 3-regular graph as a unit-weight max-3-cut instance."""
    g = nx.random_regular_graph(3, n, seed=seed)
    return dict(id=f"m3c-reg3-n{n}-s{seed}", kind="max3cut", nodes=n,
                edges=[(int(a), int(b), 1.0) for a, b in sorted(g.edges)])


def brooks_optimum(inst):
    """The maximum 3-cut of a unit-weight graph that is connected, has maximum degree at most 3 and is not K4: every
    edge (Brooks' theorem: such a graph has a proper 3-coloring). None when the theorem does not apply."""
    g = nx.Graph()
    g.add_nodes_from(range(inst["nodes"]))
    g.add_edges_from((a, b) for a, b, _ in inst["edges"])
    if any(w != 1.0 for _, _, w in inst["edges"]) or not nx.is_connected(g):
        return None
    if max(d for _, d in g.degree()) > 3 or (g.number_of_nodes() == 4 and g.number_of_edges() == 6):
        return None
    return float(g.number_of_edges())


# ---- scores of every basis state -------------------------------------------------------------------------

def decode_binary(idx, n):
    """Color of every node for basis index idx (2 bits per node: bit 2v is b0, bit 2v+1 is b1)."""
    return [min(((idx >> 2 * v) & 1) + 2 * ((idx >> 2 * v + 1) & 1), 2) for v in range(n)]


def binary_cost(n, edges):
    idx = np.arange(4 ** n, dtype=np.int64)
    col = [np.minimum(((idx >> 2 * v) & 1) + 2 * ((idx >> 2 * v + 1) & 1), 2) for v in range(n)]
    cost = np.zeros(4 ** n)
    for a, b, w in edges:
        cost += w * (col[a] != col[b])
    return cost


def onehot_cost(inst):
    """Cut of the coloring, 0 when any node is not one-hot (the same as core.cost_vector for max-3-cut)."""
    return core.cost_vector(dict(inst, kind="max3cut"))


def onehot_feasible(n):
    idx = np.arange(8 ** n, dtype=np.int64)
    ok = np.ones(8 ** n, bool)
    for v in range(n):
        ok &= (((idx >> 3 * v) & 1) + ((idx >> 3 * v + 1) & 1) + ((idx >> 3 * v + 2) & 1)) == 1
    return ok


# ---- phase layers ------------------------------------------------------------------------------------------

def qubo_phase(qc, Q, gamma):
    """exp(-i gamma f(x)) with f(x) = -x^T Q x (maximize f = minimize the QUBO energy), up to a global phase.
    x_i = (1 - Z_i)/2: x_i x_j = (1 - Z_i - Z_j + Z_i Z_j)/4."""
    n = len(Q)
    z = [0.0] * n                                         # exp(i a_i Z_i) = rz(-2 a_i)
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i][j]:
                qc.rzz(-gamma * Q[i][j] / 2, i, j)       # exp(i gamma Q_ij Z_i Z_j / 4)
                z[i] -= Q[i][j] / 4
                z[j] -= Q[i][j] / 4
        z[i] -= Q[i][i] / 2
    for i in range(n):
        if z[i]:
            qc.rz(-2 * gamma * z[i], i)


def diag_phase(qc, qubits, values, gamma):
    """exp(-i gamma diag(values)) on `qubits` (values indexed with qubits[k] as bit k), as Z-string rotations:
    values = sum_S c_S Z_S (Walsh-Hadamard); each exp(-i gamma c_S Z_S) is a CNOT ladder, rz, and the ladder back."""
    k = len(qubits)
    vals = np.asarray(values, float)
    for mask in range(1, 2 ** k):
        sign = np.array([(-1) ** bin(i & mask).count("1") for i in range(2 ** k)])
        c = float(vals @ sign) / 2 ** k
        if abs(c) < 1e-12:
            continue
        qs = [qubits[t] for t in range(k) if mask >> t & 1]
        for a, b in zip(qs, qs[1:]):
            qc.cx(a, b)
        qc.rz(2 * gamma * c, qs[-1])
        for a, b in reversed(list(zip(qs, qs[1:]))):
            qc.cx(a, b)


def edge_table_binary(w):
    """w [color(u) != color(v)] over the 4 bits (b0_u, b1_u, b0_v, b1_v): with diag_phase, exp(-i gamma cut_edge),
    the same sign convention as the Max-Cut circuits."""
    out = []
    for i in range(16):
        cu = min((i & 1) + 2 * (i >> 1 & 1), 2)
        cv = min((i >> 2 & 1) + 2 * (i >> 3 & 1), 2)
        out.append(w * (cu != cv))
    return out


def penalty_weight(inst):
    """P larger than any one node's weighted degree, so breaking a row never pays."""
    deg = [0.0] * inst["nodes"]
    for a, b, w in inst["edges"]:
        deg[a] += w
        deg[b] += w
    return 1.0 + max(deg)


def circuit(inst, encoding, p):
    """(circuit without measurements, parameter vector) for one encoding; no score vector, so any size."""
    n = inst["nodes"]
    th = ParameterVector("t", 2 * p)
    if encoding == "onehot-xy":
        return core.build(dict(inst, kind="max3cut"), f"xy-p{p}")
    if encoding == "onehot-penalty":
        Q, _ = ak.max3cut_qubo(n, inst["edges"], penalty=penalty_weight(inst))
        qc = QuantumCircuit(3 * n)
        qc.h(range(3 * n))
        for k in range(p):
            qubo_phase(qc, Q, th[2 * k])
            qc.rx(2 * th[2 * k + 1], range(3 * n))
        return qc, th
    if encoding == "binary":
        qc = QuantumCircuit(2 * n)
        qc.h(range(2 * n))
        for k in range(p):
            for a, b, w in inst["edges"]:
                diag_phase(qc, [2 * a, 2 * a + 1, 2 * b, 2 * b + 1], edge_table_binary(w), th[2 * k])
            qc.rx(2 * th[2 * k + 1], range(2 * n))
        return qc, th
    raise ValueError(encoding)


def scores(inst, encoding):
    """(score of every basis state, feasible mask or None). Exponential in n: small graphs only."""
    n = inst["nodes"]
    if encoding in ("onehot-xy", "onehot-penalty"):
        return onehot_cost(inst), onehot_feasible(n)
    if encoding == "binary":
        return binary_cost(n, inst["edges"]), None
    raise ValueError(encoding)


def build(inst, encoding, p):
    """(circuit without measurements, parameter vector, score of every basis state, feasible mask or None)."""
    qc, th = circuit(inst, encoding, p)
    return (qc, th) + scores(inst, encoding)


def start_ratio(encoding, cost, feas, opt):
    """Expected score of the encoding's own start state over the optimum: the uniform superposition for the
    penalty and binary encodings (H on every qubit), a W state per node (uniform over one-hot strings) for the XY
    encoding. The baseline that encoding's QAOA layers have to beat."""
    cost = np.asarray(cost, float)
    if encoding == "onehot-xy":
        return float(cost[feas].mean() / opt)
    return float(cost.mean() / opt)


# ---- the optimizer side ---------------------------------------------------------------------------------

def decode_onehot_cut(x, inst):
    """Cut of a one-hot assignment, or 0 if a node is not one-hot (as on the QAOA side)."""
    n = inst["nodes"]
    col = []
    for v in range(n):
        bits = x[3 * v:3 * v + 3]
        if sum(bits) != 1:
            return 0.0, False
        col.append(list(bits).index(1))
    return float(sum(w for a, b, w in inst["edges"] if col[a] != col[b])), True


def optimizer_side(inst, opt, a, client, res, base):
    import fourcolors.qubo

    n = inst["nodes"]
    P = penalty_weight(inst)
    Qp, off = ak.max3cut_qubo(n, inst["edges"], penalty=P)
    models = {"onehot-penalty": fourcolors.qubo.to_model(Qp, offset=off), "onehot-rows": ak.max3cut_model(n, inst["edges"])}
    for name, m in models.items():
        for limit in a.qubo_limits:
            for s in a.qubo_seeds:
                r, rec = ak.solve_qubo(m, time_limit_s=limit, seed=s, client=client)
                cut, feas = decode_onehot_cut(r.solution, inst) if (r is not None and r.solution is not None) else (None, None)
                res.add(kind="optimizer", encoding=name, model_variables=m.n, model_rows=len(m.constraints),
                        penalty=P if name == "onehot-penalty" else None, cut=cut, feasible=feas,
                        classical_ratio=cut / opt if cut is not None else None, **base, **rec)
    res.add(kind="optimizer", encoding="binary", model_variables=2 * n, cut=None, classical_ratio=None,
            note="not a QUBO: the binary encoding's cost has 3- and 4-variable terms (needs auxiliary variables)", **base)


def count_side(inst, a, client, res):
    """Larger graphs: compile each encoding with 4colors for gate counts only (no angles, no simulation)."""
    base = dict(instance=inst["id"], n=inst["nodes"], n_edges=len(inst["edges"]), simulated=False)
    for enc in a.encodings.split(","):
        for p in [int(x) for x in a.count_ps.split(",")]:
            qc, _ = circuit(inst, enc, p)
            inp = az.to_input(qc)
            inp.measure_all()
            row0 = dict(base, kind="count", encoding=enc, p=p, n_qubits=qc.num_qubits,
                        input_two_qubit_gates=ak.gate_stats(inp)["two_qubit_gates"])
            for seed in [int(s) for s in a.count_seeds.split(",")]:
                out, rec = ak.compile_4colors(inp, budget_s=a.budget, seed=seed, client=client,
                                              approximation_mode="exact")
                res.add(**dict(row0, **rec))
                print(inst["id"], enc, p, "4colors", seed, rec.get("two_qubit_gates"), rec.get("refusal_code"),
                      flush=True)


def main(argv=None, instances=None, count_instances=None, optimizer_instances=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="m3c")
    ap.add_argument("--encodings", default=",".join(ENCODINGS))
    ap.add_argument("--ps", default="1,2")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--e2", type=float, default=3e-3)
    ap.add_argument("--e-ro", type=float, default=0.01)
    ap.add_argument("--trajectories", type=int, default=100)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--max-active", type=int, default=24)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=1500)
    ap.add_argument("--qubo-limits", default="0.01,1")
    ap.add_argument("--qubo-seeds", default="1,2,3,4,5")
    ap.add_argument("--count-graphs", default="12,24", help="3-regular graph sizes for gate counts only ('' = none)")
    ap.add_argument("--count-ps", default="1")
    ap.add_argument("--count-seeds", default="1,2,3")
    ap.add_argument("--optimizer-graphs", default="200", help="3-regular graph sizes for the optimizer ('' = none)")
    ap.add_argument("--arms", default="4colors,qiskit-o3",
                    help="compiler arms; without 4colors there are no API jobs (no optimizer, no count rows)")
    a = ap.parse_args(argv)
    arms = [x for x in a.arms.split(",") if x]
    if not arms or set(arms) - {"4colors", "qiskit-o3"}:
        ap.error(f"--arms must name 4colors and/or qiskit-o3, got {a.arms!r}")
    api = "4colors" in arms
    a.qubo_limits = [float(x) for x in a.qubo_limits.split(",")]
    a.qubo_seeds = [int(x) for x in a.qubo_seeds.split(",")]
    res = bk.Results(Path(a.out) / "results.jsonl")
    client = bk.client() if api else None
    insts = instances if instances is not None else [
        i for i in core.instances() if i["id"].startswith(tuple(a.only.split(","))) and i["kind"] == "max3cut"]
    for inst in insts:
        n = inst["nodes"]
        opt = ak.max3cut_optimum(n, inst["edges"])            # all 3^n colorings: a certificate
        base = dict(instance=inst["id"], n=n, edges=inst["edges"], exact_optimum=opt, optimum_certificate="brute-force",
                    random_ratio=2 / 3 * sum(w for _, _, w in inst["edges"]) / opt)
        if api:
            optimizer_side(inst, opt, a, client, res, base)
        for enc in a.encodings.split(","):
            prev = None
            cost, feas = scores(inst, enc)
            for p in [int(x) for x in a.ps.split(",")]:
                qc, th = circuit(inst, enc, p)
                ang, e, evals = core.tune_circuit(qc, th, cost, restarts=1 if prev else 4, seed=7,
                                                  init=core.interp(prev, p) if prev else None, maxiter=a.maxiter)
                prev = ang
                inp = az.to_input(qc)
                inp.measure_all()
                row0 = dict(base, kind="qaoa", encoding=enc, variant=enc, p=p, n_qubits=qc.num_qubits,
                            ideal_ratio=e / opt, start_ratio=start_ratio(enc, cost, feas, opt), tune_evals=evals,
                            angles=ang, tuned_on="score (0 for infeasible)",
                            start_color_probs=list(BINARY_COLOR_PROBS) if enc == "binary" else [1 / 3] * 3,
                            input_two_qubit_gates=ak.gate_stats(inp)["two_qubit_gates"],
                            penalty=penalty_weight(inst) if enc == "onehot-penalty" else None)
                for seed in [int(s) for s in a.seeds.split(",")]:
                    for arm in arms:
                        out, rec = (ak.compile_4colors(inp, budget_s=a.budget, seed=seed, client=client,
                                                       approximation_mode="exact")
                                    if arm == "4colors" else ak.compile_qiskit(inp, budget_s=a.budget, seed=seed))
                        row = dict(row0, **rec)
                        if out is not None:
                            bound = out.assign_parameters(dict(zip(th, ang)))
                            active = core.compact(bound, qc.num_qubits)[0].num_qubits
                            row.update(active_qubits=active, simulated=active <= a.max_active)
                            if row["simulated"]:
                                e0, _ = core.noisy_expectation(bound, qc.num_qubits, cost, 0.0, e_ro=0.0, threads=a.threads)
                                st, probs = core.noisy_ratio(bound, qc.num_qubits, cost, opt, a.e2, a.e_ro,
                                                             a.trajectories, a.batches, a.threads)
                                row.update(check_ok=abs(e0 - e) < 1e-6 * opt, e2=a.e2, e1=a.e2 / 10, e_ro=a.e_ro,
                                           noisy_ratio_mean=st["mean"], noisy_ratio_se=st["se"],
                                           noisy_feasible=float(np.asarray(probs)[feas].sum()) if feas is not None else 1.0)
                        res.add(**row)
                        print(inst["id"], enc, p, arm, seed, rec.get("two_qubit_gates"), row.get("noisy_ratio_mean"),
                              rec.get("refusal_code"), flush=True)
    if not api:
        return
    counts = count_instances if count_instances is not None else [
        reg3_max3cut(int(n)) for n in a.count_graphs.split(",") if n]
    for inst in counts:
        count_side(inst, a, client, res)
    big = optimizer_instances if optimizer_instances is not None else [
        reg3_max3cut(int(n)) for n in a.optimizer_graphs.split(",") if n]
    for inst in big:
        opt = brooks_optimum(inst)
        if opt is None:
            raise SystemExit(f"{inst['id']}: no Brooks certificate (not connected, degree > 3, or K4)")
        base = dict(instance=inst["id"], n=inst["nodes"], exact_optimum=opt, optimum_certificate="brooks",
                    random_ratio=2 / 3, large=True)
        optimizer_side(inst, opt, a, client, res, base)


if __name__ == "__main__":
    main()
