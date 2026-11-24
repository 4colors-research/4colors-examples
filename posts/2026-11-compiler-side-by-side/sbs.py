"""Side-by-side inspection of compiled circuits (draft analysis code for the 'look inside the outputs' post).

Standard library only, so the same counting function runs on every arm's QASM without importing any
compiler. It reads OpenQASM 3 (and the simple OpenQASM 2 subset) as written by the 4colors compiler,
Qiskit's qasm3 dumper and pytket, on physical qubits (`$n`) or on one full-device register (`q[n]`).

What it computes per circuit (one definition for every arm):
  metrics()        two-qubit gates, two-qubit depth, depth, 1q pulses (sx + x; rz is virtual), rz count
  coupler_counts() two-qubit gates per physical coupler (heat maps)
  blocks()         maximal runs of gates on one qubit pair; each block's 4x4 unitary is classified by the
                   Shende-Markov-Bullock invariant: minimal CZ/CNOT count (0..3) and whether it is a SWAP up
                   to single-qubit gates. Actual CZs minus minimal CZs = gates a block-level resynthesis
                   could still remove ("slack"). Block.kind: stand-alone SWAP, move fused into a gate, or gate.
  layers()         ASAP layers of two-qubit gates (the per-layer timeline)
  track_swaps()    applies move blocks (SWAP-class and fused) to the initial layout and checks the result
                   against the final layout read from the terminal measurements
  esp()            estimated success probability from a published calibration (a model, not a fidelity)
and draws SVG figures (device heat map at the real heavy-hex layout, layer timeline, a logical qubit's route).

Definitions follow examples/blog/circuitkit.py: depth counts every gate except barrier/delay/measure/
reset (rz included); filtered operations still synchronize the qubits they touch, as in Qiskit's
QuantumCircuit.depth(filter).
"""
from __future__ import annotations

import ast
import cmath
import json
import math
import operator
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NON_GATES = {"barrier", "delay", "measure", "reset"}
PULSES = {"sx", "x", "sxdg"}          # physical 1q pulses on a {rz, sx, x} basis; rz is a frame change
TOL = 1e-6


# ---- parsing ----------------------------------------------------------------------------------------

@dataclass
class Op:
    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()
    clbits: Tuple[int, ...] = ()


@dataclass
class Circuit:
    ops: List[Op] = field(default_factory=list)
    n_qubits: int = 0
    n_clbits: int = 0
    annotations: Dict[str, List[str]] = field(default_factory=dict)
    global_phase: float = 0.0

    def measured(self) -> Dict[int, int]:
        """clbit -> qubit of the last measurement into each clbit."""
        out: Dict[int, int] = {}
        for op in self.ops:
            if op.name == "measure":
                out[op.clbits[0]] = op.qubits[0]
        return out


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow}
_NAMES = {"pi": math.pi, "π": math.pi, "tau": 2 * math.pi, "τ": 2 * math.pi, "euler": math.e}


def _num(expr: str) -> float:
    """Evaluate a QASM angle expression (numbers, pi, + - * / **, unary minus) without eval()."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id in _NAMES:
            return _NAMES[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        raise ValueError(f"unsupported angle expression: {expr!r}")
    return ev(ast.parse(expr.replace("π", "pi").replace("^", "**"), mode="eval"))


_DECL = re.compile(r"^(qubit|bit|qreg|creg)\s*(?:\[(\d+)\])?\s*([A-Za-z_]\w*)\s*(?:\[(\d+)\])?$")
_REF = re.compile(r"^\$(\d+)$|^([A-Za-z_]\w*)\s*\[(\d+)\]$|^([A-Za-z_]\w*)$")


def parse_qasm(text: str) -> Circuit:
    c = Circuit()
    qregs: Dict[str, Tuple[int, int]] = {}
    cregs: Dict[str, Tuple[int, int]] = {}
    nq = nc = 0
    phys_max = -1
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("@"):
            key, _, rest = s[1:].partition(" ")
            c.annotations[key] = rest.split()
            continue
        s = s.split("//")[0].strip()
        if not s:
            continue
        for stmt in s.split(";"):
            st = stmt.strip()
            if not st or st.startswith(("OPENQASM", "include")):
                continue
            m = _DECL.match(st)
            if m:
                kind, size1, name, size2 = m.groups()
                size = int(size1 or size2 or 1)
                if kind in ("qubit", "qreg"):
                    qregs[name] = (nq, size)
                    nq += size
                else:
                    cregs[name] = (nc, size)
                    nc += size
                continue
            if st.startswith(("gate ", "def ", "if", "while", "for ", "}")):
                raise ValueError(f"unsupported (dynamic or custom-gate) statement: {st[:60]!r}")

            def qref(tok: str) -> List[int]:
                nonlocal phys_max
                tok = tok.strip()
                m2 = _REF.match(tok)
                if not m2:
                    raise ValueError(f"bad operand {tok!r}")
                if m2.group(1) is not None:
                    q = int(m2.group(1))
                    phys_max = max(phys_max, q)
                    return [q]
                if m2.group(2) is not None:
                    off, _ = qregs[m2.group(2)]
                    return [off + int(m2.group(3))]
                off, size = qregs[m2.group(4)]
                return list(range(off, off + size))

            def cref(tok: str) -> int:
                tok = tok.strip()
                m2 = re.match(r"^([A-Za-z_]\w*)\s*\[(\d+)\]$", tok)
                if not m2:
                    raise ValueError(f"bad clbit {tok!r}")
                return cregs[m2.group(1)][0] + int(m2.group(2))

            if "= measure" in st:                                    # QASM 3: c[i] = measure q;
                lhs, rhs = st.split("= measure")
                c.ops.append(Op("measure", tuple(qref(rhs)), (), (cref(lhs),)))
                continue
            if st.startswith("measure") and "->" in st:              # QASM 2: measure q -> c;
                lhs, rhs = st[len("measure"):].split("->")
                c.ops.append(Op("measure", tuple(qref(lhs)), (), (cref(rhs),)))
                continue
            m = re.match(r"^([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*(.*)$", st)
            if not m:
                raise ValueError(f"cannot parse {st!r}")
            name, params, args = m.groups()
            pars = tuple(_num(p) for p in params.split(",")) if params else ()
            if name == "gphase":
                c.global_phase += pars[0] if pars else 0.0
                continue
            qs: List[int] = []
            for tok in [a for a in args.split(",") if a.strip()]:
                qs.extend(qref(tok))
            if name == "barrier" and not qs:
                qs = list(range(max(nq, phys_max + 1)))
            c.ops.append(Op(name, tuple(qs), pars))
    c.n_qubits = max(nq, phys_max + 1)
    c.n_clbits = nc
    return c


def load(path: str) -> Circuit:
    with open(path) as f:
        return parse_qasm(f.read())


# ---- metrics ----------------------------------------------------------------------------------------

def is_2q(op: Op) -> bool:
    return len(op.qubits) == 2 and op.name not in NON_GATES


def _depth(c: Circuit, keep) -> int:
    """Qiskit QuantumCircuit.depth(filter) semantics: kept ops add one level on all their wires (qubits and
    clbits); dropped ops still align their wires to the maximum level."""
    lev: Dict[Tuple[str, int], int] = {}
    best = 0
    for op in c.ops:
        wires = [("q", q) for q in op.qubits] + [("c", b) for b in op.clbits]
        if not wires:
            continue
        inc = 1 if keep(op) else 0
        top = max(lev.get(w, 0) for w in wires) + inc
        for w in wires:
            lev[w] = top
        best = max(best, top)
    return best


def metrics(c: Circuit) -> Dict[str, int]:
    names = Counter(op.name for op in c.ops)
    return {
        "two_qubit_gates": sum(1 for op in c.ops if is_2q(op)),
        "two_qubit_depth": _depth(c, is_2q),
        "depth": _depth(c, lambda op: op.name not in NON_GATES),
        "depth_with_measure": _depth(c, lambda op: op.name not in ("barrier", "delay")),
        "one_qubit_pulses": sum(names[g] for g in PULSES),
        "rz": names["rz"],
        "measurements": names["measure"],
        "ops": dict(sorted(names.items())),
    }


def coupler_counts(c: Circuit) -> Dict[Tuple[int, int], int]:
    return dict(Counter(tuple(sorted(op.qubits)) for op in c.ops if is_2q(op)))


def layers(c: Circuit) -> List[Tuple[int, int, int]]:
    """(layer, a, b) for every two-qubit gate, ASAP layering over two-qubit gates (layer 1 first); barriers
    align, exactly as two_qubit_depth counts."""
    lev: Dict[int, int] = {}
    out = []
    for op in c.ops:
        if not op.qubits:
            continue
        inc = 1 if is_2q(op) else 0
        top = max(lev.get(q, 0) for q in op.qubits) + inc
        for q in op.qubits:
            lev[q] = top
        if inc:
            a, b = sorted(op.qubits)
            out.append((top, a, b))
    return out


# ---- 2x2 / 4x4 matrices (lists of complex) -----------------------------------------------------------

def _mm(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _kron(A, B):
    return [[A[i // 2][j // 2] * B[i % 2][j % 2] for j in range(4)] for i in range(4)]


def _det(M):
    n = len(M)
    M = [row[:] for row in M]
    d = 1 + 0j
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[p][i]) < 1e-15:
            return 0j
        if p != i:
            M[i], M[p] = M[p], M[i]
            d = -d
        d *= M[i][i]
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for k in range(i, n):
                M[r][k] -= f * M[i][k]
    return d


I2 = [[1, 0], [0, 1]]
_s2 = 1 / math.sqrt(2)


def mat1(name: str, p: Sequence[float]):
    """2x2 unitary of a single-qubit gate (global phase irrelevant here)."""
    e = cmath.exp
    if name == "rz":
        t = p[0]
        return [[e(-1j * t / 2), 0], [0, e(1j * t / 2)]]
    if name in ("p", "u1"):
        return [[1, 0], [0, e(1j * p[0])]]
    if name == "sx":
        return [[(1 + 1j) / 2, (1 - 1j) / 2], [(1 - 1j) / 2, (1 + 1j) / 2]]
    if name == "sxdg":
        return [[(1 - 1j) / 2, (1 + 1j) / 2], [(1 + 1j) / 2, (1 - 1j) / 2]]
    if name == "x":
        return [[0, 1], [1, 0]]
    if name == "y":
        return [[0, -1j], [1j, 0]]
    if name == "z":
        return [[1, 0], [0, -1]]
    if name == "h":
        return [[_s2, _s2], [_s2, -_s2]]
    if name == "s":
        return [[1, 0], [0, 1j]]
    if name == "sdg":
        return [[1, 0], [0, -1j]]
    if name == "t":
        return [[1, 0], [0, e(1j * math.pi / 4)]]
    if name == "tdg":
        return [[1, 0], [0, e(-1j * math.pi / 4)]]
    if name == "id":
        return I2
    if name == "rx":
        c_, s_ = math.cos(p[0] / 2), math.sin(p[0] / 2)
        return [[c_, -1j * s_], [-1j * s_, c_]]
    if name == "ry":
        c_, s_ = math.cos(p[0] / 2), math.sin(p[0] / 2)
        return [[c_, -s_], [s_, c_]]
    if name in ("u", "u3"):
        th, ph, la = p
        c_, s_ = math.cos(th / 2), math.sin(th / 2)
        return [[c_, -e(1j * la) * s_], [e(1j * ph) * s_, e(1j * (ph + la)) * c_]]
    raise KeyError(name)


_CZ = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
_SWAP = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
_CX01 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]     # control = first qubit of the pair


def mat2(name: str, p: Sequence[float], first_is_lo: bool):
    """4x4 unitary on (lo, hi) with basis index 2*b_lo + b_hi; `first_is_lo` says the gate's first operand
    is the pair's lower qubit."""
    if name == "cz":
        return _CZ
    if name == "swap":
        return _SWAP
    if name in ("cx", "cnot"):
        return _CX01 if first_is_lo else _mm(_SWAP, _mm(_CX01, _SWAP))
    if name == "rzz":
        t = p[0]
        a, b = cmath.exp(-1j * t / 2), cmath.exp(1j * t / 2)
        return [[a, 0, 0, 0], [0, b, 0, 0], [0, 0, b, 0], [0, 0, 0, a]]
    if name == "cp":
        return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, cmath.exp(1j * p[0])]]
    if name == "ecr":
        # ECR = (IX - XY)/sqrt2 with the first operand as the "control" qubit
        X = [[0, 1], [1, 0]]
        Y = [[0, -1j], [1j, 0]]
        IX, XY = _kron(I2, X), _kron(X, Y)
        m = [[(IX[i][j] - XY[i][j]) * _s2 for j in range(4)] for i in range(4)]
        return m if first_is_lo else _mm(_SWAP, _mm(m, _SWAP))
    raise KeyError(name)


_YY = _kron([[0, -1j], [1j, 0]], [[0, -1j], [1j, 0]])


def classify(U) -> Dict[str, object]:
    """Minimal CNOT/CZ count of a two-qubit unitary and whether it is a SWAP up to 1q gates.

    Shende, Markov, Bullock, PRA 69, 062321 (2004): with U in SU(4) and gamma(U) = U (Y x Y) U^T (Y x Y),
    0 CNOTs iff gamma = +-I; 1 iff tr gamma = 0 and gamma^2 = -I; 2 iff tr gamma is real; else 3.
    gamma = +-iI is exactly the SWAP class (Weyl coordinates pi/4, pi/4, pi/4)."""
    d = _det(U)
    ph = cmath.exp(-1j * cmath.phase(d) / 4) / (abs(d) ** 0.25)
    V = [[x * ph for x in row] for row in U]
    VT = [list(r) for r in zip(*V)]
    g = _mm(_mm(V, _YY), _mm(VT, _YY))
    tr = sum(g[i][i] for i in range(4))
    off = max(abs(g[i][j]) for i in range(4) for j in range(4) if i != j)
    diag_equal = off < TOL and max(abs(g[i][i] - g[0][0]) for i in range(4)) < TOL
    swap_class = diag_equal and abs(g[0][0].real) < TOL
    if diag_equal and abs(g[0][0].imag) < TOL:
        k = 0
    else:
        g2 = _mm(g, g)
        neg_i = all(abs(g2[i][j] - (-1 if i == j else 0)) < TOL for i in range(4) for j in range(4))
        if abs(tr) < TOL and neg_i:
            k = 1
        elif abs(tr.imag) < TOL:
            k = 2
        else:
            k = 3
    return {"min_2q": k, "swap_class": swap_class}


# ---- blocks -----------------------------------------------------------------------------------------

@dataclass
class Block:
    pair: Tuple[int, int]
    start: int                 # index into ops of the first two-qubit gate
    n2: int = 0
    ops: List[int] = field(default_factory=list)
    layer: int = 0             # two-qubit layer of its first gate
    min_2q: Optional[int] = None
    swap_class: Optional[bool] = None
    min_2q_without_swap: Optional[int] = None   # minimal count of SWAP x U: the block with one move taken out

    @property
    def slack(self) -> Optional[int]:
        return None if self.min_2q is None else self.n2 - self.min_2q

    @property
    def kind(self) -> Optional[str]:
        """'swap'  SWAP-class block: a pure move (3 CZ at best), no useful interaction;
        'fused' a move merged into an interaction: U = SWAP x V with V strictly cheaper than U (CX + SWAP needs
                2 CZ, CP(t) + SWAP 3 CZ), so the block both does a logical gate and exchanges the two qubits;
        'gate'  everything else (an interaction with no move in it);
        None    unclassified (a gate the matrix table does not know).
        'fused' is a property of the block's unitary alone; a logical gate that happens to have that form (two
        opposite CXs) also counts, which the endpoint check in track_swaps() would expose."""
        if self.min_2q is None:
            return None
        if self.swap_class:
            return "swap"
        if self.min_2q_without_swap is not None and self.min_2q_without_swap < self.min_2q:
            return "fused"
        return "gate"


def blocks(c: Circuit) -> List[Block]:
    """Maximal runs of gates on one qubit pair: a two-qubit gate on (a, b) joins the block open on both a
    and b if it is that pair's block, else opens a new one; 1q gates join their qubit's open block; any
    other multi-qubit op, measurement or barrier closes the blocks on its qubits. Leading 1q gates are
    not part of a block (they cannot change its class)."""
    open_: Dict[int, Block] = {}
    out: List[Block] = []
    lay = _layer_of_ops(c)
    for i, op in enumerate(c.ops):
        if is_2q(op):
            pair = tuple(sorted(op.qubits))
            a, b = pair
            blk = open_.get(a)
            if blk is None or blk.pair != pair or open_.get(b) is not blk:
                for q in pair:
                    open_.pop(q, None)
                blk = Block(pair=pair, start=i, layer=lay[i])
                out.append(blk)
                open_[a] = open_[b] = blk
            blk.ops.append(i)
            blk.n2 += 1
        elif len(op.qubits) == 1 and op.name not in NON_GATES:
            blk = open_.get(op.qubits[0])
            if blk is not None:
                blk.ops.append(i)
        else:
            for q in op.qubits:
                open_.pop(q, None)
    for blk in out:
        _classify_block(c, blk)
    return out


def _layer_of_ops(c: Circuit) -> List[int]:
    lev: Dict[int, int] = {}
    res = []
    for op in c.ops:
        if not op.qubits:
            res.append(0)
            continue
        inc = 1 if is_2q(op) else 0
        top = max(lev.get(q, 0) for q in op.qubits) + inc
        for q in op.qubits:
            lev[q] = top
        res.append(top)
    return res


def block_unitary(c: Circuit, blk: Block):
    lo, hi = blk.pair
    U = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
    for i in blk.ops:
        op = c.ops[i]
        if len(op.qubits) == 1:
            g = mat1(op.name, op.params)
            G = _kron(g, I2) if op.qubits[0] == lo else _kron(I2, g)
        else:
            G = mat2(op.name, op.params, op.qubits[0] == lo)
        U = _mm(G, U)
    return U


def _classify_block(c: Circuit, blk: Block) -> None:
    try:
        U = block_unitary(c, blk)
    except KeyError:
        return
    r = classify(U)
    blk.min_2q, blk.swap_class = r["min_2q"], r["swap_class"]
    blk.min_2q_without_swap = classify(_mm(_SWAP, U))["min_2q"]


def block_summary(bl: List[Block]) -> Dict[str, object]:
    hist = Counter((b.n2, b.min_2q, bool(b.swap_class)) for b in bl)
    known = [b for b in bl if b.min_2q is not None]
    return {
        "blocks": len(bl),
        "unclassified_blocks": len(bl) - len(known),
        "swap_class_blocks": sum(1 for b in known if b.swap_class),
        "two_qubit_gates_in_swap_blocks": sum(b.n2 for b in known if b.swap_class),
        "fused_move_blocks": sum(1 for b in known if b.kind == "fused"),
        "two_qubit_gates_in_fused_blocks": sum(b.n2 for b in known if b.kind == "fused"),
        "slack_2q": sum(b.slack for b in known),
        "blocks_with_slack": sum(1 for b in known if b.slack > 0),
        "histogram": [{"n2": k[0], "min_2q": k[1], "swap_class": k[2], "count": v}
                      for k, v in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0][0]))],
    }


# ---- layouts and SWAP tracking -----------------------------------------------------------------------

def logical_measure_map(input_circ: Circuit) -> Dict[int, int]:
    """clbit -> logical qubit in the input circuit."""
    return input_circ.measured()


def final_layout(input_circ: Circuit, out: Circuit) -> Dict[int, int]:
    """logical -> physical at the end, read from terminal measurements of both circuits: where the data that
    started on logical qubit l is measured. SWAP gates in the input itself (e.g. a QFT's bit reversal) move
    that data to another logical wire, which is followed here so they are not mistaken for routing."""
    data_on = list(range(input_circ.n_qubits))          # logical wire -> logical data now on it
    lmap: Dict[int, int] = {}
    for op in input_circ.ops:
        if op.name == "swap" and len(op.qubits) == 2:
            a, b = op.qubits
            data_on[a], data_on[b] = data_on[b], data_on[a]
        elif op.name == "measure":
            lmap[op.clbits[0]] = data_on[op.qubits[0]]
    pmap = out.measured()
    return {lmap[cb]: pmap[cb] for cb in lmap if cb in pmap}


def initial_layout(out: Circuit, layouts_json: Optional[dict] = None, n_logical: Optional[int] = None
                   ) -> Optional[List[int]]:
    """logical -> physical at the start: the 4colors annotation, else a rival's recorded layouts.json."""
    ann = out.annotations.get("fourcolors.initial_layout")
    if ann:
        lay = [int(x) for x in ann]
    elif layouts_json and layouts_json.get("initial"):
        lay = [int(x) for x in layouts_json["initial"]]
    else:
        return None
    return lay[:n_logical] if n_logical else lay


def _simulate_moves(initial: Sequence[int], moves: List[Block]):
    occ = {p: l for l, p in enumerate(initial)}
    path: Dict[int, List[Tuple[int, int]]] = {l: [(0, p)] for l, p in enumerate(initial)}
    for b in sorted(moves, key=lambda b: b.start):
        x, y = b.pair
        lx, ly = occ.pop(x, None), occ.pop(y, None)
        if lx is not None:
            occ[y] = lx
            path[lx].append((b.layer, y))
        if ly is not None:
            occ[x] = ly
            path[ly].append((b.layer, x))
    return {l: p for p, l in occ.items()}, path


def track_swaps(initial: Sequence[int], bl: List[Block], final: Dict[int, int], resolve: bool = True
                ) -> Dict[str, object]:
    """Move logical qubits along move blocks (in program order: SWAP-class blocks and moves fused into an
    interaction, Block.kind 'swap' or 'fused') and compare with the measured final layout. `consistent`
    says those structural moves alone reproduce it. With `resolve`, any remaining mismatch is attacked
    greedily: other blocks with minimal count >= 2 are added as moves while that strictly lowers the number
    of mismatched logical qubits; the result says how many were needed and whether everything then agrees.
    The resolution is an explanation consistent with the endpoints, not a proof of what the compiler did."""
    def miss(moves):
        tracked, _ = _simulate_moves(initial, moves)
        return sorted(l for l, p in final.items() if tracked.get(l) != p)

    moves = [b for b in bl if b.kind in ("swap", "fused")]
    bad0 = miss(moves)
    hidden: List[Block] = []
    bad = bad0
    if resolve and bad:
        cands = [b for b in bl if b.kind not in ("swap", "fused") and (b.min_2q or 0) >= 2]
        improved = True
        while bad and improved:
            improved = False
            for b in cands:
                if b in hidden:
                    continue
                trial = miss(moves + hidden + [b])
                if len(trial) < len(bad):
                    hidden.append(b)
                    bad = trial
                    improved = True
                    if not bad:
                        break
    tracked, path = _simulate_moves(initial, moves + hidden)
    return {"consistent": not bad0, "mismatched_logical": bad0, "hidden_moves": len(hidden),
            "consistent_after_resolve": not bad, "paths": path, "tracked_final": tracked,
            "hidden_move_blocks": [(b.pair, b.layer, b.n2) for b in hidden]}


# ---- ESP (a model) ----------------------------------------------------------------------------------

def esp(c: Circuit, cal: dict) -> float:
    """Gate-and-readout-only ESP (no idle or duration term). Product of (1 - error) over every gate and
    measurement with a published error (rz error 0),
    calibration in the 4colors calibration JSON format (format_version 1). A model, not a fidelity."""
    q1 = {q["qubit"]: q for q in cal["qubits"]}
    q2 = {tuple(sorted(cp["qubits"])): cp for cp in cal["couplers"]}
    logp = 0.0
    for op in c.ops:
        if op.name in ("barrier", "delay"):
            continue
        if op.name == "measure":
            e = (q1.get(op.qubits[0], {}).get("readout") or {}).get("error")
        elif len(op.qubits) == 1:
            e = ((q1.get(op.qubits[0], {}).get("gates") or {}).get(op.name) or {}).get("error")
        elif len(op.qubits) == 2:
            e = (q2.get(tuple(sorted(op.qubits))) or {}).get("error")
        else:
            e = None
        if e is None:
            continue
        e = min(max(float(e), 0.0), 1.0)
        if e >= 1.0:
            return 0.0
        logp += math.log1p(-e)
    return math.exp(logp)


# ---- device geometry --------------------------------------------------------------------------------

def heavy_hex_positions(n: int, couplers: Iterable[Sequence[int]]) -> Optional[Dict[int, Tuple[float, float]]]:
    """Grid positions of an IBM-numbered heavy-hex device (Eagle, Heron), derived from the coupling map alone.

    IBM numbers these devices row by row, top to bottom: a coupler between consecutive ids is a horizontal
    step inside a row, any other coupler is a vertical step to the bridge qubit below (or the row below a
    bridge). A breadth-first walk from the lowest connected qubit places every qubit on that rule, so missing
    couplers (a broken edge splits a row in the id sequence) do not matter while the graph stays connected.
    A qubit with no coupler at all (a disabled one) is put back in its row between ids q-1 and q+1.
    Returns None unless the walk is consistent: every coupler of unit length and no two qubits on one spot."""
    und = {tuple(sorted(int(x) for x in e)) for e in couplers}
    nbr: Dict[int, set] = {q: set() for q in range(n)}
    for a, b in und:
        if a == b or not (0 <= a < n and 0 <= b < n):
            return None
        nbr[a].add(b)
        nbr[b].add(a)
    roots = [q for q in range(n) if nbr[q]]
    if not roots:
        return None
    pos: Dict[int, Tuple[float, float]] = {roots[0]: (0.0, 0.0)}
    todo = [roots[0]]
    while todo:
        a = todo.pop(0)
        for b in sorted(nbr[a]):
            ax, ay = pos[a]
            p = (ax + (b - a), ay) if abs(a - b) == 1 else (ax, ay + (1.0 if b > a else -1.0))
            if b in pos:
                if pos[b] != p:
                    return None
            else:
                pos[b] = p
                todo.append(b)
    for q in range(n):                                # isolated (disabled) qubits: back into their row
        if q in pos:
            continue
        if nbr[q]:
            return None                               # a second connected component: not one lattice
        left, right = pos.get(q - 1), pos.get(q + 1)
        if left and right and left[1] == right[1] and right[0] - left[0] == 2:
            pos[q] = (left[0] + 1, left[1])
        elif left and (left[0] + 1, left[1]) not in pos.values():
            pos[q] = (left[0] + 1, left[1])
        elif right and (right[0] - 1, right[1]) not in pos.values():
            pos[q] = (right[0] - 1, right[1])
        else:
            return None
    if len(set(pos.values())) != n:
        return None
    x0, y0 = min(p[0] for p in pos.values()), min(p[1] for p in pos.values())
    return {q: (p[0] - x0, p[1] - y0) for q, p in pos.items()}


def lattice_positions(n: int, couplers: Iterable[Sequence[int]]) -> Dict[int, Tuple[float, float]]:
    """Fallback placement as in circuitkit.lattice_positions (rows of consecutive ids joined by (i, i+1),
    the rest as bridges between rows); a ring for numberings that fit neither."""
    und = {tuple(sorted(int(x) for x in e)) for e in couplers}
    nbr: Dict[int, set] = {q: set() for q in range(n)}
    for a, b in und:
        nbr[a].add(b)
        nbr[b].add(a)
    rows, cur = [], [0]
    for q in range(1, n):
        if (q - 1, q) in und:
            cur.append(q)
        else:
            rows.append(cur)
            cur = [q]
    rows.append(cur)
    main = [r for r in rows if len(r) > 4]
    bridges = [q for r in rows if len(r) <= 4 for q in r]
    pos: Dict[int, Tuple[float, float]] = {}
    if len(main) >= 2:
        for ri, r in enumerate(main):
            for col, q in enumerate(r):
                pos[q] = (float(col), 2.0 * ri)
        for q in bridges:
            known = [pos[x] for x in nbr[q] if x in pos]
            if known:
                pos[q] = (sum(p[0] for p in known) / len(known), sum(p[1] for p in known) / len(known))
        if len(pos) == n:
            return pos
    r = max(1.0, n / (2 * math.pi))
    return {q: (r + r * math.cos(2 * math.pi * q / n), r + r * math.sin(2 * math.pi * q / n)) for q in range(n)}


def device_positions(n: int, couplers) -> Dict[int, Tuple[float, float]]:
    """The device's own layout when the coupling map is an IBM-numbered heavy-hex lattice, else the fallback."""
    couplers = [tuple(e) for e in couplers]
    return heavy_hex_positions(n, couplers) or lattice_positions(n, couplers)


# ---- SVG figures ------------------------------------------------------------------------------------

INK, INK_2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
IDLE_Q = "#c9c8c3"                            # a qubit the arm never uses
SEQ = ["#cde2fb", "#3987e5", "#0d366b"]       # one-hue sequential ramp (circuitkit.draw_device)
WORK, SWAPC, FUSED = "#2a78d6", "#eb6834", "#1baf7a"   # examplekit.SERIES slots 1, 2, 3
PATHC = "#e34948"                             # examplekit.SERIES slot 8: a logical qubit's route
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
KIND_COLOR = {"gate": WORK, "fused": FUSED, "swap": SWAPC, None: WORK}
KIND_LABEL = {"gate": "two-qubit gate, no move",
              "fused": "gate + move, fused into one block",
              "swap": "stand-alone SWAP (move only)"}


def _lerp(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def seq_color(v: float, top: float) -> str:
    t = 0.0 if top <= 0 else min(1.0, v / top)
    return _lerp(SEQ[0], SEQ[1], t * 2) if t < 0.5 else _lerp(SEQ[1], SEQ[2], (t - 0.5) * 2)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_open(W: float, H: float) -> List[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" '
            f'height="{H:.0f}" font-family="{FONT}">',
            f'<rect width="100%" height="100%" fill="{SURFACE}"/>']


def _head(parts: List[str], title: str, subtitle: str, x: float = 16) -> None:
    parts.append(f'<text x="{x}" y="24" font-size="15" font-weight="600" fill="{INK}">{_esc(title)}</text>')
    if subtitle:
        parts.append(f'<text x="{x}" y="43" font-size="12" fill="{INK_2}">{_esc(subtitle)}</text>')


def device_svg(n: int, couplers, counts: Dict[Tuple[int, int], int], *, title: str = "", subtitle: str = "",
               marked: Iterable[int] = (), ring: Iterable[int] = (), vmax: Optional[float] = None,
               path: Sequence[int] = (), scale: float = 24.0, ring_label: str = "busiest logical qubit, start",
               path_label: str = "its route", footer: str = "") -> str:
    """Device map at the device's real layout (heavy hex for IBM Eagle/Heron numbering): every coupler is a
    line, colored and widened by its two-qubit gate count (shared `vmax` keeps arms comparable); unused
    couplers are light grey, couplers absent from the coupling map are not drawn. Dark dots: `marked`
    qubits (the initial layout); open grey dots: qubits with no coupler (disabled); a ring on `ring` and an
    optional dashed route `path` of physical qubits. Couplers and qubits carry <title> tooltips."""
    pos = device_positions(n, couplers)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad, top_h, leg_h = 0.8, 56, 80
    pw, ph = (max(xs) - min(xs) + 2 * pad) * scale, (max(ys) - min(ys) + 2 * pad) * scale
    W, H = max(pw, 380) , top_h + ph + leg_h
    ox = (W - pw) / 2
    X = lambda q: ox + (pos[q][0] - min(xs) + pad) * scale
    Y = lambda q: top_h + (pos[q][1] - min(ys) + pad) * scale
    und = sorted({tuple(sorted(int(x) for x in e)) for e in couplers})
    deg = Counter(q for e in und for q in e)
    top = vmax or max(counts.values() or [1])
    parts = _svg_open(W, H)
    _head(parts, title, subtitle)
    for a, b in und:                                       # unused couplers first, used ones on top
        if counts.get((a, b), 0) == 0:
            parts.append(f'<line x1="{X(a):.1f}" y1="{Y(a):.1f}" x2="{X(b):.1f}" y2="{Y(b):.1f}" stroke="{GRID}" '
                         f'stroke-width="2.4" stroke-linecap="round"><title>{a}-{b}: unused</title></line>')
    for a, b in sorted(und, key=lambda e: counts.get(e, 0)):
        k = counts.get((a, b), 0)
        if k:
            parts.append(f'<line x1="{X(a):.1f}" y1="{Y(a):.1f}" x2="{X(b):.1f}" y2="{Y(b):.1f}" '
                         f'stroke="{seq_color(k, top)}" stroke-width="{3.2 + 3.6 * min(1, k / top):.1f}" '
                         f'stroke-linecap="round"><title>{a}-{b}: {k} two-qubit gates</title></line>')
    mk = set(marked)
    for q in range(n):
        if q not in pos:
            continue
        if deg[q] == 0:
            parts.append(f'<circle cx="{X(q):.1f}" cy="{Y(q):.1f}" r="2.6" fill="{SURFACE}" stroke="{MUTED}" '
                         f'stroke-width="1"><title>qubit {q}: no couplers</title></circle>')
        elif q in mk:
            parts.append(f'<circle cx="{X(q):.1f}" cy="{Y(q):.1f}" r="3.4" fill="{INK}"><title>qubit {q}: '
                         f'holds a logical qubit at the start</title></circle>')
        else:
            parts.append(f'<circle cx="{X(q):.1f}" cy="{Y(q):.1f}" r="2.2" fill="{IDLE_Q}"><title>qubit {q}'
                         f'</title></circle>')
    if len(path) > 1:
        pts = " ".join(f"{X(q):.1f},{Y(q):.1f}" for q in path)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{PATHC}" stroke-width="2.2" '
                     f'stroke-dasharray="5 3" stroke-linejoin="round"/>')
        parts.append(f'<circle cx="{X(path[-1]):.1f}" cy="{Y(path[-1]):.1f}" r="3.6" fill="{PATHC}">'
                     f'<title>qubit {path[-1]}: where it is measured</title></circle>')
    for q in ring:
        if q in pos:
            parts.append(f'<circle cx="{X(q):.1f}" cy="{Y(q):.1f}" r="7" fill="none" stroke="{PATHC}" '
                         f'stroke-width="2"/>')
    # legend: ramp + symbols, two rows
    ly = top_h + ph + 14
    gid = f"ramp{zlib.crc32((title + str(top)).encode()):08x}"
    parts.append(f'<defs><linearGradient id="{gid}"><stop offset="0" stop-color="{SEQ[0]}"/>'
                 f'<stop offset="0.5" stop-color="{SEQ[1]}"/><stop offset="1" stop-color="{SEQ[2]}"/>'
                 f'</linearGradient></defs>')
    parts.append(f'<rect x="16" y="{ly}" width="110" height="8" rx="2" fill="url(#{gid})"/>'
                 f'<text x="16" y="{ly + 21}" font-size="10.5" fill="{INK_2}">1</text>'
                 f'<text x="126" y="{ly + 21}" font-size="10.5" fill="{INK_2}" text-anchor="end">{top:g}</text>'
                 f'<text x="136" y="{ly + 8}" font-size="11" fill="{INK_2}">two-qubit gates per coupler</text>')
    ly2 = ly + 40
    parts.append(f'<circle cx="21" cy="{ly2 - 4}" r="3.4" fill="{INK}"/>'
                 f'<text x="30" y="{ly2}" font-size="11" fill="{INK_2}">start layout</text>')
    x = 112
    if ring:
        parts.append(f'<circle cx="{x + 5}" cy="{ly2 - 4}" r="6" fill="none" stroke="{PATHC}" stroke-width="2"/>'
                     f'<text x="{x + 16}" y="{ly2}" font-size="11" fill="{INK_2}">{_esc(ring_label)}</text>')
        x += 30 + 5.8 * len(ring_label)
    if len(path) > 1:
        parts.append(f'<line x1="{x}" y1="{ly2 - 4}" x2="{x + 18}" y2="{ly2 - 4}" stroke="{PATHC}" '
                     f'stroke-width="2.2" stroke-dasharray="5 3"/>'
                     f'<text x="{x + 24}" y="{ly2}" font-size="11" fill="{INK_2}">{_esc(path_label)}</text>')
    if footer:
        parts.append(f'<text x="{W - 12:.0f}" y="{H - 8:.0f}" font-size="10" fill="{MUTED}" text-anchor="end">'
                     f'{_esc(footer)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _nice_step(L: int) -> int:
    for s in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if L / s <= 8:
            return s
    return 2000


def gate_kinds(c: Circuit, bl: List[Block]) -> Dict[int, Optional[str]]:
    """op index -> kind of the block holding that two-qubit gate (see Block.kind)."""
    kind: Dict[int, Optional[str]] = {}
    for b in bl:
        for i in b.ops:
            if is_2q(c.ops[i]):
                kind[i] = b.kind
    return kind


def timeline_svg(c: Circuit, bl: List[Block], *, title: str = "", subtitle: str = "",
                 qubits: Optional[Sequence[int]] = None, col_w: float = 6.0, row_h: float = 6.0,
                 n_layers: Optional[int] = None, n_rows: Optional[int] = None) -> str:
    """Per-layer timeline: x = two-qubit layer (ASAP), y = the physical qubits the arm uses (in id order);
    each two-qubit gate is a vertical tick joining its qubits, colored by its block's kind: no move, a move
    fused into a gate, a stand-alone SWAP. Pass the same `col_w`, `row_h`, `n_layers` (the longest arm) and
    `n_rows` (the most qubits) to every arm of one comparison: then pixels mean the same in every panel and
    the panels have the same size."""
    lay = _layer_of_ops(c)
    kind = gate_kinds(c, bl)
    used = sorted({q for op in c.ops if is_2q(op) for q in op.qubits}) if qubits is None else list(qubits)
    row = {q: i for i, q in enumerate(used)}
    L_own = max([lay[i] for i, op in enumerate(c.ops) if is_2q(op)] or [0])
    L = max(n_layers or 0, L_own, 1)
    R = max(n_rows or 0, len(used), 1)
    left, right, top_h = 30, 16, 58
    pw, ph = L * col_w, R * row_h
    W = left + max(pw, 360) + right
    leg_y = top_h + ph + 38
    H = leg_y + 3 * 17 + 6
    parts = _svg_open(W, H)
    _head(parts, title, subtitle, x=left)
    # frame of the shared plot area; the arm's own extent is shaded
    parts.append(f'<rect x="{left}" y="{top_h}" width="{pw:.1f}" height="{ph:.1f}" fill="none" stroke="{GRID}" '
                 f'stroke-width="1"/>')
    parts.append(f'<rect x="{left}" y="{top_h}" width="{L_own * col_w:.1f}" height="{len(used) * row_h:.1f}" '
                 f'fill="#f3f2ee"/>')
    if row_h >= 4:
        for q, r in row.items():
            y = top_h + r * row_h + row_h / 2
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + L_own * col_w:.1f}" y2="{y:.1f}" '
                         f'stroke="{GRID}" stroke-width="0.6"/>')
    tw = max(1.0, min(2.2, col_w * 0.7))
    cnt = Counter()
    for i, op in enumerate(c.ops):
        if not is_2q(op):
            continue
        a, b = sorted(op.qubits)
        if a not in row or b not in row:
            continue
        k = kind.get(i)
        cnt[k or "gate"] += 1
        x = left + (lay[i] - 0.5) * col_w
        y1, y2 = top_h + row[a] * row_h + row_h * 0.2, top_h + row[b] * row_h + row_h * 0.8
        parts.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{KIND_COLOR[k]}" '
                     f'stroke-width="{tw:.1f}"><title>layer {lay[i]}: {op.name} {a},{b} '
                     f'({KIND_LABEL.get(k or "gate")})</title></line>')
    # x axis
    step = _nice_step(L)
    ax_y = top_h + ph
    for t in range(0, L + 1, step):
        x = left + t * col_w
        parts.append(f'<line x1="{x:.1f}" y1="{ax_y}" x2="{x:.1f}" y2="{ax_y + 4}" stroke="{MUTED}"/>'
                     f'<text x="{x:.1f}" y="{ax_y + 15}" font-size="10" fill="{INK_2}" text-anchor="middle">{t}'
                     f'</text>')
    parts.append(f'<text x="{left + pw:.1f}" y="{ax_y + 28}" font-size="10.5" fill="{INK_2}" text-anchor="end">'
                 f'two-qubit layer &#8594;</text>')
    parts.append(f'<text x="{left - 10}" y="{top_h + ph / 2:.1f}" font-size="10.5" fill="{INK_2}" '
                 f'text-anchor="middle" transform="rotate(-90 {left - 10} {top_h + ph / 2:.1f})">'
                 f'{len(used)} qubits used</text>')
    nblk = Counter((b.kind or "gate") for b in bl)
    for j, k in enumerate(("gate", "fused", "swap")):
        y = leg_y + j * 17
        unit = "two-qubit gates" if k == "gate" else f"two-qubit gates in {nblk[k]} block{'' if nblk[k] == 1 else 's'}"
        parts.append(f'<rect x="{left}" y="{y - 9}" width="14" height="10" rx="1.5" fill="{KIND_COLOR[k]}"/>'
                     f'<text x="{left + 20}" y="{y}" font-size="11" fill="{INK_2}">{KIND_LABEL[k]}: '
                     f'<tspan fill="{INK}" font-weight="600">{cnt[k]}</tspan> {unit}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ---- one arm, end to end ----------------------------------------------------------------------------

def inspect(input_circ: Circuit, out: Circuit, *, layouts_json: Optional[dict] = None,
            cal: Optional[dict] = None) -> Dict[str, object]:
    n_log = input_circ.n_qubits
    bl = blocks(out)
    init = initial_layout(out, layouts_json, n_log)
    fin = final_layout(input_circ, out)
    rep: Dict[str, object] = {"metrics": metrics(out), "blocks": block_summary(bl),
                              "initial_layout": init, "final_layout": [fin.get(l) for l in range(n_log)],
                              "couplers_used": len(coupler_counts(out))}
    if init is not None:
        tr = track_swaps(init, bl, fin)
        rep["swap_tracking"] = {"consistent": tr["consistent"], "mismatched_logical": tr["mismatched_logical"],
                                "hidden_moves": tr["hidden_moves"],
                                "consistent_after_resolve": tr["consistent_after_resolve"],
                                "hidden_move_blocks": tr["hidden_move_blocks"],
                                "moved_logical": sum(1 for l, p in tr["paths"].items() if len(p) > 1)}
        rep["_paths"] = tr["paths"]
    if cal is not None:
        rep["esp_gates_readout"] = esp(out, cal)
    rep["_blocks"] = bl
    return rep


def busiest_logical(input_circ: Circuit) -> int:
    """The logical qubit in the most two-qubit gates (BV's oracle qubit, a star's hub)."""
    cnt = Counter(q for op in input_circ.ops if is_2q(op) for q in op.qubits)
    return cnt.most_common(1)[0][0] if cnt else 0


def dumps(rep: Dict[str, object]) -> str:
    return json.dumps({k: v for k, v in rep.items() if not k.startswith("_")}, indent=1, sort_keys=True)
