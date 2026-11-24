"""Chips as target descriptions (format version 2), the way the 4colors compiler takes every device.

Two kinds, both sent through the public API with no name anywhere:

* shapes you draw yourself: `grid`, `ring`, `ladder`, `heavy_hex`, `all_to_all`, each built with
  `fourcolors.quantum.targets.build_description(n, "cz", ("rz", "sx", "x"), couplers)`;
* bundled design lattices: `fourcolors.quantum.targets.design_lattice(name)`, a full-size description of a
  device class (heavy-hex 156, square 108 and 54, all-to-all 56) with its native gates and no calibration.

Couplers are listed once, a < b, sorted. Qubits are numbered row by row. The public API caps a target at 156
qubits (check the live limit before a run); bigger chips are still built here, for Qiskit and for the refusal.
`qiskit_target` gives Qiskit the same qubits, couplers (both directions) and native gates, with no error rates:
both arms see only the map.
"""
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared"))

from fourcolors.quantum import targets

GATES_1Q = ("rz", "sx", "x")


def describe(n: int, couplers) -> dict:
    """A format version 2 description of an n-qubit cz chip with these couplers (or "all_to_all")."""
    if couplers != "all_to_all":
        couplers = sorted({tuple(sorted(int(x) for x in e)) for e in couplers})
    return targets.build_description(n, "cz", GATES_1Q, couplers)


def grid(rows: int, cols: int) -> dict:
    """A rows x cols square lattice: right and down neighbors."""
    q = lambda r, c: r * cols + c
    edges = [(q(r, c), q(r, c + 1)) for r in range(rows) for c in range(cols - 1)]
    edges += [(q(r, c), q(r + 1, c)) for r in range(rows - 1) for c in range(cols)]
    return describe(rows * cols, edges)


def ring(n: int) -> dict:
    """n qubits in a cycle."""
    return describe(n, [(i, (i + 1) % n) for i in range(n)])


def ladder(rungs: int) -> dict:
    """Two rails of `rungs` qubits joined at every position (a 2 x rungs grid)."""
    return grid(2, rungs)


def heavy_hex(rows: int, row_len: int) -> dict:
    """Heavy-hex style: `rows` lines of `row_len` qubits; between line r and r + 1 a bridge qubit at every
    column c with c % 4 == 0 (r even) or c % 4 == 2 (r odd), as on IBM's heavy-hex chips. Max degree 3.
    heavy_hex(3, 7) has 21 line qubits and 4 bridges: 25 qubits."""
    edges, n = [], rows * row_len
    q = lambda r, c: r * row_len + c
    for r in range(rows):
        edges += [(q(r, c), q(r, c + 1)) for c in range(row_len - 1)]
    for r in range(rows - 1):
        for c in range(row_len):
            if c % 4 == (0 if r % 2 == 0 else 2):
                edges += [(q(r, c), n), (n, q(r + 1, c))]
                n += 1
    return describe(n, edges)


def all_to_all(n: int) -> dict:
    """Every pair coupled ("all_to_all": no routing ever needed), the zero of the topology tax."""
    return describe(n, "all_to_all")


def description(chip) -> dict:
    """The description dict of a chip: a dict as built here, or a `DeviceTarget`'s `.description`."""
    return chip.description if isinstance(chip, targets.DeviceTarget) else chip


def coupler_list(chip):
    d = description(chip)
    c = d["couplers"]
    if c == "all_to_all":
        return [(a, b) for a in range(d["n_qubits"]) for b in range(a + 1, d["n_qubits"])]
    return [tuple(e) for e in c]


#: Description one-qubit gate names as Qiskit's standard gate names (prx is Qiskit's RGate, u1q a general
#: one-qubit gate, Qiskit's UGate).
QISKIT_1Q = {"rz": "rz", "sx": "sx", "x": "x", "rx": "rx", "prx": "r", "u1q": "u"}
QISKIT_2Q = {"cz": "cz", "rzz": "rzz", "rxx": "rxx"}


def qiskit_target(chip):
    """The Qiskit Target with the chip's qubits, couplers (both directions) and native gates, no error rates."""
    from qiskit.transpiler import CouplingMap, Target

    d = description(chip)
    g1 = d.get("gates_1q") or list(GATES_1Q)
    if d["two_qubit_gate"] not in QISKIT_2Q or any(g not in QISKIT_1Q for g in g1):
        raise ValueError(f"no Qiskit gate names for {d['two_qubit_gate']} / {g1}")
    pairs = coupler_list(d)
    cm = CouplingMap([[a, b] for a, b in pairs] + [[b, a] for a, b in pairs])
    return Target.from_configuration(basis_gates=[QISKIT_2Q[d["two_qubit_gate"]]] + [QISKIT_1Q[g] for g in g1]
                                     + ["measure"], num_qubits=int(d["n_qubits"]), coupling_map=cm)


#: The topology-tax chips: about 25 qubits each, so the same 12- to 16-qubit circuits fit all of them.
TAX_CHIPS = {"all_to_all_25": lambda: all_to_all(25), "grid_5x5": lambda: grid(5, 5),
             "heavy_hex_3x7": lambda: heavy_hex(3, 7), "ladder_2x13": lambda: ladder(13),
             "ring_25": lambda: ring(25)}

#: The scaling chips: square grids from 25 to 144 qubits (owner cap of 2 Nov; targets at most 156 qubits).
SCALE_GRIDS = (5, 8, 10, 12)

#: The full-size design lattices for the chemistry row (DOM CHEM-1: UCCSD for LiH on four chips). Each is a
#: bundled description with its own native gates; none carries a calibration.
DESIGN_CHIPS = ("heavy_hex_156_cz", "square_108_cz", "square_54_prx", "all_to_all_56_u1q")


def design(name: str):
    """A bundled design lattice as the client gives it (a `DeviceTarget` with no calibration)."""
    return targets.design_lattice(name)
