"""Ripple-carry adders, the arithmetic inside Shor's algorithm, and a classical check of compiled adders.

Two textbook adders from Qiskit's circuit library, both "full" (carry in and carry out):

* `cuccaro`: Cuccaro, Draper, Kutin and Moulton, "A new quantum ripple-carry addition circuit"
  (arXiv quant-ph/0410184, 2004), `CDKMRippleCarryAdder`: one ancilla (the carry in), 2n + 2 qubits;
* `vbe`: Vedral, Barenco and Ekert (1996), `VBERippleCarryAdder`: n - 1 helper qubits, 3n + 1 qubits.

Both compute |cin, a, b, cout> -> |cin, a, (a + b + cin) mod 2^n, carry> with helpers back in |0>. They are
reversible classical circuits, so a compiled adder can be checked on basis inputs: put random a, b, cin on the
qubits where the compiled circuit's layout placed them, run it, and read the sum where the layout says it ends.
That is evidence on the inputs tried, not a proof of equivalence on every input.
"""
from __future__ import annotations

import random
from typing import Dict, List

BASIS = ["cx", "rz", "rx", "ry", "h", "x", "sx", "p", "cp", "swap", "ccx"]
#: n bits per kind, chosen so the largest is 100 qubits.
SIZES = {"cuccaro": (16, 32, 49), "vbe": (11, 22, 33)}


def raw(kind: str, n: int):
    from qiskit.circuit.library import CDKMRippleCarryAdder, VBERippleCarryAdder

    return {"cuccaro": CDKMRippleCarryAdder, "vbe": VBERippleCarryAdder}[kind](n, kind="full")


def registers(kind: str, n: int) -> Dict[str, List[int]]:
    """Qubit indices of each register (cin, a, b, cout, helper) in the circuit `circuit(kind, n)`."""
    qc = raw(kind, n)
    return {r.name: [qc.find_bit(q).index for q in r] for r in qc.qregs}


def circuit(kind: str, n: int):
    """The adder block, lowered to standard gates (Toffolis kept as ccx), no measurements: the arithmetic a
    larger algorithm calls, compiled on its own."""
    from qiskit import transpile

    qc = transpile(raw(kind, n), basis_gates=BASIS, optimization_level=0)
    qc.name = f"{kind}_{n}"
    return qc


def expected(kind: str, n: int, a: int, b: int, cin: int) -> Dict[str, int]:
    s = a + b + cin
    return {"cin": cin, "a": a, "b": s % (1 << n), "cout": s >> n, "helper": 0}


def _bits(v: int, k: int) -> List[int]:
    return [(v >> i) & 1 for i in range(k)]


def check(compiled, kind: str, n: int, trials: int = 8, seed: int = 1, shots: int = 4) -> dict:
    """Run `compiled` (with a `layout`, as Qiskit's and 4colors' outputs have) on `trials` random inputs on
    Aer's matrix-product-state simulator and compare every shot with the classical sum."""
    from qiskit import ClassicalRegister
    from qiskit_aer import AerSimulator

    regs = registers(kind, n)
    width = sum(len(v) for v in regs.values())
    lay = compiled.layout
    init = lay.initial_index_layout(filter_ancillas=True) if lay is not None else list(range(width))
    final = lay.final_index_layout(filter_ancillas=True) if lay is not None else list(range(width))
    rng = random.Random(seed)
    sim = AerSimulator(method="matrix_product_state")
    failures = []
    core = compiled.remove_final_measurements(inplace=False)
    for t in range(trials):
        a, b, cin = rng.getrandbits(n), rng.getrandbits(n), rng.getrandbits(1)
        ins = {"cin": [cin], "a": _bits(a, n), "b": _bits(b, n)}
        qc = core.copy_empty_like()
        for name, bits in ins.items():
            for q, bit in zip(regs[name], bits):
                if bit:
                    qc.x(init[q])
        qc.compose(core, inplace=True)
        cr = ClassicalRegister(width, "out")
        qc.add_register(cr)
        for v in range(width):
            qc.measure(final[v], cr[v])
        counts = sim.run(qc, shots=shots).result().get_counts()
        want = expected(kind, n, a, b, cin)
        want_bits = [0] * width
        for name, idx in regs.items():
            for q, bit in zip(idx, _bits(want[name], len(idx))):
                want_bits[q] = bit
        want_key = "".join(str(x) for x in reversed(want_bits))
        if set(counts) != {want_key}:
            failures.append({"a": a, "b": b, "cin": cin, "got": sorted(counts)[:2]})
    return {"adder_check": "returns_sum" if not failures else "wrong", "adder_check_trials": trials,
            "adder_check_failures": failures[:3], "adder_check_method": "aer_mps_basis_inputs"}
