"""Custom chips: compile for a device you describe yourself.

    python run.py --out results/chips [--budgets 10] [--seeds 1,2,3,4,5] [--part tax,scale,design]

Part "tax": four circuits (QAOA on a 3-regular graph, 16 qubits; BV with 15 bits; an 8-bit Draper adder; UCCSD on
the LiH STO-3G register, 12 qubits) on five chips of about 25 qubits (all-to-all, grid, heavy-hex style 3x7,
ladder, ring). The topology tax of a chip is its extra two-qubit gates over the all-to-all chip.

Part "design": the UCCSD circuit on four bundled full-size design lattices (`targets.design_lattice`: heavy-hex 156,
square 108, square 54 with prx gates, all-to-all 56 with rzz), each with its own native gates.

Part "scale": QAOA on 16 and 64 nodes on square grids from 5x5 to 12x12 (--grids; 144 qubits). The public API caps
a custom target at 156 qubits, and the post uses only what a customer can send: a bigger chip (the probe, --grids 20)
is refused, and every refusal is a row with its code, never dropped.

Both arms get the same chip: 4colors as the target description, Qiskit as a Target with the same couplers and
native gates (chips.qiskit_target). No chip carries a calibration: both arms see only the map.

Every "result" row of both arms carries `coupler_counts` ([[a, b, two-qubit gates on (a, b)], ...] on physical
qubits): the arms check compares it, and the check before citing confirms each pair is a coupler of the chip.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit
import chips
import circuitkit as ck

POST = "compiler-custom-chips"


def tax_circuits():
    qaoa = ck.qaoa_maxcut(16, examplekit.random_regular(16, 3, seed=16), p=1)
    bv = ck.bernstein_vazirani("1" * 15)
    adder = ck.draper_adder(8)
    chem = uccsd_lih()
    return [(qaoa, ck.circuit_fields(qaoa, "qaoa_3reg_16", "qaoa_3_regular")),
            (bv, ck.circuit_fields(bv, "bv_15", "bernstein_vazirani")),
            (adder, ck.circuit_fields(adder, "adder_8b", "draper_adder")),
            (chem, ck.circuit_fields(chem, "uccsd_lih_12", "uccsd"))]


def uccsd_lih():
    """UCCSD on the LiH STO-3G register (6 spatial orbitals, 4 electrons: 12 qubits, 92 excitations), Hartree-Fock
    reference, fixed amplitudes (seed 1)."""
    return ck.uccsd(6, 4, seed=1)


def scale_circuits(sizes):
    out = []
    for n in sizes:
        qc = ck.qaoa_maxcut(n, examplekit.random_regular(n, 3, seed=n), p=1)
        out.append((qc, ck.circuit_fields(qc, f"qaoa_3reg_{n}", "qaoa_3_regular")))
    return out


def compile_on(chip_name, chip, circuits, args, out, client, part):
    desc = chips.description(chip)
    n_qubits = desc["n_qubits"]
    qt = chips.qiskit_target(chip)
    n_couplers = len(chips.coupler_list(chip))
    out.add(kind="chip", post=POST, part=part, chip=chip_name, n_qubits=n_qubits, n_couplers=n_couplers,
            connected=ck.is_connected(n_qubits, chips.coupler_list(chip)), two_qubit_gate=desc["two_qubit_gate"],
            gates_1q=desc.get("gates_1q") or list(chips.GATES_1Q), target=desc if n_qubits <= 156 else None)
    for qc, circ in circuits:
        if qc.num_qubits > n_qubits:
            out.add(**ck.row(kind="skip", post=POST, **circ, coupler_set=chip_name), part=part,
                    note="circuit wider than the chip")
            continue
        extra = {"part": part, "chip": chip_name, "chip_qubits": n_qubits, "chip_couplers": n_couplers,
                 "chip_gate": desc["two_qubit_gate"]}
        for seed in args.seeds:
            if not args.skip_qiskit:
                for r in ck.qiskit_rows(qc, {chip_name: qt}, args.budgets, seed, post=POST, circuit=circ,
                                        max_trials=args.max_qiskit_trials, with_counts=True):
                    out.add(**r, **extra)
            if not args.skip_4colors:
                for budget in args.budgets:
                    _, r = ck.fourcolors_row(qc, post=POST, circuit=circ, target=chip, coupler_set=chip_name,
                                             seed=seed, budget_s=budget, client=client, with_counts=True,
                                             approximation_mode="exact")
                    out.add(**r, **extra)
                    print(f"{part} {chip_name} {circ['circuit_id']} seed {seed}: {r['two_qubit_gates']} "
                          f"{r['refusal_code'] or ''}", flush=True)


def main(argv=None):
    p = ck.compiler_parser(__doc__, budgets="10", sizes="16,64")
    p.add_argument("--part", default="tax,scale,design", help="tax, scale, design, comma-separated (default all)")
    p.add_argument("--designs", default=",".join(chips.DESIGN_CHIPS),
                   help="bundled design lattices for part design (default: the four of chips.DESIGN_CHIPS)")
    p.add_argument("--grids", type=ck.ints, default=list(chips.SCALE_GRIDS),
                   help="side lengths of the scaling grids (default 5,8,10,12)")
    args = p.parse_args(argv)
    out = examplekit.Results(os.path.join(args.out, "results.jsonl"))
    out.add(**ck.meta_row(POST, args))
    client = None if args.skip_4colors else examplekit.client()
    parts = args.part.split(",")
    if "tax" in parts:
        circuits = tax_circuits()
        for name, build in chips.TAX_CHIPS.items():
            compile_on(name, build(), circuits, args, out, client, "tax")
    if "scale" in parts:
        circuits = scale_circuits(args.sizes)
        for side in args.grids:
            compile_on(f"grid_{side}x{side}", chips.grid(side, side), circuits, args, out, client, "scale")
    if "design" in parts:
        chem = uccsd_lih()
        circuits = [(chem, ck.circuit_fields(chem, "uccsd_lih_12", "uccsd"))]
        for name in [d for d in args.designs.split(",") if d]:
            compile_on(name, chips.design(name), circuits, args, out, client, "design")
    print(f"wrote {out.path}")


if __name__ == "__main__":
    main()
