"""Figures for the Pauli-grouping post, from the record only.

    python plot.py RUN/out --figs RUN/figs

Selection rule (owner 22 Nov: wins only, rule stated under each chart):
* groups.png: instances where the best valid 4colors grouping has fewer GC groups than both GC baselines;
* priced.png: instances where (4colors grouping, 4colors compile) has a lower modeled two-qubit cost per shot than
  (Sorted Insertion grouping, Qiskit compile), every group priced;
* shots.png: instances where the 4colors grouping's modeled shot factor is lower than Sorted Insertion's (GC).
Every other instance is listed in selection.md with the reason; nothing is dropped from the record.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(__import__("pathlib").Path(__file__).resolve().parents[2] / "shared")]

import examplekit


def summarize(rows):
    """Per instance: baseline group counts, the best valid 4colors grouping, and the priced costs."""
    inst = {}
    for r in rows:
        if r.get("kind") == "instance":
            inst[r["instance"]] = {"n_terms": r["n_terms"], "n_qubits": r["n_qubits"], "base": {}, "best": None,
                                   "priced": {}}
    for r in rows:
        s = inst.get(r.get("instance"))
        if s is None:
            continue
        if r.get("kind") == "grouping" and r.get("valid"):
            s["base"][(r["arm"], r["commute"])] = (r["n_groups"], r["modeled_shot_factor"])
        elif r.get("kind") == "solve" and r.get("valid"):
            if s["best"] is None or (r["n_groups"], r["modeled_shot_factor"]) < s["best"]:
                s["best"] = (r["n_groups"], r["modeled_shot_factor"])
        elif r.get("kind") == "priced":
            s["priced"][(r["grouping"], r["compiler"])] = r.get("modeled_2q_per_shot")
    return inst


def select(inst):
    """(groups wins, priced wins, shots wins, notes) under the rule in the module docstring."""
    g, pr, sh, notes = [], [], [], []
    for name, s in sorted(inst.items()):
        gc = [s["base"].get((a, "gc")) for a in ("sorted_insertion", "qiskit_group_commuting")]
        if s["best"] is None or None in gc:
            notes.append(f"{name}: no valid 4colors grouping or a baseline missing")
            continue
        if s["best"][0] < min(x[0] for x in gc):
            g.append(name)
        else:
            notes.append(f"{name}: groups not fewer than the best baseline ({s['best'][0]} vs {min(x[0] for x in gc)})")
        ours, theirs = s["priced"].get(("4colors-qubo", "4colors")), s["priced"].get(("sorted_insertion", "qiskit"))
        if ours is not None and theirs is not None and ours < theirs:
            pr.append(name)
        else:
            notes.append(f"{name}: priced cost not lower or unpriced ({ours} vs {theirs})")
        if s["best"][1] < gc[0][1]:
            sh.append(name)
        else:
            notes.append(f"{name}: modeled shot factor not lower than Sorted Insertion")
    return g, pr, sh, notes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", nargs="+")
    p.add_argument("--figs", required=True)
    a = p.parse_args(argv)
    plt = examplekit.plot_style()
    figs = Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    inst = summarize(examplekit.load_rows(a.run))
    g, pr, sh, notes = select(inst)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.27
    for k, (label, get) in enumerate((("Sorted Insertion", lambda s: s["base"][("sorted_insertion", "gc")][0]),
                                      ("Qiskit group_commuting", lambda s: s["base"][("qiskit_group_commuting", "gc")][0]),
                                      ("4colors coloring", lambda s: s["best"][0]))):
        ax.bar([i + (k - 1) * w for i in range(len(g))], [get(inst[n]) for n in g], w, label=label)
    ax.set_xticks(range(len(g)), [f"{n}\n{inst[n]['n_terms']} terms" for n in g])
    ax.set_ylabel("measurement groups (general commuting)")
    ax.legend()
    ax.set_title("Fewer groups at the same validity", loc="left")
    fig.text(0.01, 0.01, "Shown: instances where the 4colors grouping has fewer groups than both baselines.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figs / "groups.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar([i - 0.2 for i in range(len(pr))], [inst[n]["priced"][("sorted_insertion", "qiskit")] for n in pr], 0.4,
           label="Sorted Insertion + Qiskit O3")
    ax.bar([i + 0.2 for i in range(len(pr))], [inst[n]["priced"][("4colors-qubo", "4colors")] for n in pr], 0.4,
           label="4colors coloring + 4colors compiler")
    ax.set_xticks(range(len(pr)), pr)
    ax.set_ylabel("modeled two-qubit gates per shot")
    ax.legend()
    ax.set_title("Grouping and compiling, priced together (modeled)", loc="left")
    fig.text(0.01, 0.01, "Modeled cost, not a measurement. Shown: instances where the 4colors pipeline is lower.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figs / "priced.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar([i - 0.2 for i in range(len(sh))], [inst[n]["base"][("sorted_insertion", "gc")][1] for n in sh], 0.4,
           label="Sorted Insertion")
    ax.bar([i + 0.2 for i in range(len(sh))], [inst[n]["best"][1] for n in sh], 0.4, label="4colors coloring")
    ax.set_xticks(range(len(sh)), sh)
    ax.set_ylabel("modeled shots vs one group per term")
    ax.legend()
    ax.set_title("Fewer groups did not cost shots here (modeled)", loc="left")
    fig.text(0.01, 0.01, "Modeled (zero covariance). Shown: instances where the 4colors grouping is lower.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figs / "shots.png", dpi=150)

    (figs / "selection.md").write_text(
        "# Selection\n\n" + f"groups.png: {g}\n\npriced.png: {pr}\n\nshots.png: {sh}\n\nNot shown:\n\n"
        + "\n".join(f"- {x}" for x in notes) + "\n")
    print(f"figures in {figs}")


if __name__ == "__main__":
    main()
