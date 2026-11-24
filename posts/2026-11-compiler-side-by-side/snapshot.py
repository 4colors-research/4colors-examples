"""Freeze one device snapshot so every arm and every run sees the same published calibration.

    python snapshot.py --backend fake:FakeKingston --out snap/      # a Qiskit fake backend (tests, dry runs)
    python snapshot.py --backend ibm:ibm_kingston --out snap/       # a live IBM backend's published data, read
                                                                    # with your own qiskit-ibm-runtime account

Writes into --out:
  qiskit_target.pkl   the backend's Qiskit Target as published (what Qiskit gets, as is)
  device.json         fourcolors.quantum.targets.from_qiskit(backend).to_json(): the description and calibration
                      the 4colors compiler gets (the same numbers; a target is a description, never a name)
  snapshot.json       backend name, UTC taken, sha256 of both files, client and qiskit versions
`load(dir)` reads them back and refuses files whose sha256 does not match.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickle


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def backend(spec: str):
    """`fake:<ClassName>` from qiskit_ibm_runtime.fake_provider, or `ibm:<name>` from QiskitRuntimeService."""
    kind, _, name = spec.partition(":")
    if kind == "fake":
        import qiskit_ibm_runtime.fake_provider as fp

        return getattr(fp, name)()
    if kind == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService

        return QiskitRuntimeService().backend(name)
    raise SystemExit(f"--backend must be fake:<Class> or ibm:<name>, not {spec!r}")


def save(be, out: str, spec: str) -> dict:
    from fourcolors.quantum import targets
    import qiskit

    os.makedirs(out, exist_ok=True)
    tpath, dpath = os.path.join(out, "qiskit_target.pkl"), os.path.join(out, "device.json")
    with open(tpath, "wb") as f:
        pickle.dump(be.target, f)
    with open(dpath, "w") as f:
        f.write(targets.from_qiskit(be).to_json())
    try:
        from importlib.metadata import version
        client = version("fourcolors")
    except Exception:
        client = None
    meta = {"backend": spec, "device": getattr(be, "name", spec), "taken_utc": dt.datetime.now(dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"), "qiskit_target_sha256": sha256(tpath), "device_sha256": sha256(dpath),
            "qiskit": qiskit.__version__, "fourcolors": client}
    with open(os.path.join(out, "snapshot.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load(snap: str):
    """(Qiskit Target, fourcolors DeviceTarget, meta) from a snapshot dir, sha256-checked."""
    from fourcolors.quantum import targets

    meta = json.load(open(os.path.join(snap, "snapshot.json")))
    tpath, dpath = os.path.join(snap, "qiskit_target.pkl"), os.path.join(snap, "device.json")
    for path, key in ((tpath, "qiskit_target_sha256"), (dpath, "device_sha256")):
        if sha256(path) != meta[key]:
            raise SystemExit(f"{path}: sha256 differs from snapshot.json; re-take the snapshot")
    with open(tpath, "rb") as f:
        qt = pickle.load(f)                      # our own file, written by save() above
    return qt, targets.DeviceTarget.from_json(open(dpath).read()), meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(save(backend(a.backend), a.out, a.backend), indent=1))


if __name__ == "__main__":
    main()
