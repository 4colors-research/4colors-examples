"""Fixed compiler responses for the end-to-end tests. Test code only: nothing outside tests/ imports this.

`canned_compile` wraps metrics and a label that the test writes down by hand into the result shape the scripts
read. Nothing here places, routes, rewrites or checks a circuit: the circuit and the target are only recorded,
so a test can check what the script sent. Real compiles come only from the 4colors service.
"""
import runpy
from types import SimpleNamespace

from fourcolors.quantum import compiler


def canned_compile(two_qubit_gates, depth, *, label="verified", billed_s=1.0, circuit="OPENQASM 3.0;\n",
                   layout=None):
    return SimpleNamespace(metrics={"two_qubit_gates": two_qubit_gates, "depth": depth},
                           mapping={"initial_layout": layout or [0, 1, 2]}, circuit=circuit, billed_s=billed_s,
                           verification=SimpleNamespace(label=label))


class CannedCompileJob:
    def __init__(self, final, incumbents=()):
        self.final, self._incumbents = final, list(incumbents)

    def incumbents(self, **kw):
        yield from self._incumbents
        yield self.final

    def result(self, *a, **kw):
        return self.final

    def stop(self):
        return self.final


class CannedCompiler:
    """Replaces `compiler.compile` and `compiler.submit`: each call returns the next fixed response, in call
    order, and records the circuit, target and budget it was given."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def install(self, monkeypatch):
        monkeypatch.setattr(compiler, "compile", self.compile)
        monkeypatch.setattr(compiler, "submit", self.submit)
        return self

    def _next(self, kind, circuit, target, time_budget_s, kw):
        self.calls.append({"call": kind, "circuit": circuit, "target": target, "time_budget_s": time_budget_s,
                           **kw})
        assert self.responses, "the script made more calls than the test has canned responses"
        return self.responses.pop(0)

    def compile(self, circuit, target, time_budget_s=None, **kw):
        r = self._next("compile", circuit, target, time_budget_s, kw)
        return r.result() if isinstance(r, CannedCompileJob) else r

    def submit(self, circuit, target, time_budget_s=None, **kw):
        r = self._next("submit", circuit, target, time_budget_s, kw)
        return r if isinstance(r, CannedCompileJob) else CannedCompileJob(r)


def run_script(path, monkeypatch):
    """Run an example script top to bottom, as `python <script>` would; returns its globals."""
    monkeypatch.setenv("FOURCOLORS_API_KEY", "canned")      # the script's key check passes; no call is made
    monkeypatch.delenv("FOURCOLORS_KEY_FILE", raising=False)
    return runpy.run_path(str(path), run_name="__main__")


def gate_count(qasm, gate):
    return sum(1 for ln in qasm.splitlines() if ln.startswith(gate + " ") or ln.startswith(gate + "("))
