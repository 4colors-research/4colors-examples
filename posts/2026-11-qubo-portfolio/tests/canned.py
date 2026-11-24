"""Canned results for the blog examples' unit tests (tests only; never imported by a post script).

These stubs do not solve or compile anything. They let a test drive a script's plumbing (rows written, fields
recorded, decoding and checking run, plots drawn) with a FIXED answer the test chooses:

- `CannedQubo` returns, for every call, a hard-coded assignment: the one the test passes (a list, or a function
  of the call that returns a list), else the warm start the script sent back unchanged, else all zeros. No search,
  no enumeration, no optimization. `None` from the function means "no answer" (`no_feasible_found`).
- `CannedCompiler` returns the input circuit unchanged (no routing, no synthesis, no transpile) or a circuit the
  test passes, with fixed labels. A test can make it raise a canned refusal.

Job ids start with "canned-" and the engine version is "canned": a canned row is never a result.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, List, Optional

CANNED_T = 0.1          # the one fixed trajectory time a canned result reports


# ---- QUBO -----------------------------------------------------------------------------------------

class CannedQubo:
    """Patch `fourcolors.qubo.solve/submit` to return a fixed assignment.

        canned = CannedQubo().install(monkeypatch)                 # all zeros (or the warm start)
        canned = CannedQubo(solution=[1, 0, 1]).install(monkeypatch)
        canned = CannedQubo(solution=lambda call: None)            # every job: no answer
    `canned.calls` lists what each call sent (n, m, time_limit_s, seed, warm_start, ...)."""

    def __init__(self, solution: Any = None):
        self.solution = solution
        self.calls: List[dict] = []

    def install(self, monkeypatch) -> "CannedQubo":
        import fourcolors.qubo

        monkeypatch.setattr(fourcolors.qubo, "solve", self.solve)
        monkeypatch.setattr(fourcolors.qubo, "submit", self.submit)
        return self

    def _model(self, model, linear, vartype, sense, offset):
        import fourcolors.qubo as fq

        if isinstance(model, fq.Model):
            return model
        return fq.to_model(model, linear, vartype=vartype, sense=sense or "min", offset=offset)

    def submit(self, model=None, time_limit_s: float = 10, *, linear=None, vartype=None, sense=None, offset=0,
               seed=None, warm_start=None, client=None, **kw):
        m = self._model(model, linear, vartype, sense, offset)
        call = {"n": m.n, "m": m.m, "time_limit_s": time_limit_s, "seed": seed,
                "warm_start": None if warm_start is None else list(warm_start), **kw}
        self.calls.append(call)
        if callable(self.solution):
            x = self.solution(call)
        elif self.solution is not None:
            x = list(self.solution)
        elif warm_start is not None:
            x = list(warm_start)
        else:
            x = [0] * m.n
        return CannedJob(m, x, time_limit_s, f"canned-{len(self.calls)}")

    def solve(self, model=None, time_limit_s: float = 10, *, on_incumbent=None, **kw):
        job = self.submit(model, time_limit_s, **kw)
        if on_incumbent is not None:
            for best in job.incumbents():
                on_incumbent(best)
        return job.result()


class CannedJob:
    """The `fourcolors.Job` shapes the posts use, around one fixed assignment."""

    def __init__(self, m, x: Optional[list], time_limit_s: float, job_id: str):
        self.m, self.x, self.time_limit_s, self.id = m, x, time_limit_s, job_id
        self._final = None
        self.status = "running"

    def _result(self, status: str):
        import fourcolors.qubo as fq

        x = self.x
        running = status == "running"
        obj = None if x is None else self.m.energy(x)
        traj = [] if x is None or running else [{"seq": 0, "t_search_s": CANNED_T, "objective": obj,
                                                  "objective_exact": True}]
        reason = {"code": "stopped"} if status == "stopped" else (
            {"code": "no_feasible_found"} if x is None and not running else None)
        cons = None
        if self.m.m and not running:
            ok = x is not None and bool(self.m.feasible(x))
            cons = {"m": self.m.m, "feasible": ok, "rows_checked": self.m.m, "rows_violated": 0 if ok else None}
        billed = None if running else (CANNED_T if status == "stopped" else float(self.time_limit_s))
        return fq.Result(status=status, objective=obj, solution=x, labels=list(self.m.labels or range(self.m.n)),
                         vartype=self.m.vartype, reason=reason, engine={"version": "canned"},
                         job=SimpleNamespace(id=self.id, billed_s=billed), trajectory=traj, constraints=cons)

    @property
    def billed_s(self):
        return None if self._final is None else self._final.billed_s

    def best(self):
        if self._final is not None:
            return self._final
        return None if self.x is None else self._result("running")

    def incumbents(self, **kw):
        if self.x is not None:
            yield self._result("running")
        yield self.result()

    def result(self, *a, **kw):
        if self._final is None:
            self._final = self._result("finished")
            self.status = "done"
        return self._final

    wait = result

    def stop(self, **kw):
        if self._final is None:
            self._final = self._result("stopped")
            self.status = "done"
        return self._final


# ---- compiler -------------------------------------------------------------------------------------

class CannedCompileResult:
    """The attributes the examples read from a CompileResult, around a fixed circuit."""

    def __init__(self, qc, *, seed=None, verify=None, label: Optional[str] = None, job_id: str = "canned-0",
                 status: str = "finished"):
        from qiskit import qasm3

        self._qc = qc
        self.status = status
        self.reason = None
        self.format = "qasm3"
        try:
            self.circuit = qasm3.dumps(qc)
        except Exception:                                   # a circuit qasm3 cannot print: the text is not used
            self.circuit = "canned"
        n2 = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
        self.metrics = {"two_qubit_gates": n2}
        self.mapping = {}
        self.seed = seed
        self.engine = {"version": "canned"}
        self.timings = {}
        self.billed_s = 0.0
        self.target = {"n_qubits": qc.num_qubits}
        self.job = SimpleNamespace(id=job_id, billed_s=0.0)
        lab = label or ("skipped" if verify == "skip" else "unverified")
        self.verification = SimpleNamespace(label=lab, distance=None, reason="canned test result", method="canned")

    @property
    def parameters(self):
        return [p.name for p in self._qc.parameters] or None

    @property
    def stopped(self):
        return self.status == "stopped"

    def to_qiskit(self):
        return self._qc

    def refresh(self):
        return self

    def wait_verification(self, timeout=None, poll=2.0):
        return self.verification


class CannedCompileJob:
    def __init__(self, result: CannedCompileResult, n_incumbents: int = 1):
        self._r = result
        self.id = result.job.id
        self.n_incumbents = n_incumbents
        self.stop_called = False

    def incumbents(self, **kw):
        for _ in range(self.n_incumbents):
            self._r.status = "running"
            yield self._r
        self._r.status = "finished"
        yield self._r

    def best(self):
        return self._r

    def result(self, *a, **kw):
        return self._r

    wait = result

    def stop(self, **kw):
        self.stop_called = True
        self._r.status = "stopped"
        return self._r


def _with_trivial_layout(qc):
    """The same circuit with the identity layout attached (virtual qubit i on physical qubit i), as the
    Qiskit plugin route expects a layout on a compiled circuit. Nothing is placed or routed."""
    if getattr(qc, "layout", None) is not None:
        return qc
    from qiskit.transpiler import Layout, TranspileLayout

    out = qc.copy()
    init = Layout({q: i for i, q in enumerate(out.qubits)})
    out._layout = TranspileLayout(initial_layout=init, input_qubit_mapping={q: i for i, q in enumerate(out.qubits)},
                                  final_layout=None, _input_qubit_count=out.num_qubits,
                                  _output_qubit_list=list(out.qubits))
    return out


class CannedCompiler:
    """Patch `fourcolors.quantum.compiler.compile/submit` to return the input circuit unchanged.

        canned = CannedCompiler().install(monkeypatch)
        canned = CannedCompiler(circuit=fixture_qc, label="verified")
        canned = CannedCompiler(refuse=lambda call: call["n_qubits"] > 156)   # a canned refusal
    `canned.calls` lists what each call sent (target, time_budget_s, options, n_qubits)."""

    def __init__(self, circuit=None, label: Optional[str] = None, refuse: Optional[Callable[[dict], bool]] = None,
                 n_incumbents: int = 1):
        self.circuit = circuit
        self.label = label
        self.refuse = refuse
        self.n_incumbents = n_incumbents
        self.calls: List[dict] = []

    def install(self, monkeypatch, examplekit=None) -> "CannedCompiler":
        from fourcolors.quantum import compiler

        monkeypatch.setattr(compiler, "compile", self.compile)
        monkeypatch.setattr(compiler, "submit", self.submit)
        if examplekit is not None:
            monkeypatch.setattr(examplekit, "client", lambda: None)
        return self

    def compile(self, circuit, target=None, time_budget_s=None, **kw):
        return self.submit(circuit, target, time_budget_s, **kw).result()

    def submit(self, circuit, target=None, time_budget_s=None, **kw):
        if isinstance(circuit, (str, bytes)):               # text input: parsed back, as a text input returns
            from qiskit import qasm2, qasm3

            s = circuit.decode() if isinstance(circuit, bytes) else circuit
            circuit = qasm3.loads(s) if s.lstrip().startswith("OPENQASM 3") else qasm2.loads(s)
        call = {"target": target, "time_budget_s": time_budget_s, "n_qubits": circuit.num_qubits, **kw}
        self.calls.append(call)
        if self.refuse is not None and self.refuse(call):
            from fourcolors.errors import InvalidRequest

            raise InvalidRequest("canned refusal", code="invalid_option", field="options.target")
        qc = _with_trivial_layout(self.circuit if self.circuit is not None else circuit)
        res = CannedCompileResult(qc, seed=kw.get("seed"), verify=kw.get("verify"), label=self.label,
                                  job_id=f"canned-{len(self.calls)}")
        return CannedCompileJob(res, self.n_incumbents)
