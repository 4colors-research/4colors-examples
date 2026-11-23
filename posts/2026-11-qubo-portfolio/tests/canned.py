"""Fixed client responses for the end-to-end tests. Test code only: nothing outside tests/ imports this.

`canned_result` wraps an assignment and an objective that the test writes down by hand into the
`fourcolors.qubo.Result` shape the scripts read. Nothing here searches, optimizes, enumerates or even evaluates
the model it is handed: the model is ignored. These tests check what a script does with an answer (decode it,
verify it, record it), never how good an answer is. Real answers come only from the 4colors service.
"""
from types import SimpleNamespace

import fourcolors.qubo


def canned_result(solution, objective, *, status="finished", reason=None, constraints=None, job_id="canned-1",
                  billed_s=1.0, engine_version="canned-test", trajectory=()):
    """A `fourcolors.qubo.Result` holding exactly the values given."""
    sol = None if solution is None else list(solution)
    return fourcolors.qubo.Result(status=status, objective=objective, solution=sol,
                                  labels=list(range(len(sol or []))), vartype="BINARY",
                                  engine={"version": engine_version}, reason=reason,
                                  job=SimpleNamespace(id=job_id, billed_s=billed_s),
                                  trajectory=list(trajectory), constraints=constraints)


class CannedJob:
    """The job shape the scripts use (incumbents(), result(), id), replaying fixed results in order."""

    def __init__(self, final, incumbents=()):
        self.final, self.id = final, final.job.id
        self._incumbents = list(incumbents)

    def incumbents(self, **kw):
        yield from self._incumbents
        yield self.final

    def result(self, *a, **kw):
        return self.final


class CannedQubo:
    """Replaces `fourcolors.qubo.solve` and `submit`: each call returns the next fixed response, in call order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def install(self, monkeypatch):
        monkeypatch.setattr(fourcolors.qubo, "solve", self.solve)
        monkeypatch.setattr(fourcolors.qubo, "submit", self.submit)
        return self

    def _next(self, model, kw):
        self.calls.append({"n": getattr(model, "n", None), "seed": kw.get("seed")})
        assert self.responses, "the script made more calls than the test has canned responses"
        return self.responses.pop(0)

    def solve(self, model=None, time_limit_s=None, **kw):
        r = self._next(model, kw)
        return r.result() if isinstance(r, CannedJob) else r

    def submit(self, model=None, time_limit_s=None, **kw):
        r = self._next(model, kw)
        return r if isinstance(r, CannedJob) else CannedJob(r)
