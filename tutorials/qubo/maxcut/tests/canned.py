"""Fixed client responses for the end-to-end tests. Test code only: nothing outside tests/ imports this.

`canned_result` wraps an assignment and an objective that the test writes down by hand into the
`fourcolors.qubo.Result` shape the scripts read. Nothing here searches, optimizes, enumerates or even evaluates
the model it is handed: the model is ignored. These tests check what a script does with an answer (decode it,
verify it, print it), never how good an answer is. Real answers come only from the 4colors service.
"""
import runpy
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
    """The job shape the scripts use, replaying fixed values in order.

    `statuses`: what `job.status` returns on each read (the last one repeats). `bests`: what `job.best()` returns
    on each call (the last one repeats). `stop()` and `result()` return `final`."""

    def __init__(self, final, incumbents=(), statuses=("finished",), bests=(None,)):
        self.final, self.id = final, final.job.id
        self._incumbents = list(incumbents)
        self._statuses, self._bests = list(statuses), list(bests)
        self.stopped = False
        self.queue_position, self.estimated_start_at = 1, None
        self.billed_s = final.job.billed_s

    @property
    def status(self):
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    def best(self):
        return self._bests.pop(0) if len(self._bests) > 1 else self._bests[0]

    def incumbents(self, **kw):
        yield from self._incumbents
        yield self.final

    def result(self, *a, **kw):
        return self.final

    def stop(self):
        self.stopped = True
        return self.final


class CannedQubo:
    """Replaces `fourcolors.qubo.solve`, `submit` and `upload`: solve and submit return the next fixed response,
    in call order; upload returns `model_id` every time."""

    def __init__(self, responses, model_id="0" * 64):
        self.responses = list(responses)
        self.model_id = model_id
        self.calls = []

    def install(self, monkeypatch):
        monkeypatch.setattr(fourcolors.qubo, "solve", self.solve)
        monkeypatch.setattr(fourcolors.qubo, "submit", self.submit)
        monkeypatch.setattr(fourcolors.qubo, "upload", self.upload)
        return self

    def _next(self, kind, kw):
        self.calls.append({"call": kind, **{k: v for k, v in kw.items() if k in ("time_limit_s", "input_id")}})
        assert self.responses, "the script made more calls than the test has canned responses"
        return self.responses.pop(0)

    def solve(self, model=None, time_limit_s=None, **kw):
        r = self._next("solve", dict(kw, time_limit_s=time_limit_s))
        return r.result() if isinstance(r, CannedJob) else r

    def submit(self, model=None, time_limit_s=None, **kw):
        r = self._next("submit", dict(kw, time_limit_s=time_limit_s))
        return r if isinstance(r, CannedJob) else CannedJob(r)

    def upload(self, model=None, **kw):
        self.calls.append({"call": "upload"})
        return self.model_id


def run_script(path, monkeypatch):
    """Run an example script top to bottom, as `python <script>` would; returns its globals."""
    monkeypatch.setenv("FOURCOLORS_API_KEY", "canned")      # the script's key check passes; no call is made
    monkeypatch.delenv("FOURCOLORS_KEY_FILE", raising=False)
    return runpy.run_path(str(path), run_name="__main__")
