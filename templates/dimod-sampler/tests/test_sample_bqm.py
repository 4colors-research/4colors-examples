"""The template end to end, with the sampler's answer replaced by a fixed, hand-written SampleSet: nothing here
samples or solves. Real answers come only from the 4colors service."""
import runpy
from pathlib import Path

import dimod
import fourcolors

SCRIPT = Path(__file__).resolve().parents[1] / "sample_bqm.py"


def test_template_prints_the_best_sample(monkeypatch, capsys):
    seen = {}

    def canned_sample(self, bqm, time_limit_s=None, **kw):
        seen["bqm"], seen["time_limit_s"] = bqm, time_limit_s
        return dimod.SampleSet.from_samples({"a": 1, "b": 0, "c": 1}, "BINARY", energy=-2.0)

    # raising=True (the default) fails the test if the client no longer has this method.
    monkeypatch.setattr(fourcolors.FourColorsSampler, "sample", canned_sample)
    monkeypatch.setenv("FOURCOLORS_API_KEY", "canned")
    runpy.run_path(str(SCRIPT), run_name="__main__")
    out = capsys.readouterr().out.splitlines()
    assert seen["time_limit_s"] == 1 and seen["bqm"].num_variables == 3
    assert out[-1] == "{'a': 1, 'b': 0, 'c': 1} -2.0"
    assert seen["bqm"].energy({"a": 1, "b": 0, "c": 1}) == -2.0      # the model is the one the README shows
