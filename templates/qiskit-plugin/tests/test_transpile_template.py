"""The Qiskit template: the client registers its five compile stages with Qiskit under the name "fourcolors", and
the template runs end to end with the compile replaced by fixed, hand-made answers (nothing compiles here)."""
import runpy
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace

import pytest
import qiskit
from fourcolors.quantum import compiler

SCRIPT = Path(__file__).resolve().parents[1] / "transpile_with_fourcolors.py"
STAGES = ["init", "layout", "routing", "translation", "optimization"]


@pytest.mark.parametrize("stage", STAGES)
def test_stage_plugin_is_registered(stage):
    names = {ep.name for ep in entry_points(group=f"qiskit.transpiler.{stage}")}
    assert "fourcolors" in names


def test_template_runs(monkeypatch, capsys):
    seen = {}

    def canned_transpile(qc, backend, **kw):
        seen["transpile"] = kw
        return qc

    class CannedPM:
        property_set = {"fourcolors_result": SimpleNamespace(verification=SimpleNamespace(label="verified"))}

        def run(self, qc):
            return qc

    monkeypatch.setattr(qiskit, "transpile", canned_transpile)
    # raising=True (the default) fails the test if the client no longer has these functions.
    monkeypatch.setattr(compiler, "set_qiskit_options", lambda **kw: seen.setdefault("options", kw))
    monkeypatch.setattr(compiler, "generate_pass_manager",
                        lambda backend, **kw: seen.setdefault("pm", kw) and CannedPM())
    monkeypatch.setenv("FOURCOLORS_API_KEY", "canned")
    runpy.run_path(str(SCRIPT), run_name="__main__")
    assert {seen["transpile"][f"{s}_method"] for s in STAGES} == {"fourcolors"}
    assert seen["options"] == {"time_budget_s": 10} and seen["pm"] == {"time_budget_s": 10}
    assert capsys.readouterr().out.strip().endswith("verified")
