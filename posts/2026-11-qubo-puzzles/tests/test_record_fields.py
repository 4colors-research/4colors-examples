"""Every QUBO row names the build it measured: examplekit.result_fields records the engine and client versions."""
import fourcolors
import fourcolors.qubo

import examplekit


def test_result_fields_record_the_engine_and_client_versions(canned):
    r = fourcolors.qubo.solve({(0, 1): 2.0}, 1, linear=[-1.0, -1.0], seed=1)
    f = examplekit.result_fields(r)
    assert f["engine_version"] == "canned" and f["engine"] == {"version": "canned"}
    assert f["client_version"] == fourcolors.__version__
    assert {"job_id", "status", "reason", "solver_objective", "billed_s", "trajectory"} <= set(f)
