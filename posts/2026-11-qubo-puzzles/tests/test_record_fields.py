"""Every QUBO row names the build it measured: examplekit.result_fields records the engine and client versions."""
import fourcolors

import examplekit
from canned import canned_result


def test_result_fields_record_the_engine_and_client_versions():
    r = canned_result([1, 0], -1.0, job_id="canned-7", billed_s=1.0, engine_version="canned-test",
                      trajectory=[{"t_search_s": 0.5, "objective": -1.0}])
    f = examplekit.result_fields(r)
    assert f["engine_version"] == "canned-test" and f["engine"] == {"version": "canned-test"}
    assert f["client_version"] == fourcolors.__version__
    assert (f["job_id"], f["billed_s"], f["solver_objective"]) == ("canned-7", 1.0, -1.0)
    assert f["trajectory"] == [[0.5, -1.0]]
    assert {"job_id", "status", "reason", "solver_objective", "billed_s", "trajectory"} <= set(f)
