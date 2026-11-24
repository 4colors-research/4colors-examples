"""The HTTP quickstart needs a key and the service, so the test checks the script parses and refuses to start
when no key is set."""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "hello.sh"


def test_valid_bash():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_no_key_stops_before_any_request():
    env = {k: v for k, v in os.environ.items() if not k.startswith("FOURCOLORS_")}
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "FOURCOLORS_API_KEY" in r.stderr and "free trial" in r.stderr
