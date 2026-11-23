"""The quickstarts run offline (QUBO) or refuse clearly without the service (compiler)."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ENV = {k: v for k, v in os.environ.items() if not k.startswith("FOURCOLORS_")}


def run(script, **env):
    return subprocess.run([sys.executable, str(HERE / script)], env={**ENV, **env}, capture_output=True, text=True,
                          timeout=120)


def test_qubo_hello_offline():
    r = run("qubo_hello.py", FOURCOLORS_OFFLINE="1")
    assert r.returncode == 0, r.stderr
    assert "cut 4 of 5 edges" in r.stdout and "engine offline" in r.stdout


def test_compiler_hello_refuses_offline():
    r = run("compiler_hello.py", FOURCOLORS_OFFLINE="1")
    assert r.returncode != 0 and "needs the service" in r.stderr


def test_http_hello_is_valid_bash():
    assert subprocess.run(["bash", "-n", str(HERE / "http_hello.sh")]).returncode == 0
