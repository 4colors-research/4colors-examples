# Quickstart

Three hello-worlds, each under 30 lines.

| File | What it does | Needs a key? |
|---|---|---|
| [`qubo_hello.py`](qubo_hello.py) | Max-Cut of a 5-node ring with the 4colors Optimizer for QUBO, answer checked in plain Python | yes |
| [`compiler_hello.py`](compiler_hello.py) | A 5-qubit GHZ circuit compiled by the 4colors Quantum Compiler, with its verification label | yes |
| [`http_hello.sh`](http_hello.sh) | A QUBO job with `curl` and `jq`: submit, stream the events, read the record | yes |

1. Start a free trial at [TRIAL SIGNUP URL] and copy your key.
2. `export FOURCOLORS_API_KEY=...`, or put the key in a file and `export FOURCOLORS_KEY_FILE=/path/to/file` so it
   never appears in your shell history.
3. Run a quickstart:

```bash
pip install -r requirements.txt
python quickstart/qubo_hello.py
```

Each job bills its run time; these take a few seconds.
