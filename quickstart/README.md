# Quickstart

Three hello-worlds, each under 30 lines.

| File | What it does | Needs a key? |
|---|---|---|
| [`qubo_hello.py`](qubo_hello.py) | Max-Cut of a 5-node ring with the 4colors Optimizer for QUBO, answer checked in plain Python | no, with `FOURCOLORS_OFFLINE=1` |
| [`compiler_hello.py`](compiler_hello.py) | A 5-qubit GHZ circuit compiled by the 4colors Quantum Compiler, with its verification label | yes |
| [`http_hello.sh`](http_hello.sh) | A QUBO job with `curl` and `jq`: submit, stream the events, read the record | yes |

```bash
pip install -r requirements.txt
FOURCOLORS_OFFLINE=1 python quickstart/qubo_hello.py
```

With a key ([get one](https://account.4colors-research.com/?utm_source=github&utm_medium=examples&utm_campaign=quickstart&utm_content=readme)):
`export FOURCOLORS_API_KEY=...`, or put it in a file and `export FOURCOLORS_KEY_FILE=/path/to/file` so it never
appears in your shell history. Each job bills its run time; these take a few seconds.
