# Quickstart

One folder per product, each the docs quickstart as a script you can run.

| Folder | What it does | Docs |
|---|---|---|
| [`qubo/`](qubo/) | Solve a small QUBO in one call, then watch a longer search stream better solutions | [Quickstart: solve a QUBO](https://docs.4colors-research.com/start/quickstart-qubo/?utm_source=github&utm_medium=examples&utm_campaign=quickstart&utm_content=table) |
| [`compiler/`](compiler/) | Compile a 5-qubit GHZ circuit and read its verification label | [Quickstart: compile a circuit](https://docs.4colors-research.com/start/quickstart-compiler/?utm_source=github&utm_medium=examples&utm_campaign=quickstart&utm_content=table) |
| [`platform/`](platform/) | A QUBO job with `curl` and `jq`: submit, stream the events, read the record | [The API without Python](https://docs.4colors-research.com/start/tutorials/http-without-python/?utm_source=github&utm_medium=examples&utm_campaign=quickstart&utm_content=table) |

1. Start a free trial at [TRIAL SIGNUP URL] and copy your key.
2. `export FOURCOLORS_API_KEY=...`, or put the key in a file and `export FOURCOLORS_KEY_FILE=/path/to/file` so it
   never appears in your shell history.
3. Run a quickstart:

```bash
pip install -r requirements.txt
python quickstart/qubo/quickstart.py
```

Each job bills its run time; these take seconds to a minute, and Ctrl-C stops a job early.
