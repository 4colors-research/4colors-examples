# The API without Python

**Docs:** [this tutorial in the 4colors docs](https://docs.4colors-research.com/start/tutorials/http-without-python/?utm_source=github&utm_medium=examples&utm_campaign=http-without-python&utm_content=readme-top) · Products: 4colors API, 4colors Optimizer for QUBO, 4colors Quantum Compiler · Level: beginner

Run a QUBO job and a compile with `curl` and `jq`: the raw requests, the live event stream and the final record. Everything the Python package does is plain HTTP.

1. Start a free trial at [TRIAL SIGNUP URL] and copy your key.
2. Run it:

```bash
# needs curl and jq
export FOURCOLORS_API_KEY=...        # the key from step 1, or FOURCOLORS_KEY_FILE=/path/to/file
bash http_tutorial.sh
```

The compile's event stream can stay open a few minutes after `done`, until the check of the compiled circuit ends; the answer is already final at `done`.

The code between the `region docs:` lines in [`http_tutorial.sh`](http_tutorial.sh) is the docs page's code, word for word, so the two never differ (`tools/docs_sync.py` checks it). The explanation is on the docs page.

## Tests

```bash
python -m pytest -q tests
```

They check the script is valid bash and that it stops before any request when `FOURCOLORS_API_KEY` is not set. Nothing in the tests calls the service or compiles or solves anything: where a script runs end to end, it is fed fixed, hand-written answers from `tests/canned.py`, so the tests say nothing about answer quality.
