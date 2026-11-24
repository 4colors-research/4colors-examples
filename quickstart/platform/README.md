# Quickstart: the API with curl

**Docs:** [the related docs tutorial](https://docs.4colors-research.com/start/tutorials/http-without-python/?utm_source=github&utm_medium=examples&utm_campaign=quickstart-platform&utm_content=readme-top) · Product: 4colors API · Level: beginner

A tiny QUBO job over plain HTTP, with no Python: submit it, follow the live event stream until `done`, then read the job's record.

1. Start a free trial at [TRIAL SIGNUP URL] and copy your key.
2. Run it:

```bash
# needs curl and jq
export FOURCOLORS_API_KEY=...        # the key from step 1, or FOURCOLORS_KEY_FILE=/path/to/file
bash hello.sh
```

For more, including a compile over HTTP, see [The API without Python](../../tutorials/platform/http-without-python/).

## Tests

```bash
python -m pytest -q tests
```

They check the script is valid bash and that it stops before any request when `FOURCOLORS_API_KEY` is not set. Nothing in the tests calls the service or compiles or solves anything: where a script runs end to end, it is fed fixed, hand-written answers from `tests/canned.py`, so the tests say nothing about answer quality.
