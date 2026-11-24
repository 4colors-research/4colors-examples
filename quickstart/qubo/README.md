# Quickstart: solve a QUBO

**Docs:** [this quickstart in the 4colors docs](https://docs.4colors-research.com/start/quickstart-qubo/?utm_source=github&utm_medium=examples&utm_campaign=quickstart-qubo&utm_content=readme-top) · Product: 4colors Optimizer for QUBO · Level: beginner

Solve a 4-variable QUBO in one call, then submit a longer search and watch each better solution arrive.

1. Start a free trial at [TRIAL SIGNUP URL] and copy your key.
2. Run it:

```bash
pip install "fourcolors[optimizer]"
export FOURCOLORS_API_KEY=...        # the key from step 1, or FOURCOLORS_KEY_FILE=/path/to/file
python quickstart.py
```

The second part searches for up to 60 seconds. Press Ctrl-C to stop it early: the job stops on the service, you keep the best solution so far, and you are billed only for the time it ran.

The code between the `region docs:` lines in [`quickstart.py`](quickstart.py) is the docs page's code, word for word, so the two never differ (`tools/docs_sync.py` checks it). The explanation is on the docs page.

## Tests

```bash
python -m pytest -q tests
```

They run the script end to end and check what it prints, and that the model is the one the docs describe. Nothing in the tests calls the service or compiles or solves anything: where a script runs end to end, it is fed fixed, hand-written answers from `tests/canned.py`, so the tests say nothing about answer quality.
