# Large QUBO models: upload once, reuse

**Docs:** [this tutorial in the 4colors docs](https://docs.4colors-research.com/start/tutorials/qubo-large-models/?utm_source=github&utm_medium=examples&utm_campaign=qubo-large-models&utm_content=readme-top) · Product: 4colors Optimizer for QUBO · Level: intermediate

Build a Max-Cut model with 50,000 nodes and 150,000 edges, upload it once, solve it by its id, and stop the job yourself when the answer has not improved for 20 seconds.

1. Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-large-models&utm_content=readme-trial) and copy your key.
2. Run it:

```bash
pip install "fourcolors[optimizer]"
export FOURCOLORS_API_KEY=...        # the key from step 1, or FOURCOLORS_KEY_FILE=/path/to/file
python large_models.py
```

Uploading the same model again returns the same id and sends nothing new. You are billed only for the time the job ran.

The code between the `region docs:` lines in [`large_models.py`](large_models.py) is the docs page's code, word for word, so the two never differ (`tools/docs_sync.py` checks it). The explanation is on the docs page.

## Tests

```bash
python -m pytest -q tests
```

They run the script end to end on canned answers: one upload, then a solve by id, and the stop rule (it stops after 20 seconds without improvement, with the clock simulated). Nothing in the tests calls the service or compiles or solves anything: where a script runs end to end, it is fed fixed, hand-written answers from `tests/canned.py`, so the tests say nothing about answer quality.
