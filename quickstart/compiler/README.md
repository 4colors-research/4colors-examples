# Quickstart: compile a circuit

**Docs:** [this quickstart in the 4colors docs](https://docs.4colors-research.com/quickstart-compiler/?utm_source=github&utm_medium=examples&utm_campaign=quickstart-compiler&utm_content=readme-top) · Product: 4colors Quantum Compiler · Level: beginner

Compile a 5-qubit GHZ circuit for a 156-qubit device design that ships with the package, read its verification label, then watch better circuits arrive during a 60-second budget.

1. Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=quickstart-compiler&utm_content=readme-trial) and copy your key.
2. Run it:

```bash
pip install fourcolors
export FOURCOLORS_API_KEY=...        # the key from step 1, or FOURCOLORS_KEY_FILE=/path/to/file
python quickstart.py
```

Needs `fourcolors` 1.0.2 or later: targets are descriptions of a device, never names. For your own device, see [Compile for your own device](../../tutorials/compiler/custom-target/).

The code between the `region docs:` lines in [`quickstart.py`](quickstart.py) is the docs page's code, word for word, so the two never differ (`tools/docs_sync.py` checks it). The explanation is on the docs page.

## Tests

```bash
python -m pytest -q tests
```

They run the script end to end and check what it sends (a target description, not a name, and the budgets) and what it prints. Nothing in the tests calls the service or compiles or solves anything: where a script runs end to end, it is fed fixed, hand-written answers from `tests/canned.py`, so the tests say nothing about answer quality.
