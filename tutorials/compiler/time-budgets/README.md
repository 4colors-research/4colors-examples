# Time budgets for the compiler

**Docs:** [this tutorial in the 4colors docs](https://docs.4colors-research.com/tutorials/compiler-time-budgets/?utm_source=github&utm_medium=examples&utm_campaign=compiler-time-budgets&utm_content=readme-top) · Product: 4colors Quantum Compiler · Level: beginner

Compile a random 25-qubit circuit with 250 two-qubit gates twice: first with the budget the compiler picks, then with a 20-second budget, watching each better circuit arrive.

1. Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=compiler-time-budgets&utm_content=readme-trial) and copy your key.
2. Run it:

```bash
pip install fourcolors
export FOURCOLORS_API_KEY=...        # the key from step 1, or FOURCOLORS_KEY_FILE=/path/to/file
python time_budgets.py
```

Needs `fourcolors` 1.0.2 or later. The target is a 156-qubit heavy-hex design that ships with the package.

The code between the `region docs:` lines in [`time_budgets.py`](time_budgets.py) is the docs page's code, word for word, so the two never differ (`tools/docs_sync.py` checks it). The explanation is on the docs page.

## Tests

```bash
python -m pytest -q tests
```

They check the target description and the circuit, and run the script end to end on canned answers. Nothing in the tests calls the service or compiles or solves anything: where a script runs end to end, it is fed fixed, hand-written answers from `tests/canned.py`, so the tests say nothing about answer quality.
