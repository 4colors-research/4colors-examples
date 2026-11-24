# Compile for your own device

**Docs:** [this tutorial in the 4colors docs](https://docs.4colors-research.com/start/tutorials/compiler-custom-target/?utm_source=github&utm_medium=examples&utm_campaign=compiler-custom-target&utm_content=readme-top) · Product: 4colors Quantum Compiler · Level: intermediate

Describe a 12 x 12 grid device yourself and compile a 100-qubit QAOA circuit for it and for a heavy-hex design, then compare the two-qubit gate counts.

1. Start a free trial at [TRIAL SIGNUP URL] and copy your key.
2. Run it:

```bash
pip install fourcolors
export FOURCOLORS_API_KEY=...        # the key from step 1, or FOURCOLORS_KEY_FILE=/path/to/file
python custom_target.py
```

Needs `fourcolors` 1.0.2 or later. The docs page also shows how to add your device's calibration and how to read a target from your provider's backend object; those need your device's data, so they are not in this script.

The code between the `region docs:` lines in [`custom_target.py`](custom_target.py) is the docs page's code, word for word, so the two never differ (`tools/docs_sync.py` checks it). The explanation is on the docs page.

## Tests

```bash
python -m pytest -q tests
```

They check the graph is 3-regular, the gate counts of the QAOA circuit, that the client accepts the grid description (checked locally, nothing is sent), and the script end to end on canned answers. Nothing in the tests calls the service or compiles or solves anything: where a script runs end to end, it is fed fixed, hand-written answers from `tests/canned.py`, so the tests say nothing about answer quality.
