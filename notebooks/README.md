# Notebooks

| Notebook | Product | Open |
|---|---|---|
| [`qubo-maxcut.ipynb`](qubo-maxcut.ipynb) | 4colors Optimizer for QUBO | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4colors-research/4colors-examples/blob/main/notebooks/qubo-maxcut.ipynb) |

Rules for notebooks in this repository:

- Each notebook runs top to bottom **offline** (no key) and with a key. Offline answers come from a small local
  stand-in and are labelled as such.
- Keys come from the environment or Colab secrets, never from a cell.
- Outputs are committed only from a real run with a key, against a named `fourcolors` and engine version, so a
  reader sees real results on GitHub without running anything. Until then cells are committed without outputs.
- `tests/test_notebooks.py` checks every notebook compiles and runs offline.
