# Tutorials

The [docs tutorials](https://docs.4colors-research.com/tutorials/?utm_source=github&utm_medium=examples&utm_campaign=tutorials&utm_content=readme)
as runnable scripts, one folder per tutorial: `tutorials/<product>/<name>/`. The code between a script's
`# region docs:` lines is the docs page's code, word for word, so the page explains exactly what the script runs.

| Folder | Tutorials |
|---|---|
| [`qubo/`](qubo/) | [Max-Cut on a large graph](qubo/maxcut/) (also a notebook) · [Large QUBO models: upload once, reuse](qubo/large-models/) |
| [`compiler/`](compiler/) | [Time budgets for the compiler](compiler/time-budgets/) · [Compile for your own device](compiler/custom-target/) |
| [`platform/`](platform/) | [The API without Python](platform/http-without-python/) (curl and jq) |

A tutorial that needs several products equally goes in `joint/<name>/`.

A tutorial folder holds `README.md`, one script, `tests/` (no key, no network) and `example.json` (gallery
metadata), and may hold a notebook whose code cells are the script's regions.
