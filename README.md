# 4colors examples: QUBO optimization and quantum circuit compilation in Python

Runnable examples, tutorials and blog-post companion code for the
**4colors Optimizer for QUBO** and the **4colors Quantum Compiler**, used through the public
[`fourcolors`](https://pypi.org/project/fourcolors/) Python package and the 4colors HTTP API.

<!-- Badges: owner to confirm each target before publishing. The Codespaces and CI badges work once the repo exists. -->
[![PyPI](https://img.shields.io/pypi/v/fourcolors?label=fourcolors)](https://pypi.org/project/fourcolors/)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/4colors-research/4colors-examples)
[![Tests](https://github.com/4colors-research/4colors-examples/actions/workflows/tests.yml/badge.svg)](https://github.com/4colors-research/4colors-examples/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-docs.4colors--research.com-2a78d6)](https://docs.4colors-research.com/?utm_source=github&utm_medium=examples&utm_campaign=readme&utm_content=badge)

Every example is a small folder you can read in five minutes and run in one command. Every example sends its
model to the 4colors service, so it needs an API key.

## Quickstart

1. Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=readme&utm_content=readme-trial) and copy your key.
2. Run the examples:

```bash
git clone https://github.com/4colors-research/4colors-examples
cd 4colors-examples
pip install -r requirements.txt

export FOURCOLORS_API_KEY=...            # the key from step 1, or FOURCOLORS_KEY_FILE=/path/to/file
python quickstart/qubo/quickstart.py        # 4colors Optimizer for QUBO
python quickstart/compiler/quickstart.py    # 4colors Quantum Compiler
bash quickstart/platform/hello.sh           # the same API with curl, no Python
```

Docs: [docs.4colors-research.com](https://docs.4colors-research.com/?utm_source=github&utm_medium=examples&utm_campaign=readme&utm_content=quickstart).

## Gallery

Every example, by product: quickstarts first, then tutorials, then blog-post code. An example that uses several
products is listed under each. Browse by area in [domains/](domains/).

<!-- gallery:products:start -->
### 4colors Optimizer for QUBO

[Docs](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo&utm_content=gallery-section) · Needs: `fourcolors[optimizer]>=1.0.0`

| Example | Kind | Products | Level | Domains | Links |
|---|---|---|---|---|---|
| [Quickstart: solve a QUBO](quickstart/qubo/) | quickstart | Optimizer for QUBO | beginner | getting-started | [docs](https://docs.4colors-research.com/start/quickstart-qubo/?utm_source=github&utm_medium=examples&utm_campaign=quickstart-qubo&utm_content=gallery) |
| [The API without Python](tutorials/platform/http-without-python/) | tutorial | API (HTTP), Optimizer for QUBO, Quantum Compiler | beginner | getting-started | [docs](https://docs.4colors-research.com/start/tutorials/http-without-python/?utm_source=github&utm_medium=examples&utm_campaign=http-without-python&utm_content=gallery) |
| [Large QUBO models: upload once, reuse](tutorials/qubo/large-models/) | tutorial | Optimizer for QUBO | intermediate | graphs | [docs](https://docs.4colors-research.com/start/tutorials/qubo-large-models/?utm_source=github&utm_medium=examples&utm_campaign=qubo-large-models&utm_content=gallery) |
| [Max-Cut on a large graph](tutorials/qubo/maxcut/) | tutorial | Optimizer for QUBO | beginner | graphs | [docs](https://docs.4colors-research.com/start/tutorials/qubo-maxcut/?utm_source=github&utm_medium=examples&utm_campaign=qubo-maxcut&utm_content=gallery) · [notebook](tutorials/qubo/maxcut/maxcut.ipynb) |
| [Portfolio selection with a cardinality constraint](posts/2026-11-qubo-portfolio/) | post | Optimizer for QUBO | intermediate | finance |  |
| [Sudoku, N-queens and map coloring as constrained QUBOs](posts/2026-11-qubo-puzzles/) | post | Optimizer for QUBO | beginner | puzzles, graph-coloring |  |

### 4colors Quantum Compiler

[Docs](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler&utm_content=gallery-section) · Needs: `fourcolors>=1.0.2`

| Example | Kind | Products | Level | Domains | Links |
|---|---|---|---|---|---|
| [Quickstart: compile a circuit](quickstart/compiler/) | quickstart | Quantum Compiler | beginner | getting-started, quantum-circuits | [docs](https://docs.4colors-research.com/start/quickstart-compiler/?utm_source=github&utm_medium=examples&utm_campaign=quickstart-compiler&utm_content=gallery) |
| [Compile for your own device](tutorials/compiler/custom-target/) | tutorial | Quantum Compiler | intermediate | quantum-circuits, qaoa | [docs](https://docs.4colors-research.com/start/tutorials/compiler-custom-target/?utm_source=github&utm_medium=examples&utm_campaign=compiler-custom-target&utm_content=gallery) |
| [Time budgets for the compiler](tutorials/compiler/time-budgets/) | tutorial | Quantum Compiler | beginner | quantum-circuits | [docs](https://docs.4colors-research.com/start/tutorials/compiler-time-budgets/?utm_source=github&utm_medium=examples&utm_campaign=compiler-time-budgets&utm_content=gallery) |
| [The API without Python](tutorials/platform/http-without-python/) | tutorial | API (HTTP), Optimizer for QUBO, Quantum Compiler | beginner | getting-started | [docs](https://docs.4colors-research.com/start/tutorials/http-without-python/?utm_source=github&utm_medium=examples&utm_campaign=http-without-python&utm_content=gallery) |

### 4colors API

[Docs](https://docs.4colors-research.com/platform/?utm_source=github&utm_medium=examples&utm_campaign=platform&utm_content=gallery-section) · Needs: `curl and jq`

| Example | Kind | Products | Level | Domains | Links |
|---|---|---|---|---|---|
| [Quickstart: the API with curl](quickstart/platform/) | quickstart | API (HTTP) | beginner | getting-started | [docs](https://docs.4colors-research.com/start/tutorials/http-without-python/?utm_source=github&utm_medium=examples&utm_campaign=quickstart-platform&utm_content=gallery) |
| [The API without Python](tutorials/platform/http-without-python/) | tutorial | API (HTTP), Optimizer for QUBO, Quantum Compiler | beginner | getting-started | [docs](https://docs.4colors-research.com/start/tutorials/http-without-python/?utm_source=github&utm_medium=examples&utm_campaign=http-without-python&utm_content=gallery) |
<!-- gallery:products:end -->

## What is where

| Folder | What it holds |
|---|---|
| [`quickstart/`](quickstart/) | The docs quickstarts, one folder per product: `quickstart/<product>/`. |
| [`tutorials/`](tutorials/) | The docs tutorials as runnable scripts: `tutorials/<product>/<name>/` (`joint/` for tutorials that need several products). A notebook sits in its tutorial's folder. |
| [`posts/`](posts/) | Companion code for blog posts, one folder per post: `posts/<yyyy-mm>-<slug>/`. |
| [`domains/`](domains/) | The same examples indexed by area: finance, puzzles, graphs, quantum circuits, ... |
| [`templates/`](templates/) | Starting points: a dimod sampler, the Qiskit plugin, plain HTTP. |
| [`migrate/`](migrate/) | Moving existing code over: from Qiskit's `transpile`, from dimod samplers. |
| [`benchmarks/`](benchmarks/) | How to compare solvers fairly, and harnesses you can rerun yourself. |
| [`datasets/`](datasets/) | Loaders for public instance sets, with each set's license. No data is stored here. |
| [`shared/`](shared/) | `examplekit.py`: key handling, results files, chart style. |
| [`tools/`](tools/) | Repository tooling: tests runner, layout check, docs sync, gallery builder, leak check, post export, metrics. |
| [`products.json`](products.json) | The products this repository covers. Each has a short name (its slug: `qubo`, `compiler`, `platform`) used in folder names and release tags. |

The code in the docs pages is the code here: each docs snippet is a named region of a script in this repository
(`# region docs:<name>`), and `tools/docs_sync.py` checks that the docs, the notebooks and these scripts match.

## Run the tests

```bash
pip install -r requirements-dev.txt
python tools/test_all.py        # every example's tests, plus the layout, gallery and notebook checks
```

The tests check the code that runs on your side: model builders, decoders, validators and the repository tools.
Where a script is tested end to end, it is fed fixed, hand-written answers (each example's `tests/canned.py`), so
the tests need no key and no network. Nothing in this repository solves or compiles anything in place of the
service: every real answer comes from the 4colors service.

## Contributing

Ideas, fixes and new examples are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). Look for issues labelled
`good first issue`. Questions go to [Discussions](https://github.com/4colors-research/4colors-examples/discussions).

## Citing

If an example helps your research, see [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

## License

Code: [Apache License 2.0](LICENSE). Prose, figures and notebook text: [CC BY 4.0](LICENSE-docs), attributed as
"4colors examples, 4colors Ltd T/A 4colors Research, CC BY 4.0". See [NOTICE](NOTICE).
Using the 4colors service and the `fourcolors` package is governed by the
[4colors Terms of Service](https://4colors-research.com/terms?utm_source=github&utm_medium=examples&utm_campaign=readme&utm_content=license).
