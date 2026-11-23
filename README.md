# 4colors examples: QUBO optimization and quantum circuit compilation in Python

Runnable examples, tutorials and blog-post companion code for the
**4colors Optimizer for QUBO** and the **4colors Quantum Compiler**, used through the public
[`fourcolors`](https://pypi.org/project/fourcolors/) Python package and the 4colors HTTP API.

<!-- Badges: owner to confirm each target before publishing. The Codespaces and CI badges work once the repo exists. -->
[![PyPI](https://img.shields.io/pypi/v/fourcolors?label=fourcolors)](https://pypi.org/project/fourcolors/)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/4colors-research/4colors-examples)
[![Offline tests](https://github.com/4colors-research/4colors-examples/actions/workflows/offline-tests.yml/badge.svg)](https://github.com/4colors-research/4colors-examples/actions/workflows/offline-tests.yml)
[![Docs](https://img.shields.io/badge/docs-docs.4colors--research.com-2a78d6)](https://docs.4colors-research.com/?utm_source=github&utm_medium=examples&utm_campaign=readme&utm_content=badge)

Every example is a small folder you can read in five minutes and run in one command. Every example also runs
**without an API key** in offline mode, so you can try the code, the models and the tests before signing up.

## Quickstart

```bash
git clone https://github.com/4colors-research/4colors-examples
cd 4colors-examples
pip install -r requirements.txt

# no key needed: a small local stand-in answers instead of the service
FOURCOLORS_OFFLINE=1 python quickstart/qubo_hello.py

# with a key from account.4colors-research.com
export FOURCOLORS_API_KEY=...            # or FOURCOLORS_KEY_FILE=/path/to/file
python quickstart/qubo_hello.py
python quickstart/compiler_hello.py
bash quickstart/http_hello.sh            # the same API with curl, no Python
```

Get an API key at [account.4colors-research.com](https://account.4colors-research.com/?utm_source=github&utm_medium=examples&utm_campaign=readme&utm_content=quickstart).
Docs: [docs.4colors-research.com](https://docs.4colors-research.com/?utm_source=github&utm_medium=examples&utm_campaign=readme&utm_content=quickstart).

## Gallery

All examples, quickstarts first. Browse by area in [domains/](domains/).

<!-- gallery:all:start -->
| Example | Kind | Product | Level | Domains | Post |
|---|---|---|---|---|---|
| [Quickstart: hello QUBO, hello compiler, hello HTTP](quickstart/) | quickstart | both | beginner | getting-started |  |
| [Notebook: Max-Cut on a large graph](notebooks/) | tutorial | Optimizer for QUBO | beginner | graphs, notebooks |  |
| [Max-Cut on a large graph](tutorials/qubo/maxcut/) | tutorial | Optimizer for QUBO | beginner | graphs |  |
<!-- gallery:all:end -->

## What is where

| Folder | What it holds |
|---|---|
| [`quickstart/`](quickstart/) | Hello-world for each product and for raw HTTP: under 30 lines each. |
| [`tutorials/`](tutorials/) | Step-by-step walk-throughs (`qubo/`, `compiler/`, `http/`), the runnable versions of the docs tutorials. |
| [`posts/`](posts/) | Companion code for blog posts, one folder per post: `posts/<yyyy-mm>-<slug>/`. |
| [`notebooks/`](notebooks/) | Jupyter notebooks (open in Colab or Codespaces). |
| [`domains/`](domains/) | The same examples indexed by area: finance, puzzles, graphs, quantum circuits, ... |
| [`templates/`](templates/) | Starting points: a dimod sampler, the Qiskit plugin, plain HTTP. |
| [`migrate/`](migrate/) | Moving existing code over: from Qiskit's `transpile`, from dimod samplers. |
| [`benchmarks/`](benchmarks/) | How to compare solvers fairly, and harnesses you can rerun yourself. |
| [`datasets/`](datasets/) | Loaders for public instance sets, with each set's license. No data is stored here. |
| [`shared/`](shared/) | `examplekit.py`: key handling, results files, offline mode, chart style. |
| [`tools/`](tools/) | Repository tooling: tests runner, gallery builder, leak check, post export, metrics. |

## Run the tests

```bash
pip install -r requirements-dev.txt
python tools/test_all.py        # every example's tests, offline: no key, no network
```

## Offline mode

`FOURCOLORS_OFFLINE=1` makes `fourcolors.qubo.solve` and `fourcolors.qubo.submit` answer locally (exact enumeration
for tiny models, a short simulated annealing above that). It exists so you can run the code and the tests anywhere.
Its answers and times are made up: they are not 4colors results and say nothing about the service.

## Contributing

Ideas, fixes and new examples are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). Look for issues labelled
`good first issue`. Questions go to [Discussions](https://github.com/4colors-research/4colors-examples/discussions).

## Citing

If an example helps your research, see [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

## License

Code: see [LICENSE](LICENSE). Text, figures and notebooks' prose: see [LICENSE-docs](LICENSE-docs).
Using the 4colors service is governed by the [4colors Terms of Service](https://4colors-research.com/terms?utm_source=github&utm_medium=examples&utm_campaign=readme&utm_content=license).
