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
| [Aircraft loading, quantum and classical](posts/2026-11-aircraft-loading/) | post | Optimizer for QUBO | intermediate |  |  |
| [Approximate QFT, measured like for like](posts/2026-11-compiler-approximate-qft/) | post | Optimizer for QUBO | intermediate |  |  |
| [The arithmetic inside Shor's algorithm, compiled: 100-qubit adders on a heavy-hex chip](posts/2026-11-compiler-shor-arithmetic/) | post | Optimizer for QUBO | intermediate |  |  |
| [Verified, approximate, unverified: reading a compiler's receipt](posts/2026-11-compiler-verification/) | post | Optimizer for QUBO | intermediate |  |  |
| [Split a power grid for parallel simulation: balanced partitioning with exact rows](posts/2026-11-domain-ene-2/) | post | Optimizer for QUBO | intermediate |  |  |
| [Rebalance every morning: warm-start today's portfolio from yesterday's](posts/2026-11-domain-fin-2/) | post | Optimizer for QUBO | intermediate |  |  |
| [The neutral-atom problem at a million atoms: maximum independent set on unit-disk graphs](posts/2026-11-domain-gr-2/) | post | Optimizer for QUBO | intermediate |  |  |
| [Traffic flow, eight years on: route assignment for 10,000 cars from OpenStreetMap](posts/2026-11-domain-log-3/) | post | Optimizer for QUBO | intermediate |  |  |
| [Pick the k best features: feature selection as a QUBO, with credit scoring as the example](posts/2026-11-domain-ml-1/) | post | Optimizer for QUBO | intermediate |  |  |
| [What "hardware-native" costs you](posts/2026-11-hardware-native-cost/) | post | Optimizer for QUBO | intermediate |  |  |
| [The paint shop problem: one car line, solved as a QUBO and compiled as QAOA](posts/2026-11-paint-shop-problem/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |
| [Measure fewer times: Pauli grouping as graph coloring, priced in compiled gates](posts/2026-11-pauli-grouping-coloring/) | post | Optimizer for QUBO | intermediate |  |  |
| [One problem, three encodings: what a constraint costs in gates](posts/2026-11-qaoa-encodings/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |
| [Solve it, then compile it: an honest reference for QAOA](posts/2026-11-qaoa-solve-then-compile/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |
| [A qLDPC code end to end: bound its distance, compile its syndrome round onto its own chip](posts/2026-11-qldpc-end-to-end/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |
| [Denoise a 1440p image with one QUBO](posts/2026-11-qubo-4k-denoise/) | post | Optimizer for QUBO | intermediate |  |  |
| [EV smart charging as a maximum independent set, from 100 to 100,000 tasks](posts/2026-11-qubo-ev-charging/) | post | Optimizer for QUBO | intermediate |  |  |
| [Stop tuning penalty weights: exact constraints for QUBO and CQM](posts/2026-11-qubo-exact-constraints/) | post | Optimizer for QUBO | intermediate |  |  |
| [Six million spins, one upload: a 3D spin glass on the standard plan](posts/2026-11-qubo-huge-spin-glass/) | post | Optimizer for QUBO | intermediate |  |  |
| [MIMO detection as a QUBO: from QuAMax to a classical solver, measured in milliseconds](posts/2026-11-qubo-mimo-detection/) | post | Optimizer for QUBO | intermediate |  |  |
| [Portfolio selection with a cardinality constraint](posts/2026-11-qubo-portfolio/) | post | Optimizer for QUBO | intermediate | finance |  |
| [Sudoku, N-queens and map coloring as constrained QUBOs](posts/2026-11-qubo-puzzles/) | post | Optimizer for QUBO | beginner | puzzles, graph-coloring |  |
| [RNA folding QUBOs, refereed: the model the papers posed, solved and checked exactly](posts/2026-11-qubo-rna-folding/) | post | Optimizer for QUBO | intermediate |  |  |
| [Stop paying for idle seconds: stop, stream and incumbents](posts/2026-11-qubo-stop-and-stream/) | post | Optimizer for QUBO | intermediate |  |  |
| [Ten graph problems as QUBOs, ten lines each](posts/2026-11-qubo-ten-graph-problems/) | post | Optimizer for QUBO | intermediate |  |  |
| [QUBO is the wrong encoding for TSP; here is the right tool](posts/2026-11-qubo-tsp-honestly/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |
| [Warm starts: re-solving tomorrow's problem from today's answer](posts/2026-11-qubo-warm-start/) | post | Optimizer for QUBO | intermediate |  |  |
| [The referee: re-scoring a published QPU benchmark with a real optimum](posts/2026-11-referee-qpu-benchmark/) | post | Optimizer for QUBO | intermediate |  |  |
| [Is this instance worth QPU time? A two-call pre-flight check](posts/2026-11-worth-qpu-time/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |

### 4colors Quantum Compiler

[Docs](https://docs.4colors-research.com/compiler/?utm_source=github&utm_medium=examples&utm_campaign=compiler&utm_content=gallery-section) · Needs: `fourcolors>=1.0.2`

| Example | Kind | Products | Level | Domains | Links |
|---|---|---|---|---|---|
| [Quickstart: compile a circuit](quickstart/compiler/) | quickstart | Quantum Compiler | beginner | getting-started, quantum-circuits | [docs](https://docs.4colors-research.com/start/quickstart-compiler/?utm_source=github&utm_medium=examples&utm_campaign=quickstart-compiler&utm_content=gallery) |
| [Compile for your own device](tutorials/compiler/custom-target/) | tutorial | Quantum Compiler | intermediate | quantum-circuits, qaoa | [docs](https://docs.4colors-research.com/start/tutorials/compiler-custom-target/?utm_source=github&utm_medium=examples&utm_campaign=compiler-custom-target&utm_content=gallery) |
| [Time budgets for the compiler](tutorials/compiler/time-budgets/) | tutorial | Quantum Compiler | beginner | quantum-circuits | [docs](https://docs.4colors-research.com/start/tutorials/compiler-time-budgets/?utm_source=github&utm_medium=examples&utm_campaign=compiler-time-budgets&utm_content=gallery) |
| [The API without Python](tutorials/platform/http-without-python/) | tutorial | API (HTTP), Optimizer for QUBO, Quantum Compiler | beginner | getting-started | [docs](https://docs.4colors-research.com/start/tutorials/http-without-python/?utm_source=github&utm_medium=examples&utm_campaign=http-without-python&utm_content=gallery) |
| [The biggest circuits: where the service says no, and why](posts/2026-11-compiler-biggest-circuits/) | post | Quantum Compiler | intermediate |  |  |
| [The coupler that wasn't there: what happens when your compiler isn't told](posts/2026-11-compiler-broken-couplers/) | post | Quantum Compiler | intermediate |  |  |
| [Trivial on paper, a routing nightmare on heavy-hex: Bernstein-Vazirani at 150 qubits](posts/2026-11-compiler-bv-routing/) | post | Quantum Compiler | intermediate |  |  |
| [Which chip shape suits chemistry, QAOA or adders? Custom targets and the topology tax](posts/2026-11-compiler-custom-chips/) | post | Quantum Compiler | intermediate |  |  |
| [Depth or gates? Choosing what the compiler minimizes](posts/2026-11-compiler-depth-or-gates/) | post | Quantum Compiler | intermediate |  |  |
| [The edge of the possible: where a compiled circuit stops working on a real chip](posts/2026-11-compiler-edge-of-possible/) | post | Quantum Compiler | intermediate |  |  |
| [Give it N seconds: a compiler you can watch improve](posts/2026-11-compiler-give-it-seconds/) | post | Quantum Compiler | intermediate |  |  |
| [Give it a second, and when longer pays: what more compile time buys, for 4colors and for Qiskit](posts/2026-11-compiler-longer-compiles/) | post | Quantum Compiler | intermediate |  |  |
| [Fewer gates, better answers? Chemistry, finance and materials observables under a stated noise model](posts/2026-11-compiler-noisy-observables/) | post | Quantum Compiler | intermediate |  |  |
| [Which circuits gain most: a compiler report card by family](posts/2026-11-compiler-report-card/) | post | Quantum Compiler | intermediate |  |  |
| [Look inside the output: three compilers, one circuit, gate by gate](posts/2026-11-compiler-side-by-side/) | post | Quantum Compiler | intermediate |  |  |
| [The paint shop problem: one car line, solved as a QUBO and compiled as QAOA](posts/2026-11-paint-shop-problem/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |
| [Compile early or compile late?](posts/2026-11-parametric-compile-late/) | post | Quantum Compiler | intermediate |  |  |
| [Compile once, bind a thousand times](posts/2026-11-parametric-compile-once/) | post | Quantum Compiler | intermediate |  |  |
| [The QAOA landscape, before and after noise](posts/2026-11-parametric-landscape/) | post | Quantum Compiler | intermediate |  |  |
| [Parametric circuits at 150 qubits: compile once, and check it at full size](posts/2026-11-parametric-large/) | post | Quantum Compiler | intermediate |  |  |
| [One problem, three encodings: what a constraint costs in gates](posts/2026-11-qaoa-encodings/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |
| [Solve it, then compile it: an honest reference for QAOA](posts/2026-11-qaoa-solve-then-compile/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |
| [The QAOA zoo, priced: what each variant costs in two-qubit gates](posts/2026-11-qaoa-zoo/) | post | Quantum Compiler | intermediate |  |  |
| [A qLDPC code end to end: bound its distance, compile its syndrome round onto its own chip](posts/2026-11-qldpc-end-to-end/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |
| [QUBO is the wrong encoding for TSP; here is the right tool](posts/2026-11-qubo-tsp-honestly/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |
| [Is this instance worth QPU time? A two-call pre-flight check](posts/2026-11-worth-qpu-time/) | post | Optimizer for QUBO, Quantum Compiler | intermediate |  |  |

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
