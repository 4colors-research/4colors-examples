# Changelog

Changes to the examples, grouped by the product release they were checked against. Every entry names the
`fourcolors` client version and the engine versions the examples were run with.

## Unreleased

- Repository scaffold: quickstart, Max-Cut tutorial and notebook, two blog-post companions (puzzles, portfolio),
  tests, gallery, leak check, post export tool, metrics collector.
- Removed the stand-in solver that answered in place of the service, and its environment switch (23 Nov 2026).
  Every example needs a key; readers start with the free trial. End-to-end tests use fixed canned answers in
  `tests/`, and the leak check and the post export refuse stand-in solvers.
- Layout by product: `products.json` lists the products (`qubo`, `compiler`, `platform`); quickstarts are
  `quickstart/<product>/`, tutorials `tutorials/<product>/<name>/`, and `example.json` takes a `products` list.
  The gallery has one section per product. `tools/validate.py` checks the layout.
- All docs quickstarts and tutorials as runnable scripts: QUBO (solve, Max-Cut, large models), compiler
  (quickstart, time budgets, custom target), API (curl and jq). Their docs code is kept in named regions;
  `tools/docs_sync.py` checks the docs pages, the notebooks and the READMEs against them.
- The Max-Cut notebook moved into its tutorial's folder; its code cells are the script's regions.
- Compiler examples need `fourcolors` 1.0.2 or later (targets are descriptions, never names).
- Checked against: `fourcolors` 1.0.0 (QUBO) and 1.0.2 (compiler), unit tests only; no live run yet.
