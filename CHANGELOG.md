# Changelog

Changes to the examples, grouped by the product release they were checked against. Every entry names the
`fourcolors` client version and the engine versions the examples were run with.

## Unreleased

- Repository scaffold: quickstart, Max-Cut tutorial and notebook, two blog-post companions (puzzles, portfolio),
  tests, gallery, leak check, post export tool, metrics collector.
- Removed the stand-in solver that answered in place of the service, and its environment switch (23 Nov 2026).
  Every example needs a key; readers start with the free trial. End-to-end tests use fixed canned answers in
  `tests/`, and the leak check and the post export refuse stand-in solvers.
- Checked against: `fourcolors` 1.0.0 (unit tests only; no live run recorded yet).
