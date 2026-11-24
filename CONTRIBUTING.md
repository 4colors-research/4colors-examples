# Contributing

Thank you for helping. Fixes, new examples, translations of an example to another domain, and reports of anything
that does not run are all welcome.

## Quick path

1. Open an issue first for a new example (template: "Example request"), so we can agree on scope.
2. Fork, create a branch, and add your example in the right place (see below).
3. `pip install -r requirements-dev.txt`, then `python tools/test_all.py` and `python tools/leak_check.py .`.
4. Open a pull request; the template has a short checklist.

New here? Look for issues labelled `good first issue`.

## What makes a good example

- **Small and focused**: one idea, one folder, readable in five minutes.
- **Runs in one command** with a key (readers get one from the free trial). Never add a local solver, compiler or
  simulator that answers in place of the 4colors service, not even as a fallback or a demo.
- **Checks its own answer** in plain Python: decode the 0/1 solution or the compiled circuit and verify it.
- **Has tests** in `tests/` that need no key and no network: they test the code on the reader's side (models,
  decoders, validators). An end-to-end test feeds the script a fixed, hand-written answer from `tests/canned.py`;
  a canned answer never searches or optimizes, and it never lives in `shared/`.
- **Names versions**: rows record the `fourcolors` client version and the engine version the service reports.
- **US spelling**, plain sentences, no marketing claims.
- **No secrets**: keys come from `FOURCOLORS_API_KEY` or a file named by `FOURCOLORS_KEY_FILE`, never from code.

## Where things go

Every example belongs to one or more products, named by their slug in [`products.json`](products.json) (`qubo`,
`compiler`, `platform`).

| You are adding | Put it in |
|---|---|
| a product's first, smallest call | `quickstart/<product>/` (one per product; mirrors the docs quickstart) |
| a step-by-step walk-through | `tutorials/<product>/<name>/`, where `<name>` is the slug without `<product>-` (slug `qubo-maxcut` → `tutorials/qubo/maxcut/`) |
| a walk-through that needs several products equally | `tutorials/joint/<slug>/` |
| a notebook | in the folder of the tutorial or post it belongs to; tie each code cell to a script region (see below) |
| code for a published post | `posts/<yyyy-mm>-<slug>/` (maintainers, see below) |
| a starter for an ecosystem | `templates/<ecosystem>/` |

Each example folder has an `example.json`:

```json
{"kind": "tutorial", "slug": "qubo-maxcut", "title": "...", "summary": "...", "products": ["qubo"],
 "domains": ["graphs"], "level": "beginner", "docs_url": "https://docs.4colors-research.com/start/tutorials/qubo-maxcut/",
 "post_url": null, "requires": "fourcolors[optimizer]>=1.0.0", "notebook": "maxcut.ipynb"}
```

`python tools/validate.py` checks these rules, and `python tools/gallery.py` updates the gallery tables.
Both run as part of `python tools/test_all.py`.

## Code that the docs show

A docs page and this repository share one copy of each snippet. In the script, the snippet is a named region:

```python
# region docs:build
...
# endregion
```

The script runs its regions top to bottom, so a tutorial's regions are its docs blocks in reading order. Code the
docs do not show (the key setup, a self-check) goes outside the regions. A notebook cell shows a region when its
metadata says `{"example": {"region": "tutorials/qubo/maxcut/maxcut.py#build"}}`.

```bash
python tools/docs_sync.py check                      # notebooks and READMEs here match the scripts
python tools/docs_sync.py check --docs <docs source> # and the docs pages too
python tools/docs_sync.py write --docs <docs source> # copy changed regions into the docs and notebooks
```

Change the code here, never in the docs; `write` updates the docs pages from the scripts.

## Adding a product

1. Add a row to `products.json`: slug, name, short name, status (`live` or `preview`), docs URL, requirement.
   Add it on the day the product is public. Until then its name appears nowhere in this repository.
2. Add `quickstart/<slug>/` and, as they come, `tutorials/<slug>/<name>/`.
3. Run `python tools/test_all.py`: the layout check fails until the product has a quickstart.

## Licensing of contributions

Owner decision pending (see LICENSE). Proposed: contributions are accepted under the repository's licenses, with a
Developer Certificate of Origin sign-off (`git commit -s`) rather than a CLA.

## Maintainers: exporting a post

Post code is developed internally and exported with:

```bash
python tools/export_post.py <internal post dir> --slug <slug> --month <yyyy-mm> --products qubo \
    --title "..." --summary "..." --domains finance --level beginner --post-url <published post URL> \
    --denylist <organization denylist kept outside this repository>
```

The tool copies reader files only, rewrites helper imports, removes internal notes, adds post and docs links, and
refuses to write anything if the leak check finds a problem. Organization terms live in a denylist outside the
repository; the check reports a denylist hit by list and line number only.
