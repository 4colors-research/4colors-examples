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
- **Runs in one command**, and **runs offline**: `FOURCOLORS_OFFLINE=1` (or the test suite) must work with no key.
- **Checks its own answer** in plain Python: decode the 0/1 solution or the compiled circuit and verify it.
- **Has tests** in `tests/` that need no key and no network.
- **Names versions**: rows record the `fourcolors` client version and the engine version the service reports.
- **US spelling**, plain sentences, no marketing claims.
- **No secrets**: keys come from `FOURCOLORS_API_KEY` or a file named by `FOURCOLORS_KEY_FILE`, never from code.

## Where things go

| You are adding | Put it in |
|---|---|
| a minimal call | `quickstart/` |
| a step-by-step walk-through | `tutorials/<qubo|compiler|http>/<name>/` |
| code for a published post | `posts/<yyyy-mm>-<slug>/` (maintainers, see below) |
| a notebook | `notebooks/` (runs offline top to bottom) |
| a starter for an ecosystem | `templates/` |

Each example folder has an `example.json` (see `tools/gallery.py` for the fields); run `python tools/gallery.py` to
update the gallery tables.

## Licensing of contributions

Owner decision pending (see LICENSE). Proposed: contributions are accepted under the repository's licenses, with a
Developer Certificate of Origin sign-off (`git commit -s`) rather than a CLA.

## Maintainers: exporting a post

Post code is developed internally and exported with:

```bash
python tools/export_post.py <internal post dir> --slug <slug> --month <yyyy-mm> --product qubo \
    --title "..." --summary "..." --domains finance --level beginner --post-url <published post URL> \
    --denylist <organization denylist kept outside this repository>
```

The tool copies reader files only, rewrites helper imports, removes internal notes, adds post and docs links, and
refuses to write anything if the leak check finds a problem. Organization terms live in a denylist outside the
repository; the check reports a denylist hit by list and line number only.
