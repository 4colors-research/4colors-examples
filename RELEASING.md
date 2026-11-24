# Releasing (maintainers)

Each product ships on its own schedule, so each product's examples are checked and tagged on their own.

## When a product releases

For product `<slug>` at version `<version>`:

1. Update the scripts for any changed calls. Change the code here, never in the docs.
2. Run every example of that product live, with a key, against the release:
   `python tools/live_check.py --products <slug> --dry-run` lists them and the most execution time they can bill
   (`billed_s_max` in each `example.json`); then run it with `--out <dir>`. Every example must exit 0.
3. Raise the pin in `requirements.txt` (and the product's `requires` in `products.json`) to the client release the
   examples were run with.
4. Copy the changed regions into the docs: `python tools/docs_sync.py write --docs <docs source>`, then
   `check` must report no drift. Docs changes go through the docs' own review.
5. Add a `CHANGELOG.md` entry under the product, naming the `fourcolors` client version and the engine version
   the service reported.
6. Tag the commit `<slug>/v<version>` (for example `qubo/v1.0.7`). A versioned docs page for that release shows
   the code at that tag, so older docs keep matching older code.

## When a post publishes

Export its code with `tools/export_post.py` (see CONTRIBUTING.md) on the day the post goes live, with
`--post-url`. The post links to its folder at that commit, so the link keeps working when the code moves on.

## Before every push

```bash
python tools/test_all.py                         # tests, layout, gallery, notebooks
python tools/leak_check.py . --require-denylist  # with the organization denylist
python tools/docs_sync.py check --docs <docs source>
```
