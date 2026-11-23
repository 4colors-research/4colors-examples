# Blog-post companion code

One folder per post, named `<yyyy-mm>-<slug>/`, with:

- `README.md`: what you will learn, the model, the commands, and a link to the post (the numbers and charts live
  in the post, which names the release they were measured on);
- the scripts the post used, unchanged except for import paths;
- `tests/`: offline tests (no key, no network), including the scripts end to end against a local stand-in;
- `example.json`: gallery metadata.

Posts' code reaches this repository only through `tools/export_post.py`, which copies the reader files, leaves out
internal notes and results, and runs the leak check. See [CONTRIBUTING.md](../CONTRIBUTING.md#maintainers-exporting-a-post).
