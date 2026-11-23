# Blog-post companion code

One folder per post, named `<yyyy-mm>-<slug>/`, with:

- `README.md`: what you will learn, the model, the commands, and a link to the post (the numbers and charts live
  in the post, which names the release they were measured on);
- the scripts the post used, unchanged except for import paths;
- `tests/`: tests that need no key and no network: the models, decoders and checks, and the scripts end to end
  on fixed, hand-written answers (`tests/canned.py`);
- `example.json`: gallery metadata.

Posts' code reaches this repository only through `tools/export_post.py`, which copies the reader files, leaves out
internal notes and results, and runs the leak check. See [CONTRIBUTING.md](../CONTRIBUTING.md#maintainers-exporting-a-post).
