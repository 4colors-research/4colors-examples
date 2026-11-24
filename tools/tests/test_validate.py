import json

import validate

PRODUCTS = {"products": [
    {"slug": "qubo", "name": "N", "short": "n", "status": "live", "docs_url": "https://docs.4colors-research.com/q/",
     "requires": "x"},
    {"slug": "compiler", "name": "N", "short": "n", "status": "live",
     "docs_url": "https://docs.4colors-research.com/c/", "requires": "x"},
]}
DOCS = "https://docs.4colors-research.com/start/"


def example(repo, path, **meta):
    base = {"title": "T", "summary": "S", "domains": [], "level": "beginner", "requires": "x"}
    d = repo / path
    (d / "tests").mkdir(parents=True)
    (d / "tests" / "test_x.py").write_text("def test_x():\n    pass\n")
    (d / "README.md").write_text("# T\n")
    (d / "example.json").write_text(json.dumps({**base, **meta}))


def repo_with(tmp_path, *examples):
    (tmp_path / "products.json").write_text(json.dumps(PRODUCTS))
    for path, meta in examples:
        example(tmp_path, path, **meta)
    return tmp_path


GOOD = [
    ("quickstart/qubo", dict(kind="quickstart", slug="quickstart-qubo", products=["qubo"],
                             docs_url=DOCS + "quickstart-qubo/")),
    ("tutorials/qubo/maxcut", dict(kind="tutorial", slug="qubo-maxcut", products=["qubo"],
                                   docs_url=DOCS + "tutorials/qubo-maxcut/")),
    ("tutorials/joint/solve-then-compile", dict(kind="tutorial", slug="solve-then-compile",
                                                products=["qubo", "compiler"],
                                                docs_url=DOCS + "tutorials/solve-then-compile/")),
    ("posts/2026-11-puzzles", dict(kind="post", slug="puzzles", month="2026-11", products=["qubo", "compiler"],
                                   post_url="https://4colors-research.com/blog/puzzles/")),
]


def test_a_good_repository_passes(tmp_path):
    assert validate.validate(repo_with(tmp_path, *GOOD)) == []


def problems(tmp_path, path, **meta):
    return validate.validate(repo_with(tmp_path, (path, meta)))


def test_unknown_product(tmp_path):
    errs = problems(tmp_path, "tutorials/qubo/x", kind="tutorial", slug="qubo-x", products=["qubo", "scheduler"],
                    docs_url=DOCS + "tutorials/qubo-x/")
    assert any("'scheduler' is not in products.json" in e for e in errs)


def test_tutorial_in_the_wrong_product_folder(tmp_path):
    errs = problems(tmp_path, "tutorials/compiler/maxcut", kind="tutorial", slug="qubo-maxcut", products=["qubo"],
                    docs_url=DOCS + "tutorials/qubo-maxcut/")
    assert any("first product is 'compiler'" in e for e in errs)


def test_tutorial_folder_must_follow_the_slug(tmp_path):
    errs = problems(tmp_path, "tutorials/qubo/max-cut", kind="tutorial", slug="qubo-maxcut", products=["qubo"],
                    docs_url=DOCS + "tutorials/qubo-maxcut/")
    assert any("must be 'maxcut'" in e for e in errs)


def test_tutorial_docs_url_follows_the_slug(tmp_path):
    errs = problems(tmp_path, "tutorials/qubo/maxcut", kind="tutorial", slug="qubo-maxcut", products=["qubo"],
                    docs_url=DOCS + "tutorials/maxcut/")
    assert any("docs_url must be" in e for e in errs)


def test_joint_needs_two_products(tmp_path):
    errs = problems(tmp_path, "tutorials/joint/x", kind="tutorial", slug="x", products=["qubo"],
                    docs_url=DOCS + "tutorials/x/")
    assert any("two or more products" in e for e in errs)


def test_post_folder_and_month(tmp_path):
    errs = problems(tmp_path, "posts/2026-10-puzzles", kind="post", slug="puzzles", month="2026-11",
                    products=["qubo"])
    assert any("posts/2026-11-puzzles/" in e for e in errs)


def test_duplicate_slugs(tmp_path):
    repo = repo_with(tmp_path, *GOOD, ("posts/2026-12-qubo-maxcut", dict(kind="post", slug="qubo-maxcut",
                                                                          month="2026-12", products=["qubo"])))
    assert any("slug 'qubo-maxcut' is also used by" in e for e in validate.validate(repo))


def test_bad_products_file(tmp_path):
    (tmp_path / "products.json").write_text(json.dumps({"products": [
        {"slug": "Qubo", "name": "N", "short": "n", "status": "hidden", "docs_url": "https://example.com/",
         "requires": "x"}]}))
    errs = validate.validate(tmp_path)
    assert any("lowercase" in e for e in errs) and any("status" in e for e in errs)
    assert any("docs_url" in e for e in errs)
