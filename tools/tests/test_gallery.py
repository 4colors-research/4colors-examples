import json

import gallery

PRODUCTS = {"products": [
    {"slug": "qubo", "name": "4colors Optimizer for QUBO", "short": "Optimizer for QUBO", "status": "live",
     "docs_url": "https://docs.4colors-research.com/qubo/", "requires": "fourcolors[optimizer]>=1.0.0"},
    {"slug": "compiler", "name": "4colors Quantum Compiler", "short": "Quantum Compiler", "status": "preview",
     "docs_url": "https://docs.4colors-research.com/compiler/", "requires": "fourcolors>=1.0.2"},
]}


def make_repo(tmp_path):
    (tmp_path / "domains").mkdir()
    (tmp_path / "products.json").write_text(json.dumps(PRODUCTS))
    (tmp_path / "README.md").write_text("x\n<!-- gallery:products:start -->\nold\n<!-- gallery:products:end -->\ny\n")
    (tmp_path / "domains" / "README.md").write_text(
        "<!-- gallery:domains:start -->\n<!-- gallery:domains:end -->\n")
    for path, meta in {"posts/2026-11-a": {"kind": "post", "slug": "a", "title": "A", "products": ["qubo", "compiler"],
                                            "domains": ["finance"], "level": "beginner", "month": "2026-11",
                                            "post_url": "https://4colors-research.com/blog/a/"},
                       "quickstart/qubo": {"kind": "quickstart", "slug": "quickstart-qubo", "title": "Q",
                                           "products": ["qubo"], "domains": [], "level": "beginner",
                                           "docs_url": "https://docs.4colors-research.com/quickstart-qubo/"},
                       }.items():
        (tmp_path / path).mkdir(parents=True)
        (tmp_path / path / "example.json").write_text(json.dumps(meta))
    return tmp_path


def test_gallery_writes_and_checks(tmp_path):
    repo = make_repo(tmp_path)
    assert gallery.main(["--repo", str(repo), "--check"]) == 1
    assert gallery.main(["--repo", str(repo)]) == 0
    assert gallery.main(["--repo", str(repo), "--check"]) == 0
    text = (repo / "README.md").read_text()
    assert text.startswith("x\n") and text.endswith("y\n")
    qubo, compiler = text.split("### 4colors Quantum Compiler (preview)")
    assert "### 4colors Optimizer for QUBO" in qubo
    assert qubo.index("[Q](quickstart/qubo/)") < qubo.index("[A](posts/2026-11-a/)")     # quickstarts first
    assert "[A](posts/2026-11-a/)" in compiler and "[Q](quickstart/qubo/)" not in compiler  # under both products
    assert "Optimizer for QUBO, Quantum Compiler" in qubo
    assert "utm_campaign=a" in text and "utm_campaign=quickstart-qubo" in text
    dom = (repo / "domains" / "README.md").read_text()
    assert "### finance" in dom and "(../posts/2026-11-a/)" in dom and "### general" in dom
