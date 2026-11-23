import json

import gallery


def make_repo(tmp_path):
    (tmp_path / "domains").mkdir()
    (tmp_path / "README.md").write_text("x\n<!-- gallery:all:start -->\nold\n<!-- gallery:all:end -->\ny\n")
    (tmp_path / "domains" / "README.md").write_text(
        "<!-- gallery:domains:start -->\n<!-- gallery:domains:end -->\n")
    for path, meta in {"posts/2026-11-a": {"kind": "post", "slug": "a", "title": "A", "product": "qubo",
                                            "domains": ["finance"], "level": "beginner", "month": "2026-11",
                                            "post_url": "https://4colors-research.com/blog/a/"},
                       "quickstart": {"kind": "quickstart", "slug": "qs", "title": "Q", "product": "both",
                                      "domains": [], "level": "beginner"}}.items():
        (tmp_path / path).mkdir(parents=True)
        (tmp_path / path / "example.json").write_text(json.dumps(meta))
    return tmp_path


def test_gallery_writes_and_checks(tmp_path):
    repo = make_repo(tmp_path)
    assert gallery.main(["--repo", str(repo), "--check"]) == 1
    assert gallery.main(["--repo", str(repo)]) == 0
    assert gallery.main(["--repo", str(repo), "--check"]) == 0
    text = (repo / "README.md").read_text()
    assert text.index("[Q](quickstart/)") < text.index("[A](posts/2026-11-a/)")       # quickstarts first
    assert "utm_campaign=a" in text and text.startswith("x\n") and text.endswith("y\n")
    dom = (repo / "domains" / "README.md").read_text()
    assert "### finance" in dom and "(../posts/2026-11-a/)" in dom and "### general" in dom
