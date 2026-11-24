"""The repository itself follows its rules: layout, gallery, and notebooks in step with the scripts."""
import io

import docs_sync
import gallery
import products
import validate


def test_layout_rules():
    assert validate.validate() == []


def test_gallery_is_up_to_date():
    assert gallery.main(["--check"]) == 0


def test_notebooks_match_their_regions():
    out = io.StringIO()
    assert docs_sync.run("check", out=out) == 0, out.getvalue()


def test_every_product_has_a_quickstart():
    have = {m["path"] for m in products.examples() if m["kind"] == "quickstart"}
    assert have == {f"quickstart/{p['slug']}" for p in products.load()}
