"""live_check on a throwaway repository whose "examples" are tiny local scripts: nothing calls the service."""
import json
import sys

import live_check

PRODUCTS = {"products": [{"slug": s, "name": s, "short": s, "status": "live",
                          "docs_url": "https://docs.4colors-research.com/", "requires": "x"} for s in ("qubo", "compiler")]}


def make(tmp_path):
    (tmp_path / "products.json").write_text(json.dumps(PRODUCTS))
    scripts = {"quickstart/qubo": ("quickstart-qubo", "quickstart", ["qubo"],
                                   "import os; print('key seen' if os.environ.get('FOURCOLORS_API_KEY') else 'no key')"),
               "tutorials/qubo/bad": ("qubo-bad", "tutorial", ["qubo"], "raise SystemExit(3)"),
               "quickstart/compiler": ("quickstart-compiler", "quickstart", ["compiler"], "print('c')")}
    for path, (slug, kind, prods, code) in scripts.items():
        d = tmp_path / path
        d.mkdir(parents=True)
        (d / "s.py").write_text(code + "\n")
        (d / "example.json").write_text(json.dumps({"kind": kind, "slug": slug, "products": prods,
                                                    "run": "python s.py", "billed_s_max": 10}))
    return tmp_path


def test_dry_run_lists_only_the_chosen_products(tmp_path, capsys):
    repo = make(tmp_path)
    assert live_check.main(["--products", "qubo", "--dry-run", "--repo", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "quickstart/qubo" in out and "tutorials/qubo/bad" in out and "compiler" not in out
    assert "2 example(s); at most 20 s" in out


def test_unknown_product(tmp_path):
    assert live_check.main(["--products", "nope", "--dry-run", "--repo", str(make(tmp_path))]) == 2


def test_refuses_to_run_with_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("FOURCOLORS_API_KEY", raising=False)
    monkeypatch.delenv("FOURCOLORS_KEY_FILE", raising=False)
    repo = make(tmp_path)
    assert live_check.main(["--products", "qubo", "--out", str(tmp_path / "o"), "--repo", str(repo)]) == 2


def test_runs_records_and_fails_on_a_bad_example(tmp_path, monkeypatch, capsys):
    key = tmp_path / "k"
    key.write_text("from-file\n")
    monkeypatch.setenv("FOURCOLORS_KEY_FILE", str(key))
    monkeypatch.delenv("FOURCOLORS_API_KEY", raising=False)
    repo, out = make(tmp_path), tmp_path / "o"
    assert live_check.main(["--products", "qubo", "--out", str(out), "--repo", str(repo)]) == 1
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    assert [(r["slug"], r["exit"]) for r in rows] == [("quickstart-qubo", 0), ("qubo-bad", 3)]
    assert (out / "quickstart-qubo.out").read_text().strip() == "key seen"
    text = (out / "results.jsonl").read_text() + capsys.readouterr().out
    assert "from-file" not in text                                   # the key is never written or printed
