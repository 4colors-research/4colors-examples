"""The leak check. Every leak below is built at run time from pieces, so this file itself stays clean."""
from pathlib import Path

import leak_check as lc

REPO = Path(__file__).resolve().parents[2]


def rules(text, **kw):
    return {f.rule for f in lc.check_text(text, "x", **kw)}


def test_generic_rules_fire():
    home = "/ho" + "me/alice/data"
    assert "home-path" in rules(f"see {home}")
    assert "private-ip" in rules("host 192." + "168.1.20 and 10." + "1.2.3")
    assert "local-host" in rules("ssh box" + ".local")
    assert "token" in rules("ghp" + "_" + "a" * 30)
    assert "secret-literal" in rules('api_key = "' + "q" * 20 + '"')
    assert "key-file" in rules("FOURCOLORS_KEY_FILE=~/keys/team" + ".key")
    assert "run-id" in rules("run 2026-11-02-" + "agent-blog-puzzles")
    assert "commit-hash" in rules("at " + "51667" + "0e0" + "e1f3")
    assert "email" in rules("mail someone" + "@" + "corp.example.net")
    assert "forbidden-link" in rules("https://api.4colors-research.com/" + "account/")
    assert "internal-record" in rules("see EXPERIMENTS" + ".md")
    assert "british-spelling" in rules("the " + "lic" + "ence and " + "col" + "our")
    assert "placeholder" in rules("matches on [" + "N] of them")



def test_stand_in_solvers_are_refused():
    for text in ("from examplekit import Off" + "lineQubo",
                 "FOURCOLORS_OFF" + "LINE=1 python solve.py",
                 "if examplekit.offline_en" + "abled():",
                 "A small local stand" + "-in answers instead of the service",
                 "run it against the offline stand" + "-in",
                 "every example runs without an API " + "key",
                 "Without a " + "key: put it in front"):
        assert "stand-in-solver" in rules(text), text
    assert "stand-in-solver" not in rules("Start a free trial at [TRIAL SIGNUP URL] and copy your key.")
    assert "stand-in-solver" not in rules("canned_result([1, 0], -1.0)  # a fixed answer for a test")

def test_clean_text_and_allowed_shapes():
    ok = ("pip install fourcolors; colors #2a78d6; 0x9E3779B97F4A7C15; date 2025-01-05, month 2026-11-qubo; "
          "bot 1+bot" + "@" + "users.noreply.github.com; license, color, optimize, modeling; threading" + ".local()")
    assert rules(ok) == set()
    assert "placeholder" not in rules("[" + "N]", placeholders=False)


def test_allowlist_accepts_an_exact_hash():
    h = "51667" + "0e0" + "e1f3"
    assert "commit-hash" not in rules(h, allow={h})


def test_denylist_hits_never_print_the_term(tmp_path):
    term = "hostal" + "pha"
    dl = tmp_path / "terms.txt"
    dl.write_text("# comment\n" + term + "\nre:\\bteam-[0-9]+\\b\n")
    deny = lc.load_denylists([str(dl)])
    found = lc.check_text(f"ssh {term.upper()} for team-42", "f.md", deny=deny)
    assert [f.rule for f in found] == ["denylist", "denylist"]
    assert all(term not in str(f).lower() and "team-42" not in str(f) for f in found)
    assert "line 2" in str(found[0]) and "line 3" in str(found[1])


def test_file_names_are_checked(tmp_path):
    (tmp_path / ("notes-" + "2026-11-02-" + "a-b.md")).write_text("fine\n")
    found = lc.check_paths([str(tmp_path)])
    assert any(f.rule == "run-id" and "[name]" in f.path for f in found)


def test_require_denylist(tmp_path, monkeypatch):
    monkeypatch.delenv("FOURCOLORS_LEAK_DENYLIST", raising=False)
    (tmp_path / "a.md").write_text("ok\n")
    assert lc.main([str(tmp_path), "--require-denylist"]) == 2
    assert lc.main([str(tmp_path)]) == 0


def test_this_repository_is_clean():
    """The whole public tree passes the generic rules (the organization denylist is checked before publishing)."""
    found = lc.check_paths([str(REPO)], allow=lc.load_allowlist(lc.DEFAULT_ALLOWLIST),
                           skip=[lc.DEFAULT_ALLOWLIST])
    assert found == [], "\n".join(str(f) for f in found)
