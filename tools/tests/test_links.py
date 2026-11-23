from links import utm


def test_own_links_get_utm_tags():
    u = utm("https://docs.4colors-research.com/qubo/?x=1&utm_source=old", "qubo-puzzles", "readme-top")
    assert u.startswith("https://docs.4colors-research.com/qubo/?x=1&utm_source=github&utm_medium=examples")
    assert "utm_campaign=qubo-puzzles" in u and "utm_content=readme-top" in u and "old" not in u


def test_other_sites_are_left_alone():
    assert utm("https://colab.research.google.com/github/x/y", "c") == "https://colab.research.google.com/github/x/y"
    assert utm("https://4colors-research.com.evil.example/", "c") == "https://4colors-research.com.evil.example/"
