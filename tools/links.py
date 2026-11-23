"""Links from this repository back to the 4colors site and docs, tagged so visits from here can be counted.

Only links to 4colors' own sites get UTM tags; links to anyone else's site are left as they are.
The tags say where the click came from (this repository, which page, which spot), never who clicked.
"""
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

OWN_HOSTS = ("4colors-research.com",)
SOURCE = "github"
MEDIUM = "examples"


def is_own(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    return any(host == h or host.endswith("." + h) for h in OWN_HOSTS)


def utm(url: str, campaign: str, content: str = "") -> str:
    """`url` with utm_source=github, utm_medium=examples, utm_campaign=<campaign>[, utm_content=<content>].

    Existing query parameters are kept; existing utm_* values are replaced. Other sites' links come back as given."""
    if not is_own(url):
        return url
    parts = urlsplit(url)
    q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.startswith("utm_")]
    q += [("utm_source", SOURCE), ("utm_medium", MEDIUM), ("utm_campaign", campaign)]
    if content:
        q.append(("utm_content", content))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment))
