# Traction metrics

How many people find, read, clone, star and run these examples, and how many install `fourcolors`.
Nothing here runs on its own: the collector is a script, and the systemd units are templates to install by hand.

## The metric that matters: trial sign-ups

The goal of this repository is that readers start a free trial. Everything below (views, clones, stars,
installs) is a leading signal; the number we judge the repository by is **trial sign-ups that came from it**.

- Every trial link in the repository carries UTM tags (`utm_source=github&utm_medium=examples&utm_campaign=<example>
  &utm_content=<spot>`), built with `tools/links.py`. Until the sign-up URL is confirmed the link is the literal
  placeholder `[TRIAL SIGNUP URL]`, which carries no tags; when it is filled in, tag every occurrence.
- Attribution happens on the site, not here: a sign-up counts for the repository only if the site's analytics and
  back office keep the campaign from the first visit through to the trial sign-up. Whether that works, and for how
  long the first-visit campaign is kept, depends on the site's analytics setup; this collector cannot see it.
- Report it weekly beside the GitHub numbers: trial sign-ups with `utm_source=github&utm_medium=examples`, by
  campaign (example) and spot.

## What is measured, and where it comes from

| Signal | Source | Kept by the source | Collected as |
|---|---|---|---|
| Page views, unique visitors per day | GitHub traffic API `traffic/views?per=day` | 14 days | `github-traffic.jsonl` (kind `views`) |
| Clones, unique cloners per day | GitHub traffic API `traffic/clones?per=day` | 14 days | `github-traffic.jsonl` (kind `clones`) |
| Top 10 referrers, top 10 pages | `traffic/popular/referrers`, `traffic/popular/paths` | 14 days, top 10 | `github-traffic.jsonl` (kinds `referrers`, `paths`) |
| Stars, forks, watchers | `GET /repos/{owner}/{repo}` | current value only | `github-repo.jsonl` (a snapshot per run) |
| Star history | `stargazers` with `Accept: application/vnd.github.star+json` | all stars, with time | `github-stars.jsonl` (time and salted account hash) |
| Release downloads | `releases[].assets[].download_count` | running total | `github-releases.jsonl` |
| `fourcolors` installs | pypistats.org `recent` and `overall` | 180 days | `pypi.jsonl` |
| Binder launches | mybinder.org public launch archive | permanent | `binder.jsonl` |
| Clicks to the site and docs | UTM tags on every link to 4colors sites (`tools/links.py`) | the site's own analytics | not collected here |
| **Trial sign-ups from the repository** | UTM-tagged trial links, attributed by the site's analytics and back office | the site | not collected here |
| Colab opens | not reported by Google; see "Colab" below | | |

GitHub's traffic numbers exist for 14 days only, so a run must happen at least every few days. Run it daily:
each day's numbers are revised while the day is still running, and the report keeps the latest value per day.

PyPI counts are downloads, not people: CI systems and mirrors download too. pypistats separates mirrors
(`without_mirrors` is what the report uses). For finer cuts (Python version, OS, installer), the public BigQuery
dataset `bigquery-public-data.pypi.file_downloads` can be queried by the owner with their own Google Cloud project.

### Colab

Google does not report notebook opens to repository owners. Two options, an owner decision:

1. Plain "Open in Colab" badges (no counting). Private for readers.
2. Badges that go through a redirect on the 4colors site, which counts the click and forwards to Colab. The
   site's server log then records the reader's visit like any other page visit. That must be covered by the
   site's privacy notice, and the badge should say it goes through 4colors.

## Run it by hand

```bash
mkdir -p ~/.config/fourcolors-metrics ~/.local/state/fourcolors-metrics
cp tools/metrics/config.example.json ~/.config/fourcolors-metrics/config.json
# a GitHub token with read-only Administration + Metadata on the repositories, in a file only you can read:
install -m 600 /dev/null ~/.config/fourcolors-metrics/github-token && $EDITOR ~/.config/fourcolors-metrics/github-token
FOURCOLORS_METRICS_TOKEN_FILE=~/.config/fourcolors-metrics/github-token \
  python3 tools/metrics/collect.py --config ~/.config/fourcolors-metrics/config.json
python3 tools/metrics/report.py > report.md
```

`--only pypi binder` runs the sources that need no token.

## Install as a systemd user timer (the owner does this; nothing here installs it)

```bash
mkdir -p ~/.config/systemd/user ~/.local/state/fourcolors-metrics
sed "s|@CHECKOUT@|$PWD|" tools/metrics/systemd/fourcolors-metrics.service > ~/.config/systemd/user/fourcolors-metrics.service
cp tools/metrics/systemd/fourcolors-metrics.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start fourcolors-metrics.service      # one run now; check collect.log
systemctl --user enable --now fourcolors-metrics.timer
systemctl --user list-timers fourcolors-metrics.timer
```

The timer has `Persistent=true`, so a run missed while the machine was off or suspended happens at the next start.
For runs while nobody is logged in, lingering must be on for the user (`loginctl enable-linger`).

## Privacy

- No reader identity is stored. Stars are kept as a time plus a salted hash of the account (only to avoid counting
  a star twice); the salt is random per installation and never leaves the machine.
- GitHub's traffic API gives counts only, never who visited.
- The pypistats and Binder data are public aggregates.
- The token is read from a file named by an environment variable; it is never logged or written to the outputs.
