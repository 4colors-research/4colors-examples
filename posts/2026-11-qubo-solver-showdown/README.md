# Run the QUBO solver comparison yourself

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-solver-showdown&utm_content=readme-top)

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-solver-showdown&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-solver-showdown&utm_content=readme-text).
2. Install the client and the other solvers, and point the client at the key:

```bash
pip install "fourcolors[optimizer]" numpy scipy matplotlib dimod dwave-samplers openjij
git clone https://github.com/MQLib/MQLib && (cd MQLib && make)          # builds MQLib/bin/MQLib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
export MQLIB_BIN=$PWD/MQLib/bin/MQLib
```

## What you get

One row per instance, setting, budget and seed in `results/try/results.jsonl`: `objective` (the cut, or
the spin-glass energy), re-evaluated from the returned spins by `instances.energy`; `reported_energy`
(what the solver said) and `agrees`; `gap`; `wall_s` (the solver call only) and `prep_s` (building its
input, outside the clock); for 4colors the job id, run and billed seconds; for the fill settings the
calibration. A solver that fails or is skipped gets a row saying so.

`figures/gap_by_budget.png` (median over seeds, then over instances) and `figures/wall_vs_budget.png`
(longest wall time against the budget asked for, with the line wall = budget).

We do not quote reader-version numbers: they are your own result, on your hardware.

Want to try it on your own problem? Start a free trial at https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-solver-showdown&utm_content=readme-text.

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
