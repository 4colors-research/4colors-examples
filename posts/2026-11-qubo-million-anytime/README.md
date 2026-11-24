# A good answer to a million-variable QUBO while you wait: first answer, then twice as close

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-million-anytime&utm_content=readme-top)

## What you will learn

- How to generate million-variable test models (3D and 2D spin glasses, random 3-regular Max-Cut) reproducibly
  from a seed, and fingerprint them.
- How to upload a model once and solve it by id at several budgets, and what to record for each job: the release
  that answered, search seconds, billed seconds, the time to the first answer.
- How to give other solvers a fair run: same CPU class, the same core count, the same budgets, one clock rule that
  counts input loading, and the arms interleaved in one window.
- How to read a time-to-target ladder as a bracket, not a point.

## The models

| family | spec | variables | couplings | reference |
|---|---|---|---|---|
| 3D ±J spin glass | `ea3d:L=100,seed=k` | 1,000,000 | 3,000,000 | -1.7863 per spin (published ground-state density) |
| 2D ±J spin glass on a torus | `torus2d:L=1000,seed=k` | 1,000,000 | 2,000,000 | -1.4015 per spin (published ground-state density) |
| random 3-regular Max-Cut | `regular:n=1000000,seed=k` | 1,000,000 | 1,500,000 | best cut found by any arm in the run (may be ours) |

Each model is an Ising energy H(s) = Σ c_ij s_i s_j over spins s_i = ±1. The spin glasses have couplings
c = -J with J = ±1 at random. For Max-Cut, c = 1 on every edge, and the cut is (m - H) / 2. The gap of an answer is
(energy per spin - reference) / |reference| for the glasses and (best cut - cut) / best cut for the 3-regular
graphs. Lower is better.

A ground-state density is a property of the family in the limit of a large lattice, not a proved optimum of one
instance. At a million spins the instance-to-instance spread is small, but a finite lattice can still land
slightly above or below it, so gaps below a few hundredths of a percent are inside that scatter. The 3-regular
graphs have no published constant that we use, so their reference is the best cut found. That reference can be
our own answer, and the text says so wherever those numbers appear.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-million-anytime&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-million-anytime&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" numpy scipy matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

## Step 1: one model, one second

Start with a small model so the first call returns at once:

`families.score` re-evaluates the returned spins on your machine. It does not trust the service's objective, and
the harness checks that the two agree (`agrees` in every row).

## Step 2: a million variables, uploaded once

At a million variables you do not want to send the model with every job. Upload it once and name it by id:

```python
inst = families.build("ea3d:L=100,seed=1")            # 1M spins, 3M couplings, a few seconds to build
model = run.api_model(inst)
model_id = fourcolors.qubo.upload(model)              # sent once (zstd); skipped if the service already holds it
for T in (1, 3, 10, 30, 60):
    r = fourcolors.qubo.solve(input_id=model_id, time_limit_s=T, seed=1)
    print(T, families.score(inst, r.solution)["gap"], r.billed_s)
```

`run.py` does this for every instance and budget, with the first-answer clock (`on_incumbent`) and the recorded
fields:

```bash
python run.py --out results/ea3d --instance ea3d:L=100,seed=1 --arms api --budgets 1 3 10 30 60 --seeds 1 2 3
```

Each budget is its own job, because the service plans its search for the time it is given. A "curve" here is a
sweep of budgets, not snapshots of one long run.

## Step 3: the other solvers, on the same footing

```bash
python run.py --out results/ea3d --instance ea3d:L=100,seed=1 \
  --arms api neal neal-fill openjij openjij-fill mqlib-hh mqlib-burer tabu sb-fill \
  --budgets 1 3 10 30 60 --seeds 1 2 3 4 5 6 7 8 9 10 --cpus 4 5 --streams 2
```

The rules the harness enforces:

- **One clock rule.** A local solver's time starts when it begins building its input from the instance arrays and
  ends when it returns. Slow input paths count, as they do for a user. Our time is the client wall time from submit
  to answer, with the network included. The model was uploaded beforehand, and the upload time is reported on its
  own.
- **Same cores.** The local solvers are single-threaded. When the service reports that a job used k threads,
  each local solver gets k cores and runs k independent seeds in parallel, keeping the best (`--streams k`). The
  rivals run on the same CPU class as the service.
- **Fair settings.** Each annealer runs twice: once with library defaults, and once with its sweep count calibrated
  on the same model to end at the budget. The calibration pilots run outside the clock. A solver that runs past
  its budget keeps its answer and is flagged `overran`; that only ever favors it.
- **Interleaved.** For every (instance, budget, seed), all arms run back to back in a seeded random order, so
  drift on the machine hits every arm alike.

## Step 4: the time-to-target ladder

The budget sweep says how close each solver gets at equal time. The ladder turns it around: how much time does
another solver need to match our 10-second answer? The budget-filling annealers rerun at 90, 120, 180, 240 and
300 s:

```bash
python run.py --out results/ea3d-ladder --instance ea3d:L=100,seed=1 --arms neal-fill openjij-fill sb-fill \
  --budgets 90 120 180 240 300 --seeds 1 2 3 4 5 6 7 8 9 10 --cpus 4 5 --streams 2
python analyze.py results/ea3d/results.jsonl results/ea3d-ladder/results.jsonl --out results/analysis
python plot.py results/analysis/analysis.json --out figures
```

The answer is a bracket: the largest budget at which the rival still missed our 10 s gap, and the first at which
it reached it. We report the bracket ("between 6x and 12x"), never a point inside it.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the generators, harness and analysis; they do not call the service
```

The tests check the generators (a periodic ±1 lattice, a simple 3-regular graph, the seeded fingerprint), the
references and gap signs, that the API model's energy equals the evaluator on sampled assignments, that the sparse
simulated bifurcation reaches the exact ground state of a 9-spin glass, the selection rule and the time-to-target
bracket on hand-made rows, that a rival error is recorded and not dropped, and the API plumbing and the plots on a
fixed canned answer.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
