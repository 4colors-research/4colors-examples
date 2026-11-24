# Stop tuning penalty weights: exact constraints for QUBO and CQM

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-exact-constraints&utm_content=readme-top)

## What you will learn

- How the penalty recipe turns a constraint into `lam * (violation)^2`, and why an inequality needs slack bits.
- How to keep the objective as it is and add each constraint with `Model.add_constraint`, with no `lam` and no slack.
- How to send the same model as a dimod `ConstrainedQuadraticModel` through `FourColorsSampler`, if your code already
  builds CQMs.
- How to check an answer three ways: the service's verdict (`result.constraints`), the client's exact check
  (`model.feasible`), and your own plain Python.

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-exact-constraints&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-exact-constraints&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer,ocean]" scipy matplotlib pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

The scripts import `../../shared/examplekit.py`; run them from this directory. What the steps bill: Step 2 runs 54 jobs at 5 s
(4.5 search-minutes), Step 3 runs 6 jobs at 5 s, Step 4 runs 6 jobs at 1 s: about 5 search-minutes in all.
Every search runs to its time limit and bills it.

## Step 1: build both models

`constrained_problems.py` holds the builders. The penalty and exact versions of the assignment problem:

```python
import fourcolors.qubo
import constrained_problems as cp

c = cp.assignment_instance(12, seed=7)

penalty = cp.assignment_penalty_model(c, lam=50.0)    # 144 variables, 1,584 quadratic terms, no rows

exact = fourcolors.qubo.to_model({}, [c[i][j] for i in range(12) for j in range(12)], n=144)
for i in range(12):
    exact.add_constraint({i * 12 + j: 1 for j in range(12)}, lo=1, hi=1)   # worker i does one job
for j in range(12):
    exact.add_constraint({i * 12 + j: 1 for i in range(12)}, lo=1, hi=1)   # job j gets one worker
```

The knapsack capacity is one row: `m.add_constraint({i: w[i] for i in range(n)}, hi=capacity)`. Its penalty version
needs slack bits 1, 2, 4, ..., 128 and a last one of 155, so that every unused capacity from 0 to 410 can be written
exactly; `cp.slack_bits(410)` returns them.

Try the exact model once before any sweep (1 s is enough on a problem this size):

```python
r = fourcolors.qubo.solve(exact, time_limit_s=1, seed=1)
r.constraints              # the service's check: {"m": 24, "feasible": True, "rows_checked": 24, "rows_violated": 0}
exact.feasible(r.solution) # the same check, exactly, on your machine
```

If no answer meeting every row is found in time, the job still finishes, with `r.reason["code"] ==
"no_feasible_found"` and `r.solution is None`. That is not a proof that none exists; the scripts record it as
infeasible.

## Step 2: the search for lam (script A)

```bash
python sweep_penalty.py --out runs/penalty --time-limit 5 --seeds 1 2 3
```

For each problem, each `lam` (a multiple of the largest cost for the assignment, of the largest value for the
knapsack: 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100) and each seed, it solves the penalty QUBO for the same time limit
and judges the answer in plain Python. That is 27 jobs per problem to learn which `lam` works. Each line printed
reads like `assignment lam=0.1 seed=1 feasible=False objective=96 (optimum 134)`.

At small `lam` the answers break the constraint: for the assignment, leaving workers idle costs less than the
penalty. All three seeds give feasible answers from `lam` = 0.3 x the largest cost (assignment) and 0.03 x the
largest value (knapsack). Once feasible, the assignment answers are at the optimum (median gap 0%). On the knapsack,
the smallest feasible weights (0.03 to 0.3 x the largest value) still lose a median 0.45% to 0.79% of the best value;
from 1 x the largest value up, the median gap is 0%. So the search does find a good `lam`, and it takes the whole
sweep to know which one.

## Step 3: the same problems with exact rows (script B)

```bash
python exact_rows.py --out runs/exact --time-limit 5 --seeds 1 2 3
```

Each line reads like `assignment exact rows seed=1 feasible: service=True client=True python=True objective=134
(optimum 134)`. Here the three checks agree on every run (six runs), and the median gap to the optimum is 0%. Three
jobs per problem, not 27, and nothing to choose beforehand.

## Step 4: if your code builds a dimod CQM (script C)

If you model with dimod, you do not need to rewrite anything. `FourColorsSampler` takes a
`ConstrainedQuadraticModel` whose variables are all binary and whose constraints are linear and hard, and sends each
constraint as an exact row:

```python
import dimod
from fourcolors import FourColorsSampler

x = [dimod.Binary(f"x{i}") for i in range(len(w))]
cqm = dimod.ConstrainedQuadraticModel()
cqm.set_objective(-dimod.quicksum(vi * xi for vi, xi in zip(v, x)))
cqm.add_constraint(dimod.quicksum(wi * xi for wi, xi in zip(w, x)) <= capacity, label="capacity")

ss = FourColorsSampler().sample(cqm, time_limit_s=1, seed=1)
ss.first.sample, ss.first.energy   # the one best answer, with dimod's objective
ss.info["constraints"]             # the service's exact check: the verdict to trust
```

```bash
python cqm_path.py --out runs/cqm --time-limit 1 --seeds 1 2 3
```

Two things differ from a sampler you may know. The SampleSet holds one sample, the best answer found; `num_reads`
above 1 is refused, because the service returns one answer per job. And `ss.first.is_feasible` is dimod's own check
in floating point with a tolerance, which can disagree with the exact check for a row sitting right at its bound;
check, dimod's check and plain Python agreed on every answer (six answers).

## Step 5: plot

```bash
python plot.py runs/penalty runs/exact --out figures
```

## What each row in results.jsonl holds

`kind` (`reference`, `penalty`, `exact` or `cqm`), the problem and its size, `n_vars`, `n_quad` (quadratic terms),
`n_rows`, `lam_factor` and `lam`, `seed`, `time_limit_s`, `optimum`, the plain-Python verdict (`feasible`,
`objective`, `gap`, `penalty_energy_python`), `model_energy_python` (the model's energy of the returned vector,
computed here), and what the service reported (`solver_objective`, `service_constraints`, `status`, `reason`,
`billed_s`, `job_id`, `engine_version`, `trajectory`). CQM rows add `service_feasible`, `dimod_feasible` and
`checks_agree`.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check every builder against the plain-Python objective on all assignments of tiny cases, the exact
references against brute force, the model sizes the post states (slack bits, quadratic terms, rows), that a dimod
CQM becomes the same exact rows, and every script's plumbing on fixed canned answers.

## Results

The numbers and charts are in the post (link added on publication). Every number there names the 4colors release and the `fourcolors` client version it was measured with; each row this code writes records both (`engine_version`, `client_version`).

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
