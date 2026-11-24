# EV smart charging as a maximum independent set, from 100 to 100,000 tasks

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-ev-charging&utm_content=readme-top)

## The model: tasks, clashes and two kinds of rows

A **task** is one way to serve one vehicle: (vehicle, charger, start slot), running for the slots the vehicle
needs, inside its arrival-departure window. Two tasks clash when they

1. serve the same vehicle (a vehicle charges once), or
2. use the same charger in an overlapping slot.

Pick clash-free tasks that maximize delivered value (priority x energy): that is maximum-weight independent set
(MWIS) on the clash graph. A QUBO solver without rows needs it as a penalty, `-sum w x + P sum_edges x_u x_v`, with
P larger than any task's value. With exact rows the clashes are stated directly, and more compactly:

```python
import charging

inst = charging.for_tasks(1000, seed=1)            # a synthetic depot with about 1,000 candidate tasks
m = charging.model(inst)                           # maximize value; rows below
# one row per vehicle:            sum of its tasks          <= 1
# one row per (charger, slot):    sum of tasks using it     <= 1
print(len(inst.tasks), m.m, len(charging.conflict_edges(inst)))   # tasks, rows, the clash graph's edges
```

Every pairwise clash lies inside one of those groups, so the group rows say exactly what the edges say: the tests
check that the two forms admit the same schedules, on every assignment of a small depot. And a constraint that is
not a clash at all, a **site power cap** ("the chargers running in any slot may draw at most 60 kW"), is one more row
per slot (`site_kw=60`); on a pure independent-set encoding it has no natural place.

## Step by step

### Setup

1. Get an API key: start a free trial at https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-ev-charging&utm_content=readme-text, then create a key at account.4colors-research.com
   (docs: docs.4colors-research.com).
2. Install the client and point it at the key:

```bash
pip install fourcolors scipy matplotlib pyyaml pytest
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

Run the scripts from this directory (they import `../../shared/examplekit.py`).

### Step 1: solve, decode, check

```python
import fourcolors.qubo

r = fourcolors.qubo.solve(m, time_limit_s=1, seed=1)
schedule = charging.decode(inst, r.solution)       # [(vehicle, charger, start), ...]
print(charging.check(inst, schedule))              # {"valid": True, "value": ..., "served": ..., ...}
print(m.feasible(r.solution), r.constraints["feasible"])   # the client's row check and the service's
```

`check` knows nothing about the model: it walks the schedule and confirms that no vehicle charges twice, every task
fits its window, no charger is double-booked and the cap (if any) holds. Every constrained answer is checked by the
service against the rows too.

### Step 2: a proven reference

```python
ref = charging.milp_reference(inst, time_limit_s=60)   # HiGHS through scipy.optimize.milp, same rows
print(ref["reference_value"], ref["optimal"], ref["upper_bound"])
```

Where HiGHS proves optimality in its time limit, the post compares with the proven optimum; where it does not, it
says "at least the reference's best", never "optimal".

### Step 3: the whole run

```bash
python run.py --out results/ev                 # about 100, 1,000, 10,000 and 100,000 tasks; 1 and 10 s; seeds 1-3
python plot.py results/ev --figs figs
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

They check that the generator is deterministic and every task fits its window, that the group rows and the
pairwise rows admit exactly the valid schedules (every assignment of a small depot, with and without a power cap),
that the penalty form has the same optimum, that the HiGHS reference is the brute-force optimum, that the checker
catches double bookings and window violations, the selection rules, and the plumbing of `run.py` and `plot.py` on a
fixed canned result.

## Run it yourself

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-ev-charging&utm_content=readme-trial) for a key. Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
