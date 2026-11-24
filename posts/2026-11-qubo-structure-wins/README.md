# Structure wins: knapsack, number partitioning and constraint detection

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-structure-wins&utm_content=readme-top)

## What you will learn

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-structure-wins&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-structure-wins&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" numpy matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

## Step 1: the knapsack as the textbook writes it

```python
import fourcolors.qubo
import structure as S

inst = S.knapsack_instance(500, seed=1)
Q, lin, offset, n_vars, lam = S.knapsack_penalty_qubo(inst)    # items + slack bits, all integers
m = fourcolors.qubo.to_model(Q, lin, n=n_vars, offset=offset)

r = fourcolors.qubo.solve(m, time_limit_s=10, seed=1)
print(S.knapsack_judge(inst, r.solution))                     # feasible? value? weight?
print(S.knapsack_optimum(inst))                              # the exact optimum, by dynamic programming
```

## Step 2: number partitioning as one square

```python
inst = S.numpart_instance(200, seed=1)
m = S.numpart_model(inst)                                   # (sum a_i s_i)^2 as an Ising model
r = fourcolors.qubo.solve(m, time_limit_s=10, seed=1)
d = S.numpart_diff(inst, r.solution)
print(d, "proved optimal" if d == S.numpart_lower_bound(inst) else "")
```

The difference of the two sums always has the parity of the total, so an answer with difference 0 (even total) or
1 (odd total) is optimal, with no solver needed to say so.

## Step 3: what the service saw

`run.py` stores every result attribute it does not already know under `extra_fields`, so the record keeps the
report whatever it is called.

## Step 4: run the whole set

```bash
python run.py --out runs/structure --budgets 1 10 60 --seeds 1 2 3
python plot.py runs/structure/results.jsonl runs/structure-rivals/results.jsonl --out figures
```

Each knapsack instance runs twice: as the penalty QUBO (`4colors-penalty`) and with the explicit row
(`4colors-rows`), so the post can show how close the plain QUBO gets to the model you would write by hand.

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests check that the slack bits reach exactly 0..C, that the penalty QUBO equals its definition on every
assignment of a small case and that its minimum is the feasible optimum, that the dynamic-programming optimum
equals brute force, that the partitioning energy is the squared difference on each returned answer, that every coefficient
stays an exact integer at the largest size, the selection rule on hand-made rows, and the scripts' plumbing on fixed
canned answers.

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
