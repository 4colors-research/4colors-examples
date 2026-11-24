# Stop paying for idle seconds: stop, stream and incumbents

**Docs:** [4colors Optimizer for QUBO](https://docs.4colors-research.com/qubo/?utm_source=github&utm_medium=examples&utm_campaign=qubo-stop-and-stream&utm_content=readme-top)

## Setup

Start a free trial at [account.4colors-research.com](https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-stop-and-stream&utm_content=readme-setup) and copy your key, then:

1. Get an API key (a free trial key works: https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-stop-and-stream&utm_content=readme-text).
2. Install the client and point it at the key:

```bash
pip install "fourcolors[optimizer]" matplotlib
export FOURCOLORS_KEY_FILE=/path/to/key-file   # or FOURCOLORS_API_KEY=<your trial key>
```

The model is Max-Cut on a random 3-regular graph with 100,000 nodes (`maxcut_model.py`), big enough that
improvements keep arriving for a while. Max-Cut asks for a split of the nodes into two sides that cuts as many edges
as possible; written as a QUBO, the energy is minus the cut.

What this post bills if you run every command: at most 60 s for Step 1 and at most 4 x 60 s for Step 2, so under
5 search-minutes, and less because the patience jobs stop early.

## Step 1: watch it work

`fourcolors.qubo.submit` returns a running job at once; the search goes on while your code runs.
`job.incumbents()` yields each better answer as it is found and then the final result, always last.

```python
import fourcolors.qubo

job = fourcolors.qubo.submit(model, time_limit_s=60, seed=1)
for best in job.incumbents():
    print(best.status, best.objective)      # "running" for each new best, then the final result
print(best.billed_s)                        # the final result's billed search seconds
```

```bash
python watch.py --out runs/watch --n 100000 --time-limit 60
python watch.py --out runs/watch --n 100000 --time-limit 60 --live-plot      # a window that redraws
```

## Step 2: a no-progress stop rule

`job.best()` peeks at the best answer so far without waiting (None until the first one); `job.stop()` stops the
job and returns the best answer it had. The rule below stops once the cut has not improved for `patience` seconds:

```python
best_energy, last_gain = None, time.monotonic()
while True:
    b = job.best()                                   # None until the first answer
    if b is not None and b.status != "running":      # the job ended by itself
        result = b; break
    if b is not None and (best_energy is None or b.objective < best_energy):
        best_energy, last_gain = b.objective, time.monotonic()
    if best_energy is not None and time.monotonic() - last_gain >= patience:
        result = job.stop(); break                   # keeps the best so far
    time.sleep(1)
```

```bash
python stop_rule.py --out runs/stop5  --n 100000 --time-limit 60 --patience 5  --seeds 1 2
python stop_rule.py --out runs/stop15 --n 100000 --time-limit 60 --patience 15 --seeds 1 2
```

### Pick the patience from the record, not a guess

A trajectory only gets a point when the service reports an improvement, so the gaps between points set a floor on a
sensible patience. Read them from your own run:

```bash
python cadence.py runs/watch/results.jsonl
```

## Step 3: plot

```bash
python plot.py runs/watch runs/stop5 runs/stop15 --out figures
```

## Run the unit tests

```bash
python -m pytest -q tests      # unit tests of the model code; they do not call the service
```

The tests drive the stop rule with a fake clock and a job that holds one fixed answer: it stops exactly `patience`
seconds after the last gain, returns a job that ended by itself as it is, and keeps the best on Ctrl-C. They also check the
cadence summary.

## Run it yourself

Set `FOURCOLORS_API_KEY` (or `FOURCOLORS_KEY_FILE=/path/to/file`, which keeps the key out of your shell history) and run the commands above. Every answer comes from the 4colors service.
