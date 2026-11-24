"""Large QUBO models: upload once, solve by id, stop when the answer stops improving.

    pip install "fourcolors[optimizer]"
    python tutorials/qubo/large-models/large_models.py    # 50,000 nodes, 150,000 edges, up to 120 s

Needs a key (FOURCOLORS_API_KEY or FOURCOLORS_KEY_FILE): start a free trial at
    https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=qubo-large-models&utm_content=script
and copy it.

The code between the `region docs:` lines is the docs tutorial, word for word
(https://docs.4colors-research.com/tutorials/qubo-large-models/). The job stops itself after 20 seconds
without a better answer, and you are billed only for the time it ran.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
import examplekit  # noqa: E402

examplekit.load_key()

# region docs:upload
import random
import fourcolors

n, m = 50_000, 150_000
rnd = random.Random(11)
edges = set()
while len(edges) < m:
    i, j = rnd.randrange(n), rnd.randrange(n)
    if i != j:
        edges.add((min(i, j), max(i, j)))
Q = {e: 2 for e in edges}
linear = [0] * n
for i, j in edges:
    linear[i] -= 1
    linear[j] -= 1

model_id = fourcolors.qubo.upload(Q, linear=linear)
print(model_id)           # the model's id: 64 hexadecimal digits

again = fourcolors.qubo.upload(Q, linear=linear)
print(again == model_id)  # True: the same model has the same id; nothing new is stored
# endregion

# region docs:submit
job = fourcolors.qubo.submit(input_id=model_id, time_limit_s=120)
print(job.status, job.queue_position, job.estimated_start_at)
# endregion

# region docs:stop
import time

patience = 20       # seconds without improvement before we stop
best_cut, improved_at = None, time.monotonic()

while job.status in ("queued", "running"):
    b = job.best()  # the best so far, or None
    if (b is not None and b.objective is not None
            and (best_cut is None or -b.objective > best_cut)):
        best_cut, improved_at = -b.objective, time.monotonic()
        print(f"best cut so far {best_cut:,}")
    if time.monotonic() - improved_at > patience:
        break
    time.sleep(2)

# stops the job and returns its best; on an ended job, just its result
final = job.stop()
print(final.status, -final.objective, job.billed_s)
# endregion
