"""Quickstart: solve a QUBO, then watch a longer search.

    pip install "fourcolors[optimizer]"
    python quickstart/qubo/quickstart.py      # needs FOURCOLORS_API_KEY or FOURCOLORS_KEY_FILE

No key yet? Start a free trial at
    https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=quickstart-qubo&utm_content=script
and copy your key.

The code between the `region docs:` lines is the docs quickstart, word for word
(https://docs.4colors-research.com/start/quickstart-qubo/). The second part searches for up to 60 seconds; press
Ctrl-C to stop it early and keep the best solution so far.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
import examplekit  # noqa: E402

examplekit.load_key()

# region docs:solve
import fourcolors

Q = {(0, 1): 2, (1, 2): 2, (2, 3): 2, (0, 3): -3}
linear = [-1, -1, -1, -1]

result = fourcolors.qubo.solve(Q, time_limit_s=1, linear=linear)

print(result.status)     # finished
print(result.objective)  # the best energy found
print(result.solution)   # the values, in variable order
print(result.billed_s)   # execution time billed for this job, in seconds
# endregion

# region docs:watch
import fourcolors

job = fourcolors.qubo.submit(Q, time_limit_s=60, linear=linear)
print(job.status)      # queued or running

# each new best as it is found; the final result comes last
for best in job.incumbents():
    print(best.objective)

result = job.result()  # the final result (already ended here)
# endregion
