#!/usr/bin/env bash
# The API without Python: a QUBO job and a compile with curl and jq.
#
#   export FOURCOLORS_API_KEY=...        # no key yet? start a free trial at https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=http-without-python&utm_content=script
#   bash tutorials/platform/http-without-python/http_tutorial.sh
#
# The code between the "region docs:" lines is the docs tutorial, word for word
# (https://docs.4colors-research.com/tutorials/http-without-python/). The compile's event stream can stay
# open for a few minutes after "done", until the check of the compiled circuit ends.
set -euo pipefail
: "${FOURCOLORS_API_KEY:?set FOURCOLORS_API_KEY (no key yet? start a free trial at https://account.4colors-research.com/account/?utm_source=github&utm_medium=examples&utm_campaign=http-without-python&utm_content=script)}"
B=${FOURCOLORS_BASE_URL:-https://api.4colors-research.com}
A="Authorization: Bearer $FOURCOLORS_API_KEY"

# region docs:plan
curl -s -H "$A" $B/v1/license | jq '{budget, limits}'
# endregion

# region docs:submit
JOB=$(curl -s -H "$A" -H "Content-Type: application/json" -X POST $B/v1/jobs -d '{
  "solver": "optimizer.qubo",
  "input": {"qubo": {"linear": [-1, -1, -1], "quadratic": [[0, 1, 2], [1, 2, 2]]}},
  "options": {"time_limit_s": 5}
}' | jq -r .id)
echo $JOB
# endregion

# region docs:events
curl -sN -H "$A" $B/v1/jobs/$JOB/events
# endregion

# region docs:record
curl -s -H "$A" $B/v1/jobs/$JOB | jq '{status, billed_s,
  objective: .result.objective, solution: .result.solution_list}'
# endregion

# region docs:compile
TARGET='{"format_version": 2, "n_qubits": 3, "two_qubit_gate": "cz",
         "couplers": [[0, 1], [1, 2]]}'

BODY=$(jq -n --arg c 'OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
' --argjson t "$TARGET" '{solver: "quantum.compiler",
    input: {circuit: $c},
    options: {target: $t, time_budget_s: 15}}')

JOB=$(curl -s -H "$A" -H "Content-Type: application/json" \
  -X POST $B/v1/jobs -d "$BODY" | jq -r .id)
curl -sN -H "$A" $B/v1/jobs/$JOB/events
curl -s -H "$A" $B/v1/jobs/$JOB | jq '{status, billed_s,
  metrics: .result.metrics, verification: .result.verification.label}'
# endregion
