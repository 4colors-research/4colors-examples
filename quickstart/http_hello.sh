#!/usr/bin/env bash
# Hello, HTTP: a tiny QUBO job with curl and jq, no Python. Needs FOURCOLORS_API_KEY.
#   bash quickstart/http_hello.sh
set -euo pipefail
: "${FOURCOLORS_API_KEY:?set FOURCOLORS_API_KEY (get a key at account.4colors-research.com)}"
B=${FOURCOLORS_BASE_URL:-https://api.4colors-research.com}
A="Authorization: Bearer $FOURCOLORS_API_KEY"

# minimize -x0 - x1 - x2 + 2 x0 x1 + 2 x1 x2
JOB=$(curl -sf -H "$A" -H "Content-Type: application/json" -X POST "$B/v1/jobs" -d '{
  "solver": "optimizer.qubo",
  "input": {"qubo": {"linear": [-1, -1, -1], "quadratic": [[0, 1, 2], [1, 2, 2]]}},
  "options": {"time_limit_s": 2}
}' | jq -r .id)
echo "job $JOB"
curl -sfN -H "$A" "$B/v1/jobs/$JOB/events"          # live updates until "done"
curl -sf -H "$A" "$B/v1/jobs/$JOB" | jq '{status, billed_s, objective: .result.objective, solution: .result.solution_list}'
