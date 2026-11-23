# HTTP template

Every client call is plain HTTPS with a bearer key. The full API is described by the OpenAPI document at
`https://api.4colors-research.com/openapi.json`; generate a client for your language from it, or call it directly:

| Step | Request |
|---|---|
| Your plan and limits | `GET /v1/license` |
| Submit a job | `POST /v1/jobs` with `{"solver": "optimizer.qubo" or "quantum.compiler", "input": ..., "options": ...}` |
| Follow it live | `GET /v1/jobs/{id}/events` (server-sent events: `incumbent`, `done`, ...) |
| Stop it, keep the best so far | `POST /v1/jobs/{id}/stop` |
| Read the record | `GET /v1/jobs/{id}` |

A runnable version: [`quickstart/http_hello.sh`](../../quickstart/http_hello.sh).
