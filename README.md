# Autonomous PR Reviewer

A multi-agent service that reviews GitHub pull requests automatically. When a PR
is opened or updated, it fetches the diff, runs a few specialized agents over it
in parallel (diff analysis, test suggestions, security audit), and posts a
single prioritized review comment back on the PR.

It's built as a LangGraph state graph rather than one big prompt loop. A
coordinator decides which agents to run, the specialists run concurrently and
return validated Pydantic objects, and a summarizer merges their findings into
one Markdown comment. The agents call the real GitHub API and work off real
diffs.

Tech stack: Python, FastAPI, LangGraph, Pydantic, Redis, Docker, Langfuse, pytest.

## Features

- Multi-agent graph: a coordinator, three parallel specialists, and a fan-in summarizer.
- Conditional routing. The coordinator skips the security and test agents on docs-only PRs.
- Structured output throughout. Every finding is a validated Pydantic model, so there's no regex parsing of model text.
- Per-agent error isolation. If one agent or its LLM call fails, the review still gets posted with that section marked unavailable.
- Guardrails: secret redaction and prompt-injection neutralization on input, prompt-leak stripping and length caps on output.
- Observability: Langfuse tracing (one trace per PR, a span per agent), per-PR token and cost accounting, and a `/stats` endpoint.
- Eval harness: 12 fixture PRs and a golden set that scores security recall, false-positive rate, and LLM-as-judge faithfulness.

See [architecture.md](architecture.md) for the graph diagram, state flow, and
failure handling.

## Architecture

```mermaid
flowchart LR
    W([PR webhook]) --> C[coordinator]
    C --> D[diff_analyzer]
    C --> T[test_suggester]
    C --> S[security_auditor]
    D --> SUM[summarizer]
    T --> SUM
    S --> SUM
    SUM --> P([post review comment])
```

- Coordinator: fetches PR data, sets `agents_to_run`, drives the conditional fan-out.
- Diff analyzer: summarizes intent and flags risky changes per file.
- Test suggester: proposes missing test cases, focused on edge and error paths.
- Security auditor: looks for injection, hardcoded secrets, unsafe deserialization, path traversal, SSRF, weak crypto, and missing validation.
- Summarizer: writes one prioritized review, ordered security, then correctness, then tests.

## Getting started

### Prerequisites

- Python 3.12. LangGraph and LangChain don't play well with the newest Python releases yet, so stick to 3.12.
- Redis for request dedup, or just use the Docker Compose setup below.
- A GitHub token with `repo` scope (or a fine-grained token with Pull requests read/write and Contents read).
- An LLM API key (Anthropic or Groq).

### Configure

```bash
cp .env.example .env
# then fill in GITHUB_TOKEN, GITHUB_WEBHOOK_SECRET, LLM_API_KEY, etc.
```

### Run locally

```bash
# developed with uv:
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# or plain pip:
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

uvicorn app.main:app --reload
curl localhost:8000/health   # {"status":"ok","version":"0.1.0"}
curl localhost:8000/stats    # aggregate review metrics
```

### Run with Docker (app + Redis)

```bash
docker compose up --build
# app on http://localhost:8000, redis on :6379
```

### Run the tests

```bash
pytest -q   # unit + integration, fully mocked, no API key needed
```

### Run the evals

```bash
python evals/run_evals.py   # needs a real LLM_API_KEY for live numbers
```

## GitHub webhook

Point a repo webhook at `https://<your-host>/webhook/github`:

- Payload URL: `https://<your-host>/webhook/github`
- Content type: `application/json`
- Secret: the same value as `GITHUB_WEBHOOK_SECRET`
- Events: Pull requests (the app acts on `opened` and `synchronize`)

Requests with an invalid signature are rejected with 401. Each head SHA is
reviewed only once (Redis dedup). The endpoint returns 200 right away and runs
the review in the background so GitHub doesn't time out.

## Evals

`python evals/run_evals.py` scores the agents against 12 fixture PRs: 8 with
planted vulnerabilities, 3 clean, and 1 with a correctness bug.

Latest run (`llama-3.3-70b-versatile` via Groq):

```
Security recall:      7/8  (87.5%)
False positives:      0 findings on 4 clean fixtures
Diff analysis (judge): 2.5/5
```

The provider is switchable via `LLM_PROVIDER` (`groq` for free local dev,
`anthropic` for higher quality). Recall and false positives hold up even on an
open model; the judge score depends on which model runs the judge, and goes up
with a stronger one.

The scoring code (recall, false positives, aggregation) is unit-tested
independently of the LLM in [tests/test_evals.py](tests/test_evals.py).

## Design notes

| Choice | Reason |
|---|---|
| LangGraph | Explicit control flow: conditional edges, parallel fan-out, reducers, instead of an opaque agent loop. The graph decides which agents run. |
| Structured output (`with_structured_output`) | Findings are validated Pydantic models, not parsed free text. Easier to test and harder to break. |
| Reducers on shared state | Parallel agents write `errors` and `token_usage` at the same time; reducers merge those instead of clobbering. |
| Per-agent try/except | One agent failing shouldn't crash the whole review, so each degrades with a note. |
| Guardrails as a layer | Secrets never reach the model or the comment, injected instructions are treated as data, and oversized diffs are capped. |
| Langfuse + `/stats` | Every run is traced and costed, so "how much does a review cost" has a real answer. |
| Configurable provider | Runs on Anthropic or Groq via `LLM_PROVIDER` / `LLM_MODEL`. Adaptive-thinking models don't get sampling params. |
| httpx GitHub client | Full control over rate limits, retries, and typed errors. |

### Project layout

```
app/
  main.py            FastAPI: /health, /stats, /webhook/github
  config.py          Pydantic settings (fails loud on missing env)
  github/            client.py (async, retries, typed errors) + webhook.py (HMAC)
  agents/            state.py, graph.py, coordinator + 3 specialists + summarizer
  tools/             GitHub tools as LangChain @tools ({ok, data, error})
  guardrails/        input_guard.py, output_guard.py
  observability/     tracing.py (Langfuse), cost.py, stats.py
  review.py          fetch PR -> run graph -> post -> trace/cost/stats
evals/               fixtures/, golden_set.json, scoring.py, judge.py, run_evals.py
tests/               github client, graph, agents, guardrails, webhook, evals, ...
```

## License

MIT
