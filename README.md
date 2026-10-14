# enterprise-llm-gateway

A small Python service that sits between an application and a Large Language
Model (LLM) API - the "gateway" pattern used to integrate LLMs into
enterprise software.

> **Learning project.** Built by Harshitha Chittineni with AI-assistant
> pairing as a hands-on study project for LLM integration patterns. It is not
> production software and handles no real data.

## What it does

- Exposes a JSON HTTP API (`POST /v1/chat`, `GET /health`, `GET /metrics`)
  using only the Python standard library.
- Talks to the LLM behind a **provider interface** (`LLMProvider`), so the
  rest of the code does not care which model or vendor is behind it.
- Ships two interchangeable providers:
  - `MockProvider` - deterministic, fully offline. **This is the default, so
    the service and all tests run with no API key and no network.**
  - `OpenAICompatibleProvider` - a minimal HTTP client for any
    OpenAI-compatible `/chat/completions` endpoint.
- Adds the cross-cutting concerns an enterprise integration needs:
  - request validation with clear 400 errors,
  - retries with exponential backoff for transient failures only,
  - a TTL response cache for identical requests,
  - usage/latency metrics on `/metrics`,
  - structured JSON logs.

## Run it (offline, no keys needed)

```bash
python -m llm_gateway.server          # listens on 127.0.0.1:8080
curl -X POST http://127.0.0.1:8080/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Summarise our refund policy."}]}'
curl http://127.0.0.1:8080/metrics
```

To point it at a real OpenAI-compatible API instead:

```bash
export LLM_PROVIDER=openai-compatible
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini          # any model your endpoint serves
python -m llm_gateway.server
```

All settings come from environment variables - see `llm_gateway/config.py`.

## Tests

```bash
pip install -r requirements.txt
python -m pytest
```

**32 tests, all passing** (run 2026-10-14, Python 3.10, pytest 9.1).
They cover validation, retry/give-up behaviour, permanent-vs-transient error
handling, cache hits/misses/expiry, metrics, and the live HTTP API (the tests
start a real server on an ephemeral port). The service was also exercised
manually with `curl` against a running instance (health, chat, cache,
metrics, and 400 handling verified).

## Docker

A `Dockerfile` is included to practise containerising a Python service.
**It has not been built or run** - there was no Docker runtime in the
development environment - so treat it as configuration, not a verified
artifact. The verified way to run this project is `python -m
llm_gateway.server` as above.

## What this project deliberately does not claim

- No cloud deployment (no Azure/AWS/GCP), no Kubernetes, no CI pipeline.
- The `OpenAICompatibleProvider` HTTP path is written and unit-testable via
  fakes, but was not exercised against a paid live API; the mock provider is
  the verified runtime path.
- No streaming, no auth on the gateway itself, no rate limiting.

See `STUDY_GUIDE.md` for how to explain every design decision in an interview.
