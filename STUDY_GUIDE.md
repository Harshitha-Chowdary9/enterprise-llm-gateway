# Study guide - enterprise-llm-gateway

Run the service and tests yourself before any interview. Thirty minutes:

1. `python -m llm_gateway.server` in one terminal.
2. In another: the three `curl` commands from the README. Call the same chat
   request twice, then open `/metrics` and notice `cache_hits: 1`.
3. `python -m pytest -v` and read the test names - they are the specification.

## Questions you should be able to answer

- **Why put a gateway in front of the LLM instead of calling the API
  directly?** One place for validation, retries, caching, metrics and vendor
  swap. In an enterprise, many apps share these concerns; duplicating them in
  every app is how you get inconsistent behaviour and runaway costs.
- **Why a provider interface?** Dependency inversion. The gateway depends on
  the `LLMProvider` protocol, not on OpenAI. Swapping vendors - or the mock -
  is a config change, not a code change. This is also what makes the system
  testable offline.
- **Why retry only transient errors?** A 401/bad-request will fail forever;
  retrying it wastes money and hides bugs. A 429/5xx/timeout may succeed
  later, so the gateway retries with exponential backoff (`0.25s, 0.5s, 1s`)
  and gives up after `max_retries`.
- **Why cache?** Identical prompts are common (FAQ-style traffic). The cache
  key is a SHA-256 of messages + temperature; entries expire after a TTL so
  answers cannot go stale forever.
- **What would you add before production?** Authn/authz on the gateway,
  per-consumer rate limits, streaming responses, async I/O, a real cache
  (Redis), secrets management, CI, and deploy to a cloud runtime. Know that
  this project stops before those on purpose.

## Skills this project genuinely demonstrates

- Python 3 (type hints, dataclasses, protocols, stdlib `http.server`,
  `urllib`, `hashlib`, `json`, `logging`).
- Designing an HTTP JSON API and validating untrusted input.
- LLM integration pattern: provider abstraction, offline mock mode, retries
  with exponential backoff, response caching, token-usage metrics.
- Testing with pytest: parametrized validation cases, fake providers, and
  end-to-end HTTP tests against a live server socket.
