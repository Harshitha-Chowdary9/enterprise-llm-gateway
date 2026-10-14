import pytest

from llm_gateway.gateway import LLMGateway, ValidationError, validate_messages
from llm_gateway.providers import (
    ChatResponse,
    MockProvider,
    ProviderError,
    TransientProviderError,
)


def make_gateway(provider=None, **kwargs):
    kwargs.setdefault("sleep", lambda seconds: None)  # never really sleep in tests
    return LLMGateway(provider or MockProvider(), **kwargs)


class FlakyProvider:
    """Fails with a transient error a fixed number of times, then succeeds."""

    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def chat(self, messages, *, temperature=0.2):
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientProviderError("temporary outage")
        return ChatResponse(content="ok", model="flaky", prompt_tokens=1, completion_tokens=1)


class AlwaysDownProvider:
    def chat(self, messages, *, temperature=0.2):
        raise TransientProviderError("still down")


class BrokenProvider:
    def chat(self, messages, *, temperature=0.2):
        raise ProviderError("bad API key")


MESSAGES = [{"role": "user", "content": "Summarise our refund policy in two sentences."}]


class TestValidation:
    def test_accepts_well_formed_messages(self):
        cleaned = validate_messages(MESSAGES)
        assert cleaned == MESSAGES

    @pytest.mark.parametrize("bad", [
        None,
        [],
        "hello",
        [{"role": "user"}],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": "   "}],
        [{"role": "root", "content": "hi"}],
        [{"content": "hi"}],
        ["not-a-dict"],
        [{"role": "system", "content": "only system"}],
        [{"role": "user", "content": 42}],
    ])
    def test_rejects_malformed_messages(self, bad):
        with pytest.raises(ValidationError):
            validate_messages(bad)

    @pytest.mark.parametrize("bad_temp", [-0.1, 2.1, "0.5", True, None])
    def test_rejects_bad_temperature(self, bad_temp):
        gateway = make_gateway()
        with pytest.raises(ValidationError):
            gateway.chat(MESSAGES, temperature=bad_temp)


class TestRetries:
    def test_recovers_after_transient_failures(self):
        provider = FlakyProvider(failures=2)
        gateway = make_gateway(provider, max_retries=3)
        response = gateway.chat(MESSAGES)
        assert response.content == "ok"
        assert provider.calls == 3
        assert gateway.metrics.retries == 2

    def test_gives_up_after_max_retries(self):
        gateway = make_gateway(AlwaysDownProvider(), max_retries=2)
        with pytest.raises(TransientProviderError):
            gateway.chat(MESSAGES)
        assert gateway.metrics.retries == 1
        assert gateway.metrics.errors == 1

    def test_permanent_errors_are_not_retried(self):
        provider = BrokenProvider()
        gateway = make_gateway(provider, max_retries=5)
        with pytest.raises(ProviderError):
            gateway.chat(MESSAGES)
        assert gateway.metrics.retries == 0
        assert gateway.metrics.errors == 1


class TestCache:
    def test_identical_request_is_served_from_cache(self):
        provider = MockProvider()
        gateway = make_gateway(provider)
        first = gateway.chat(MESSAGES)
        second = gateway.chat(MESSAGES)
        assert first.content == second.content
        assert len(provider.calls) == 1
        assert gateway.metrics.cache_hits == 1

    def test_different_temperature_bypasses_cache(self):
        provider = MockProvider()
        gateway = make_gateway(provider)
        gateway.chat(MESSAGES, temperature=0.1)
        gateway.chat(MESSAGES, temperature=0.9)
        assert len(provider.calls) == 2

    def test_expired_entry_is_refetched(self):
        provider = MockProvider()
        now = [1000.0]
        gateway = make_gateway(provider, cache_ttl_seconds=10, clock=lambda: now[0])
        gateway.chat(MESSAGES)
        now[0] += 30  # past the TTL
        gateway.chat(MESSAGES)
        assert len(provider.calls) == 2
        assert gateway.metrics.cache_hits == 0


class TestMetrics:
    def test_token_and_request_counters(self):
        gateway = make_gateway()
        gateway.chat(MESSAGES)
        gateway.chat([{"role": "user", "content": "A different question about overdraft fees."}])
        snap = gateway.metrics.snapshot()
        assert snap["requests"] == 2
        assert snap["errors"] == 0
        assert snap["prompt_tokens"] > 0
        assert snap["completion_tokens"] > 0


class TestMockProvider:
    def test_deterministic_offline_response(self):
        provider = MockProvider()
        a = provider.chat(MESSAGES)
        b = provider.chat(MESSAGES)
        assert a.content == b.content
        assert "refund policy" in a.content

    def test_gateway_requires_positive_retries(self):
        with pytest.raises(ValueError):
            make_gateway(max_retries=0)
