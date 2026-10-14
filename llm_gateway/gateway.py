"""Gateway core: retries, caching, validation, metrics and structured logs."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from .providers import ChatResponse, LLMProvider, TransientProviderError

logger = logging.getLogger("llm_gateway")

ALLOWED_ROLES = {"system", "user", "assistant"}


class ValidationError(ValueError):
    """The request payload failed validation."""


def validate_messages(messages) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValidationError("messages must be a non-empty list")
    cleaned = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValidationError(f"messages[{i}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise ValidationError(f"messages[{i}].role must be one of {sorted(ALLOWED_ROLES)}")
        if not isinstance(content, str) or not content.strip():
            raise ValidationError(f"messages[{i}].content must be a non-empty string")
        cleaned.append({"role": role, "content": content})
    if not any(m["role"] == "user" for m in cleaned):
        raise ValidationError("at least one message must have role 'user'")
    return cleaned


@dataclass
class Metrics:
    requests: int = 0
    errors: int = 0
    cache_hits: int = 0
    retries: int = 0
    total_latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def snapshot(self) -> dict:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "avg_latency_seconds": (
                round(self.total_latency_seconds / self.requests, 6) if self.requests else 0.0
            ),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass
class _CacheEntry:
    response: ChatResponse
    stored_at: float


class LLMGateway:
    """Wraps a provider with retry/backoff, a TTL cache and metrics."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_retries: int = 3,
        cache_ttl_seconds: float = 300.0,
        backoff_base_seconds: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.provider = provider
        self.max_retries = max_retries
        self.cache_ttl_seconds = cache_ttl_seconds
        self.backoff_base_seconds = backoff_base_seconds
        self.sleep = sleep
        self.clock = clock
        self.metrics = Metrics()
        self._cache: dict[str, _CacheEntry] = {}

    def _cache_key(self, messages: list[dict[str, str]], temperature: float) -> str:
        blob = json.dumps({"messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def chat(self, messages, *, temperature: float = 0.2) -> ChatResponse:
        cleaned = validate_messages(messages)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValidationError("temperature must be a number")
        if not 0.0 <= float(temperature) <= 2.0:
            raise ValidationError("temperature must be between 0 and 2")

        self.metrics.requests += 1
        started = self.clock()
        key = self._cache_key(cleaned, float(temperature))

        entry = self._cache.get(key)
        if entry and (self.clock() - entry.stored_at) < self.cache_ttl_seconds:
            self.metrics.cache_hits += 1
            logger.info(json.dumps({"event": "cache_hit", "key": key[:12]}))
            return entry.response

        try:
            response = self._call_with_retry(cleaned, float(temperature))
        except Exception:
            self.metrics.errors += 1
            raise
        finally:
            self.metrics.total_latency_seconds += self.clock() - started

        self._cache[key] = _CacheEntry(response=response, stored_at=self.clock())
        self.metrics.prompt_tokens += response.prompt_tokens
        self.metrics.completion_tokens += response.completion_tokens
        logger.info(json.dumps({
            "event": "completion",
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        }))
        return response

    def _call_with_retry(self, messages: list[dict[str, str]], temperature: float) -> ChatResponse:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.provider.chat(messages, temperature=temperature)
            except TransientProviderError:
                if attempt >= self.max_retries:
                    raise
                self.metrics.retries += 1
                delay = self.backoff_base_seconds * (2 ** (attempt - 1))
                logger.warning(json.dumps({
                    "event": "retry", "attempt": attempt, "delay_seconds": delay,
                }))
                self.sleep(delay)
