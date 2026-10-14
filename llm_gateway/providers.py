"""LLM provider implementations behind a common interface.

The gateway talks to providers through the ``LLMProvider`` protocol, so a
real HTTP provider and the deterministic offline mock are interchangeable.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderError(Exception):
    """A permanent provider failure (do not retry)."""


class TransientProviderError(ProviderError):
    """A temporary provider failure (safe to retry)."""


@dataclass
class ChatResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> ChatResponse:
        ...


class MockProvider:
    """Deterministic offline provider.

    Produces a simple extractive echo of the conversation so the whole stack
    runs - and is testable - without network access or API keys.
    """

    def __init__(self, model: str = "mock-small") -> None:
        self.model = model
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> ChatResponse:
        self.calls.append(messages)
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        words = last_user.split()
        summary = " ".join(words[:24])
        content = (
            "[mock-completion] Received a {n}-message conversation. "
            "Last user message begins: \"{summary}\"".format(n=len(messages), summary=summary)
        )
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        completion_tokens = len(content.split())
        return ChatResponse(
            content=content,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class OpenAICompatibleProvider:
    """Minimal HTTP client for OpenAI-compatible /chat/completions APIs.

    Uses only the standard library. Retries are handled by the gateway, not
    here: this client raises TransientProviderError on timeouts and 5xx/429.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: float = 30.0) -> None:
        if not api_key:
            raise ProviderError("api_key is required for the openai-compatible provider")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> ChatResponse:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                raise TransientProviderError(f"provider HTTP {exc.code}") from exc
            raise ProviderError(f"provider HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TransientProviderError(f"provider unreachable: {exc}") from exc

        try:
            choice = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected provider response shape: {body!r:.200}") from exc
        return ChatResponse(
            content=choice,
            model=body.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            raw=body,
        )


def build_provider(config) -> LLMProvider:
    """Factory: pick a provider from config."""
    if config.provider == "mock":
        return MockProvider(model=config.model)
    if config.provider == "openai-compatible":
        return OpenAICompatibleProvider(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
        )
    raise ProviderError(f"unknown provider: {config.provider!r}")


def _monotonic() -> float:
    return time.monotonic()
