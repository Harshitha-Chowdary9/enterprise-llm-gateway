"""Configuration for the LLM gateway, loaded from environment variables.

The service runs fully offline by default (provider="mock") so it can be
developed and tested without API keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GatewayConfig:
    provider: str = "mock"          # "mock" or "openai-compatible"
    model: str = "mock-small"
    api_key: str = ""               # only needed for a real provider
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    cache_ttl_seconds: float = 300.0
    host: str = "127.0.0.1"
    port: int = 8080

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        return cls(
            provider=os.environ.get("LLM_PROVIDER", "mock"),
            model=os.environ.get("LLM_MODEL", "mock-small"),
            api_key=os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.environ.get("LLM_MAX_RETRIES", "3")),
            cache_ttl_seconds=float(os.environ.get("LLM_CACHE_TTL_SECONDS", "300")),
            host=os.environ.get("GATEWAY_HOST", "127.0.0.1"),
            port=int(os.environ.get("GATEWAY_PORT", "8080")),
        )
