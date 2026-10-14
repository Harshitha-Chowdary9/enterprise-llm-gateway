"""HTTP JSON API for the gateway (standard library only).

Endpoints:
  GET  /health   -> {"status": "ok"}
  GET  /metrics  -> gateway metrics snapshot
  POST /v1/chat  -> {"messages": [...], "temperature": 0.2}
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import GatewayConfig
from .gateway import LLMGateway, ValidationError
from .providers import ProviderError, build_provider


def make_server(gateway: LLMGateway, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LLMGateway/0.1"

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # keep test output clean
            pass

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            elif self.path == "/metrics":
                self._send_json(200, gateway.metrics.snapshot())
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except (ValueError, json.JSONDecodeError):
                self._send_json(400, {"error": "request body must be valid JSON"})
                return
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "request body must be a JSON object"})
                return
            try:
                response = gateway.chat(
                    payload.get("messages"),
                    temperature=payload.get("temperature", 0.2),
                )
            except ValidationError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except ProviderError as exc:
                self._send_json(502, {"error": f"upstream provider error: {exc}"})
                return
            self._send_json(200, {
                "content": response.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                },
            })

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    config = GatewayConfig.from_env()
    gateway = LLMGateway(
        build_provider(config),
        max_retries=config.max_retries,
        cache_ttl_seconds=config.cache_ttl_seconds,
    )
    server = make_server(gateway, config.host, config.port)
    print(f"llm-gateway listening on http://{config.host}:{config.port} (provider={config.provider})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
