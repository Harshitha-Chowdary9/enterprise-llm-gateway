import json
import threading
import urllib.error
import urllib.request

import pytest

from llm_gateway.gateway import LLMGateway
from llm_gateway.providers import MockProvider
from llm_gateway.server import make_server


@pytest.fixture()
def running_server():
    gateway = LLMGateway(MockProvider(), sleep=lambda s: None)
    server = make_server(gateway, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield f"http://{host}:{port}", gateway
    server.shutdown()
    thread.join(timeout=5)


def post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


class TestHttpApi:
    def test_health(self, running_server):
        base, _ = running_server
        status, body = get_json(base + "/health")
        assert status == 200
        assert body == {"status": "ok"}

    def test_chat_roundtrip_offline(self, running_server):
        base, _ = running_server
        status, body = post_json(base + "/v1/chat", {
            "messages": [{"role": "user", "content": "Explain SEPA transfers briefly."}],
        })
        assert status == 200
        assert "SEPA" in body["content"]
        assert body["usage"]["prompt_tokens"] > 0

    def test_chat_validation_error_is_400(self, running_server):
        base, _ = running_server
        status, body = post_json(base + "/v1/chat", {"messages": [{"role": "user"}]})
        assert status == 400
        assert "error" in body

    def test_non_object_body_is_400(self, running_server):
        base, _ = running_server
        status, _ = post_json(base + "/v1/chat", [1, 2, 3])
        assert status == 400

    def test_unknown_route_is_404(self, running_server):
        base, _ = running_server
        try:
            get_json(base + "/nope")
            assert False, "expected 404"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404

    def test_metrics_endpoint_counts_requests(self, running_server):
        base, gateway = running_server
        post_json(base + "/v1/chat", {"messages": [{"role": "user", "content": "hi there"}]})
        status, body = get_json(base + "/metrics")
        assert status == 200
        assert body["requests"] == 1
        assert gateway.metrics.requests == 1
