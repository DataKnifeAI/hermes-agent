"""Tool-call bench for managed vLLM — fake HTTP, no live GPU."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

import pytest

from hermes_cli.vllm_runtime.bench import (
    probe_models, probe_tool_calls, response_has_tool_calls, verify_tool_calls,
)


def test_response_has_tool_calls_contract():
    assert response_has_tool_calls({
        "choices": [{"message": {"tool_calls": [{"id": "c1", "function": {"name": "calculator"}}]}}],
    })
    assert not response_has_tool_calls({
        "choices": [{"message": {"content": "7+8=15", "tool_calls": []}}],
    })
    assert not response_has_tool_calls({"choices": [{"message": {"content": "hi"}}]})
    assert not response_has_tool_calls({"object": "list", "data": [{"id": "hermes3:8b"}]})


class _ModelsOnly(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            body = b'{"object":"list","data":[{"id":"hermes3:8b"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = b'{"choices":[{"message":{"content":"15"}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _ToolCallOk(_ModelsOnly):
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        payload = json.loads(raw or b"{}")
        assert payload.get("tool_choice") == "required"
        tools = payload.get("tools") or []
        assert any(
            (t.get("function") or {}).get("name") == "calculator"
            for t in tools if isinstance(t, dict)
        )
        body = json.dumps({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"expression":"7+8"}'},
                    }],
                },
            }],
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(handler_cls):
    httpd = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_bench_fails_when_models_ok_but_no_tool_calls(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)

    httpd = _serve(_ModelsOnly)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        models = probe_models(url)
        assert models["ok"] is True
        assert "hermes3:8b" in models["models"]
        result = probe_tool_calls(url)
        assert result["ok"] is False
        assert result["tool_calls"] is False
        with pytest.raises(RuntimeError, match="tool_calls"):
            verify_tool_calls(url)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_bench_passes_when_response_has_tool_calls(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)

    httpd = _serve(_ToolCallOk)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        result = verify_tool_calls(url, model="hermes3:8b")
        assert result["ok"] is True
        assert result["tool_calls"] is True
        from hermes_cli.vllm_runtime.bench import last_bench

        saved = last_bench()
        assert saved is not None
        assert saved["ok"] is True
        assert saved["tool_calls"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
