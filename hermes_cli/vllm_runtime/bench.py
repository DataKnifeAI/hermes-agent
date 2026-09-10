"""Readiness smoke for managed vLLM: ``GET /v1/models`` then a required tool call.

A server that lists models but never emits ``tool_calls`` still breaks every
Hermes tool-using session. The calculator round-trip is the contract; last
result is persisted for ``hermes doctor``.
"""

from __future__ import annotations

from contextlib import suppress
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


_CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a simple arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression such as 7+8",
                },
            },
            "required": ["expression"],
        },
    },
}


def bench_path() -> Path:
    from hermes_cli.vllm_runtime.venv import runtimes_root

    return runtimes_root() / "bench.json"


def response_has_tool_calls(body: object) -> bool:
    """True when an OpenAI-compatible chat completion contains ``tool_calls``."""
    if not isinstance(body, dict):
        return False
    if body.get("tool_calls"):
        return True
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    message = first.get("message") if isinstance(first.get("message"), dict) else first
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    return bool(calls)


def probe_models(base_url: str, timeout_s: float = 3.0) -> dict:
    """Hit ``{base_url}/models``. *base_url* is the OpenAI root ending in ``/v1``."""
    url = str(base_url).rstrip("/") + "/models"
    started = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            body = json.loads(resp.read() or b"{}")
        ids: list[str] = []
        data = body.get("data") if isinstance(body, dict) else None
        if isinstance(data, list):
            ids = [str(row.get("id") or "") for row in data if isinstance(row, dict) and row.get("id")]
        result = {
            "ok": True,
            "url": url,
            "models": ids,
            "tool_calls": False,
            "elapsed_s": round(time.monotonic() - started, 3),
            "error": "",
        }
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        result = {
            "ok": False,
            "url": url,
            "models": [],
            "tool_calls": False,
            "elapsed_s": round(time.monotonic() - started, 3),
            "error": str(exc),
        }
    write_bench_result(result)
    return result


def probe_tool_calls(base_url: str, *, model: str = "", timeout_s: float = 30.0) -> dict:
    """POST a ``tool_choice=required`` calculator round-trip. Succeeds only on ``tool_calls``."""
    url = str(base_url).rstrip("/") + "/chat/completions"
    started = time.monotonic()
    payload = {
        "model": model or "hermes3:8b",
        "messages": [{"role": "user", "content": "What is 7 plus 8? Use the calculator tool."}],
        "tools": [_CALCULATOR_TOOL],
        "tool_choice": "required",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read() or b"{}")
        ok = response_has_tool_calls(body)
        result = {
            "ok": ok,
            "url": url,
            "models": [model] if model else [],
            "tool_calls": ok,
            "elapsed_s": round(time.monotonic() - started, 3),
            "error": "" if ok else "response had no tool_calls (hermes parser silent?)",
        }
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        result = {
            "ok": False,
            "url": url,
            "models": [model] if model else [],
            "tool_calls": False,
            "elapsed_s": round(time.monotonic() - started, 3),
            "error": str(exc),
        }
    write_bench_result(result)
    return result


def verify_tool_calls(base_url: str, *, model: str = "") -> dict:
    """Models preflight then required-tool smoke. Raises if the parser is silent."""
    models = probe_models(base_url)
    if not models["ok"]:
        raise RuntimeError(models.get("error") or "GET /v1/models failed")
    result = probe_tool_calls(base_url, model=model or (models["models"][0] if models["models"] else ""))
    if not result["ok"]:
        raise RuntimeError(result.get("error") or "vLLM tool-call bench failed")
    return result


def write_bench_result(payload: dict) -> None:
    path = bench_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def last_bench() -> dict | None:
    path = bench_path()
    if not path.exists():
        return None
    with suppress(json.JSONDecodeError, OSError, TypeError):
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    return None
