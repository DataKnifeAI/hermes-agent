"""Readiness smoke for managed vLLM: ``GET /v1/models``.

Full tool-call bench (required ``calculator``) is a later phase. This module
persists the last probe so ``hermes doctor`` can show it.
"""

from __future__ import annotations

from contextlib import suppress
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def bench_path() -> Path:
    from hermes_cli.vllm_runtime.venv import runtimes_root

    return runtimes_root() / "bench.json"


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
            "elapsed_s": round(time.monotonic() - started, 3),
            "error": "",
        }
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        result = {
            "ok": False,
            "url": url,
            "models": [],
            "elapsed_s": round(time.monotonic() - started, 3),
            "error": str(exc),
        }
    write_bench_result(result)
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
