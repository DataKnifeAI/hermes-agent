"""Endpoint resolution for managed vLLM (provider integration)."""

from __future__ import annotations

from contextlib import suppress
import json
import logging

logger = logging.getLogger(__name__)


def _pid_alive(pid: int) -> bool:
    if not pid or pid < 0:
        return False
    with suppress(Exception):
        import psutil  # type: ignore

        return psutil.pid_exists(pid)
    return True


def _state_endpoint() -> dict | None:
    from hermes_cli.vllm_runtime.supervisor import state_path

    path = state_path()
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    base_url = state.get("base_url", "")
    if not base_url:
        return None
    if not _pid_alive(int(state.get("pid") or 0)):
        return None
    return {"base_url": base_url, "pid": state.get("pid")}


def resolve_vllm_endpoint() -> dict | None:
    """Managed vLLM base_url from the supervisor state file, or None."""
    with suppress(Exception):
        return _state_endpoint()
    return None
