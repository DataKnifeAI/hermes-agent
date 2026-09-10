"""Endpoint resolution for managed vLLM (provider integration).

``provider: vllm`` with no explicit (or a loopback) base_url resolves to the
supervised server. When the engine is enabled and installed, a missing state
file kicks ``ensure_managed_engine`` the way llama.cpp's ``_kick_managed_boot``
does — occupancy is not swallowed.
"""

from __future__ import annotations

from contextlib import suppress
import json
import logging
import threading
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})
_KICK_LOCK = threading.Lock()


def _pid_alive(pid: int) -> bool:
    if not pid or pid < 0:
        return False
    with suppress(Exception):
        import psutil  # type: ignore

        return psutil.pid_exists(pid)
    return True


def is_loopback_url(url: str) -> bool:
    """True when *url* is empty or a loopback OpenAI base — managed boot may own it."""
    text = (url or "").strip()
    if not text:
        return True
    try:
        host = (urlparse(text).hostname or "").lower()
    except ValueError:
        return False
    if host in _LOOPBACK_HOSTS:
        return True
    return host.startswith("127.")


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
    return {
        "base_url": base_url,
        "api_key": state.get("api_key", ""),
        "pid": state.get("pid"),
        "served_model_name": str(state.get("served_model_name") or "").strip(),
    }


def resolve_vllm_endpoint(config: dict | None = None,
                          wait_for_boot_s: float = 8.0) -> dict | None:
    """Managed-first endpoint for ``provider: vllm``.

    Status/doctor pass ``wait_for_boot_s=0`` so a poll does not spawn. Chat and
    gateway resolution use the default wait and kick an on-demand boot when the
    engine is enabled and the isolated venv is installed.
    """
    managed = _state_endpoint()
    if managed:
        return managed

    if wait_for_boot_s > 0 and _boot_in_flight(config):
        from hermes_cli.vllm_runtime.occupancy import require_gpu_free

        require_gpu_free()
        _kick_managed_boot(config)
        deadline = time.monotonic() + wait_for_boot_s
        while time.monotonic() < deadline:
            time.sleep(0.25)
            managed = _state_endpoint()
            if managed:
                return managed
    return None


# llama.cpp + Ollama defaults — a session pinned there is not managed vLLM.
_FOREIGN_LOOPBACK_PORTS = frozenset({18434, 11434})


def _url_port(url: str) -> int | None:
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return None
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return None


def _engine_is_vllm(config: dict | None = None) -> bool:
    with suppress(Exception):
        config = _load_config_if_none(config)
        from hermes_cli.local_engines import engine_from_config

        return engine_from_config(config) == "vllm"
    return False


def follow_live_managed_vllm(
    base_url: str,
    model: str = "",
    config: dict | None = None,
) -> dict | None:
    """Rewrite a stale loopback pin to the live managed serve.

    Sessions persist ``provider: custom`` + the port from first Use. After an
    ephemeral rebind (18435 busy → 53351) that pin connection-refuses. Follow
    ``server.json`` when the pin is loopback, not llama.cpp/Ollama, and either
    the default managed port or the active engine is vLLM.
    """
    pinned = (base_url or "").strip().rstrip("/")
    if not pinned or not is_loopback_url(pinned):
        return None
    live = resolve_vllm_endpoint(config, wait_for_boot_s=0)
    if not live:
        return None
    live_url = str(live.get("base_url") or "").strip().rstrip("/")
    if not live_url:
        return None
    port = _url_port(pinned)
    if port in _FOREIGN_LOOPBACK_PORTS:
        return None
    from hermes_cli.vllm_runtime.supervisor import DEFAULT_LISTEN_PORT

    if port != DEFAULT_LISTEN_PORT and not _engine_is_vllm(config):
        return None
    served = str(live.get("served_model_name") or "").strip()
    if not served:
        from hermes_cli.vllm_runtime.supervisor import state_served_model_name

        served = state_served_model_name()
    if pinned.lower() == live_url.lower() and (not served or served == (model or "").strip()):
        return None
    return {
        "base_url": live_url,
        "api_key": live.get("api_key") or "",
        "served_model_name": served,
        "pid": live.get("pid"),
    }


def _load_config_if_none(config: dict | None) -> dict | None:
    if config is not None:
        return config
    from hermes_cli.config import load_config

    return load_config()


def _kick_managed_boot(config: dict | None) -> None:
    """Start the managed vLLM server when resolution finds it missing."""
    if not _KICK_LOCK.acquire(blocking=False):
        return

    def _boot() -> None:
        try:
            from hermes_cli.local_engines import ensure_managed_engine

            ensure_managed_engine(_load_config_if_none(config))
        except Exception:  # noqa: BLE001 — waiter surfaces occupancy / off
            logger.warning("on-demand managed vLLM boot failed", exc_info=True)
        finally:
            _KICK_LOCK.release()

    threading.Thread(target=_boot, daemon=True,
                     name="vllm-on-demand-boot").start()


def _boot_in_flight(config: dict | None) -> bool:
    """True when managed vLLM is the enabled engine and the isolated venv exists."""
    with suppress(Exception):
        config = _load_config_if_none(config)
        section = (config or {}).get("local_runtime") or {}
        if not section.get("enabled"):
            return False
        from hermes_cli.local_engines import engine_from_config

        if engine_from_config(config) != "vllm":
            return False
        from hermes_cli.vllm_runtime.supervisor import (
            configured_cache_missing, configured_model_id, vllm_settings)
        from hermes_cli.vllm_runtime.venv import venv_ready

        settings = vllm_settings(config)
        if not configured_model_id(settings) or configured_cache_missing(settings):
            return False
        return venv_ready()
    return False
