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


def _state_endpoint(device: str | None = None, config: dict | None = None) -> dict | None:
    from hermes_cli.local_engines import vllm_device_from_config
    from hermes_cli.vllm_runtime.supervisor import state_path

    path = state_path(device if device is not None else vllm_device_from_config(config))
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


def _models_ready(endpoint: dict | None) -> bool:
    """True when GET /v1/models on this serve returned a model id.

    ``server.json`` is written at spawn, before CUDA/CPU warmup. Chat must
    not treat that file as ready.
    """
    if not endpoint:
        return False
    from hermes_cli.vllm_runtime.supervisor import probe_served_model_name

    return bool(probe_served_model_name(str(endpoint.get("base_url") or "")))


def resolve_vllm_endpoint(config: dict | None = None,
                          wait_for_boot_s: float = 8.0) -> dict | None:
    """Managed-first endpoint for the selected vLLM device.

    Status/doctor pass ``wait_for_boot_s=0`` so a poll does not spawn. Chat
    passes the supervisor warmup budget and, when ``local_runtime.enabled``
    is on and this engine is vLLM, kicks only that device. Readiness is
    GET /v1/models, not the spawn-time state file. An endpoint that is
    already up is returned as-is so a stale pin can follow it.
    """
    from hermes_cli.local_engines import vllm_device_from_config

    device = vllm_device_from_config(config)
    managed = _state_endpoint(device, config)
    if managed or wait_for_boot_s <= 0:
        return managed

    if not _boot_in_flight(config):
        return None

    from hermes_cli.vllm_runtime.device import CPU
    from hermes_cli.vllm_runtime.occupancy import require_gpu_free

    if device != CPU:
        require_gpu_free()
    _kick_managed_boot(config)
    deadline = time.monotonic() + wait_for_boot_s
    while time.monotonic() < deadline:
        time.sleep(0.25)
        managed = _state_endpoint(device, config)
        if managed and _models_ready(managed):
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

        from hermes_cli.vllm_runtime.device import is_vllm_engine

        return is_vllm_engine(engine_from_config(config))
    return False


def _device_for_managed_pin(url: str) -> str | None:
    """gpu / cpu when *url* is that device's managed loopback, else None.

    Default ports win. An ephemeral pin matches the live state of one device
    only — never the sibling — so a GPU chat does not follow CPU 4B.
    """
    from hermes_cli.vllm_runtime.device import (
        CPU, CPU_ENDPOINT_NAME, GPU, GPU_ENDPOINT_NAME,
        managed_endpoint_name_for_url,
    )

    label = managed_endpoint_name_for_url(url)
    if label == GPU_ENDPOINT_NAME:
        return GPU
    if label == CPU_ENDPOINT_NAME:
        return CPU
    return None


def follow_live_managed_vllm(
    base_url: str,
    model: str = "",
    config: dict | None = None,
) -> dict | None:
    """Rewrite a stale loopback pin to the live serve of that same device.

    Sessions persist ``provider: custom`` + the port from first Use. After an
    ephemeral rebind (18435 busy → 53351) that pin connection-refuses. Follow
    that device's ``server.json`` when the pin is loopback, not llama.cpp/Ollama.
    Never adopt the sibling (GPU 14B pin must not become CPU 4B just because
    the CPU serve is healthy or ``local_runtime.vllm.device`` flipped).
    """
    pinned = (base_url or "").strip().rstrip("/")
    if not pinned or not is_loopback_url(pinned):
        return None
    port = _url_port(pinned)
    if port in _FOREIGN_LOOPBACK_PORTS:
        return None
    pin_device = _device_for_managed_pin(pinned)
    if pin_device is None:
        if not _engine_is_vllm(config):
            return None
        live = resolve_vllm_endpoint(config, wait_for_boot_s=0)
    else:
        live = _state_endpoint(pin_device, config)
    if not live:
        return None
    live_url = str(live.get("base_url") or "").strip().rstrip("/")
    if not live_url:
        return None
    served = str(live.get("served_model_name") or "").strip()
    if not served:
        from hermes_cli.vllm_runtime.supervisor import state_served_model_name

        served = state_served_model_name(pin_device or "gpu")
    if pinned.lower() == live_url.lower() and (not served or served == (model or "").strip()):
        return None
    logger.info(
        "follow_live_managed_vllm: pin %s model=%s -> %s served=%s device=%s",
        pinned, model or "", live_url, served, pin_device or "selected")
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

        from hermes_cli.vllm_runtime.device import is_vllm_engine

        if not is_vllm_engine(engine_from_config(config)):
            return False
        from hermes_cli.local_engines import vllm_device_from_config
        from hermes_cli.vllm_runtime.supervisor import (
            configured_cache_missing, configured_model_id, vllm_settings)
        from hermes_cli.vllm_runtime.venv import venv_ready

        settings = vllm_settings(config)
        if not configured_model_id(settings) or configured_cache_missing(settings):
            return False
        return venv_ready(vllm_device_from_config(config))
    return False
