"""Bootstrap for the managed vLLM engine: config -> isolated venv -> supervised serve.

``ensure_vllm_runtime(config)`` is safe at session start: disabled -> no-op.
When enabled, Hermes creates the venv and pip-installs vLLM if needed — the user
does not run pip, uv, or a third-party installer. ``OccupyingLlmError`` is never
swallowed (chat / serve must not silent-fall to a cloud provider). Other boot
failures still log and return None.
"""

from __future__ import annotations

from contextlib import suppress
import logging
import threading

from hermes_cli.vllm_runtime.endpoint import is_loopback_url as _is_loopback_url

logger = logging.getLogger(__name__)

# One supervisor per device. GPU and CPU vLLM run at the same time; replacing
# a model stops only that device's process.
_SUPERVISORS: dict[str, object | None] = {"gpu": None, "cpu": None}
# Serialize spawn per device. Two starts of the same device must not share a
# port. The other device has its own lock and is not stopped.
_START_LOCKS = {"gpu": threading.Lock(), "cpu": threading.Lock()}


def ensure_vllm_runtime(config: dict | None = None, force: bool = False,
                        *, executable=None, timeout_s: int = 1800):
    """Idempotent boot of managed vLLM. Returns the supervisor or None.

    First start downloads HF weights inside ``vllm serve``, so the default
    ready-timeout is long. Tests inject a fake executable and a short timeout.
    """
    section = (config or {}).get("local_runtime") or {}
    if not force and not section.get("enabled"):
        return None

    from hermes_cli.local_engines import vllm_device_from_config
    from hermes_cli.vllm_runtime.device import normalize_device

    device = normalize_device(vllm_device_from_config(config))
    with _START_LOCKS[device]:
        return _ensure_vllm_runtime_locked(
            config, force=force, executable=executable, timeout_s=timeout_s)


def _ensure_vllm_runtime_locked(config: dict | None, *, force: bool,
                                executable, timeout_s: int):
    from pathlib import Path

    from hermes_cli.local_engines import vllm_device_from_config
    from hermes_cli.vllm_runtime.device import CPU, normalize_device
    from hermes_cli.vllm_runtime.endpoint import _state_endpoint
    from hermes_cli.vllm_runtime.occupancy import OccupyingLlmError, require_gpu_free
    from hermes_cli.vllm_runtime.supervisor import (
        MODEL_REMOVED_MSG, VllmSupervisor, configured_cache_missing,
        configured_model_id, configured_unservable_reason, disable_auto_start,
        state_served_model_name, vllm_settings, write_last_error)
    from hermes_cli.vllm_runtime.venv import ensure_vllm_venv

    device = normalize_device(vllm_device_from_config(config))
    settings = vllm_settings(config)
    wanted = str(settings.get("served_model_name") or "").strip()
    current = _SUPERVISORS.get(device)
    if current is not None:
        got = str(getattr(current, "settings", {}).get("served_model_name") or "").strip()
        if not wanted or not got or wanted == got:
            return current
        current.stop()
        _SUPERVISORS[device] = None
    state = _state_endpoint(device)
    if state is not None:
        got = state_served_model_name(device)
        if not wanted or not got or wanted == got:
            logger.info("managed vLLM already running (another process)")
            if device == CPU:
                _bind_cpu_compression(state.get("base_url"), state.get("served_model_name"))
            return None
        # Leftover model still resident on THIS device — replace it.
        # The sibling vLLM server is not this state file.
        from hermes_cli.local_engines import stop_state_pid
        from hermes_cli.vllm_runtime.supervisor import state_path

        stop_state_pid(state_path(device))

    if device != CPU:
        require_gpu_free()

    if not configured_model_id(settings):
        logger.warning("managed vLLM has no configured model — not starting")
        return None
    blocked = configured_unservable_reason(settings)
    if blocked:
        write_last_error(blocked, device)
        disable_auto_start()
        logger.warning("%s", blocked)
        return None
    if configured_cache_missing(settings):
        write_last_error(MODEL_REMOVED_MSG, device)
        disable_auto_start()
        logger.warning("%s", MODEL_REMOVED_MSG)
        return None
    if executable is not None:
        exe_path = Path(executable)
    else:
        try:
            exe_path = ensure_vllm_venv(str(settings.get("python") or ""), device=device)
        except OccupyingLlmError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("vLLM venv install failed: %s", exc)
            return None
    if not exe_path.is_file():
        logger.warning("vLLM executable missing: %s", exe_path)
        return None

    sup = None
    try:
        sup = VllmSupervisor(settings, executable=exe_path, device=device)
        # Visible to stop before wait_ready returns, so Restore can abort an
        # in-flight boot of THIS device instead of SIGKILL via the state pid.
        _SUPERVISORS[device] = sup
        sup.start(timeout_s=timeout_s)
        if device == CPU:
            _bind_cpu_compression(sup.base_url, settings.get("served_model_name"))
        return sup
    except OccupyingLlmError:
        if _SUPERVISORS.get(device) is sup:
            _SUPERVISORS[device] = None
        raise
    except Exception as exc:  # noqa: BLE001 — never break session start
        if _SUPERVISORS.get(device) is sup and sup is not None:
            with suppress(Exception):
                sup.stop()
            _SUPERVISORS[device] = None
        logger.warning("managed vLLM runtime unavailable: %s", exc)
        if not (sup is not None and sup._stopping):
            disable_auto_start()
        return None


def _bind_cpu_compression(base_url, model) -> None:
    try:
        from hermes_cli.local_engines import maybe_bind_cpu_compression

        maybe_bind_cpu_compression(str(base_url or ""), str(model or ""))
    except Exception as exc:  # noqa: BLE001 — serve is up; compression pin is best-effort
        logger.warning("CPU vLLM compression base_url was not written: %s", exc)


def shutdown_vllm_runtime(device: str | None = None) -> None:
    """Stop one device, or both when ``device`` is omitted.

    Must not take the start lock — Restore stops an in-flight wait_ready.
    """
    from hermes_cli.vllm_runtime.device import CPU, GPU, normalize_device

    devices = (GPU, CPU) if device is None else (normalize_device(device),)
    for dev in devices:
        sup = _SUPERVISORS.get(dev)
        _SUPERVISORS[dev] = None
        if sup is not None:
            sup.stop()


def get_supervisor(device: str | None = None):
    """In-process supervisor for ``device``, or the only live one when omitted."""
    from hermes_cli.vllm_runtime.device import normalize_device

    if device is not None:
        return _SUPERVISORS.get(normalize_device(device))
    live = [sup for sup in _SUPERVISORS.values() if sup is not None]
    if len(live) == 1:
        return live[0]
    return None


def iter_live_supervisors():
    return [sup for sup in _SUPERVISORS.values() if sup is not None]


def start_in_flight(device: str | None = None) -> bool:
    """Serve is spawning or warming in this process. Status polls this.

    ``ensure_vllm_runtime`` holds that device's lock through ``wait_ready``
    (GET /v1/models). The supervisor is assigned before that wait so a
    concurrent status read does not see "no pid → stopped". Omit ``device``
    to ask whether either engine is in flight.
    """
    from hermes_cli.vllm_runtime.device import CPU, GPU, normalize_device

    if device is None:
        return any(start_in_flight(dev) for dev in (GPU, CPU))
    dev = normalize_device(device)
    if _SUPERVISORS.get(dev) is not None:
        return True
    return _START_LOCKS[dev].locked()


def _model_section(config: dict | None) -> dict:
    model = (config or {}).get("model")
    return model if isinstance(model, dict) else {}


def activate_vllm_provider(config: dict | None = None) -> str:
    """Point ``model.provider`` at managed (or already-remote) vLLM via the config API.

    Overwrites ``model.base_url`` only when the current URL is empty or loopback.
    A remote ``provider: vllm`` URL is left alone. Returns the URL that applies.
    """
    from cli import save_config_value
    from hermes_cli.config import load_config, save_config
    from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint
    from hermes_cli.vllm_runtime.supervisor import openai_base_url, vllm_settings

    cfg = config if config is not None else load_config()
    settings = vllm_settings(cfg)
    from hermes_cli.vllm_runtime.recommend import _DEFAULT_SERVED

    served = str(settings.get("served_model_name") or _DEFAULT_SERVED)
    from hermes_cli.local_engines import vllm_device_from_config
    from hermes_cli.vllm_runtime.device import managed_endpoint_name

    device = vllm_device_from_config(cfg)
    sup = get_supervisor(device)
    if sup is not None:
        managed = sup.base_url
    else:
        state = resolve_vllm_endpoint(cfg, wait_for_boot_s=0)
        managed = (state or {}).get("base_url") or openai_base_url(settings)
    current = str(_model_section(cfg).get("base_url") or "").strip()
    write_url = managed if _is_loopback_url(current) else current

    save_config_value("model.provider", "vllm")
    save_config_value("model.default", served)
    if _is_loopback_url(current):
        save_config_value("model.base_url", managed)

    live = load_config()
    providers = live.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    entry = dict(providers.get("vllm") or {}) if isinstance(providers.get("vllm"), dict) else {}
    entry.update({
        "name": managed_endpoint_name(device),
        "base_url": write_url.rstrip("/"),
        "model": served,
        "discover_models": True,
    })
    models = dict(entry.get("models") or {}) if isinstance(entry.get("models"), dict) else {}
    models.setdefault(served, {})
    entry["models"] = models
    providers["vllm"] = entry
    live["providers"] = providers
    with suppress(Exception):
        save_config(live, merge_existing=True)
    return write_url


def start_managed_vllm(config: dict | None = None, *, apply_recommend: bool = True):
    """One-click: recommend (optional) → isolated venv → supervised serve → activate.

    Writes config via the config API. The user does not pip, edit YAML, or run
    a third-party installer. Raises if the GPU cannot hold the 64k floor, or if
    another LLM is already occupying the GPU.
    """
    from cli import save_config_value
    from hermes_cli.config import load_config
    from hermes_cli.local_engines import ensure_managed_engine
    from hermes_cli.vllm_runtime.recommend import as_vllm_config, recommend_vllm

    if apply_recommend:
        rec = recommend_vllm()
        if not rec.feasible:
            raise RuntimeError(
                f"this GPU cannot run managed vLLM at the 64k tool-loop floor ({rec.reason})"
            )
        for key, value in as_vllm_config(rec).items():
            save_config_value(f"local_runtime.vllm.{key}", value)
    save_config_value("local_runtime.enabled", True)
    save_config_value("local_runtime.engine", "vllm")
    cfg = load_config() if config is None else config
    if apply_recommend:
        cfg = load_config()
    from hermes_cli.local_engines import stop_llama_engine
    from hermes_cli.vllm_runtime.occupancy import require_gpu_free

    stop_llama_engine()
    require_gpu_free()
    sup = ensure_managed_engine(cfg, force=True)
    if sup is None:
        from hermes_cli.vllm_runtime.venv import server_log_hint

        raise RuntimeError(f"managed vLLM did not start — see {server_log_hint()}")
    from hermes_cli.vllm_runtime.bench import verify_tool_calls

    verify_tool_calls(sup.base_url)
    activate_vllm_provider(load_config())
    return sup

