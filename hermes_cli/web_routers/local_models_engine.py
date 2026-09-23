"""Engine-aware helpers for the Local Models HTTP surface.

llama.cpp GGUF routes stay in ``local_models.py``. This sibling owns the
``local_runtime.engine`` contract: status extras, set-engine, start/stop
dispatch, llama-only guards, and vLLM install/use/recommend jobs.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from hermes_cli.local_engines import engine_from_config, vllm_device_from_config
from hermes_cli.vllm_runtime.device import (
    ALL_ENGINES, CPU, ENGINE_GPU, engine_to_device, is_vllm_engine,
)
from hermes_cli.vllm_runtime.supervisor import READY_TIMEOUT_S

_STORAGE_TTL_S = 30.0
_storage_cache: tuple[float, str, int] | None = None

_LLAMA_ONLY_DETAIL = (
    "This action is for llama.cpp GGUF models. "
    "Switch the local engine to llama.cpp first."
)
_ENGINE_NAMES = ALL_ENGINES
# POST /use, /quickstart (starting-server), /server start all wait on
# supervisor._wait_ready. Must be >= READY_TIMEOUT_S so the job does not
# fail while CUDA graphs are still capturing.
VLLM_START_TIMEOUT_S = READY_TIMEOUT_S
_LOG_PHASES = (
    ("Downloading", "Downloading model weights"),
    ("download", "Downloading model weights"),
    ("Loading weights", "Loading weights"),
    ("Loading safetensors", "Loading weights"),
    ("warmup", "Warming up GPU"),
    ("Warming up", "Warming up GPU"),
    ("Capturing CUDA graph", "Capturing CUDA graphs"),
    ("Application startup complete", "Server ready"),
    ("Uvicorn running", "Server ready"),
)


_CLIENT_HINTS = (
    "too big", "gated", "awq", "not downloaded", "gguf", "exl2",
    "missing file", "invalid id", "tool-loop floor",
    "sigkill", "oom", "out of memory",
)
_VLLM_RESTORE_KEYS = (
    "model", "served_model_name", "max_model_len",
    "gpu_memory_utilization", "quantization", "kv_cache_dtype",
    "tool_call_parser",
)
_ALREADY_STARTING = "vLLM is already starting"
_SWITCH_LOCK = threading.Lock()
_switch_in_flight = 0


def raise_engine_http(exc: BaseException) -> None:
    """Re-raise ``exc`` as FastAPI. Never map a client 4xx onto 502."""
    from hermes_cli.vllm_runtime.inventory import hf_http_status_and_detail, job_failure_detail
    from hermes_cli.vllm_runtime.occupancy import OccupyingLlmError
    from hermes_cli.vllm_runtime.supervisor import LEFTOVER_AWQ_MSG

    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, OccupyingLlmError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    mapped = hf_http_status_and_detail(exc)
    if mapped:
        raise HTTPException(status_code=mapped[0], detail=mapped[1]) from exc
    text = job_failure_detail(exc)
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    if isinstance(code, int) and 400 <= code < 500:
        raise HTTPException(status_code=400, detail=text) from exc
    if isinstance(exc, ValueError) or any(hint in text.lower() for hint in _CLIENT_HINTS):
        raise HTTPException(status_code=400, detail=text) from exc
    if LEFTOVER_AWQ_MSG.lower() in text.lower():
        raise HTTPException(status_code=400, detail=text) from exc
    raise HTTPException(status_code=502, detail=text) from exc


def configured_engine(config: dict | None = None) -> str:
    if config is None:
        from hermes_cli import config as config_mod

        config = config_mod.load_config()
    return engine_from_config(config)


def display_user_path(path: Path) -> str:
    """User-facing path. Profile home uses ``display_hermes_home()``; else ``~/…``."""
    from hermes_constants import display_hermes_home, get_default_hermes_root, get_hermes_home

    raw = Path(path).expanduser()
    try:
        resolved = raw.resolve(strict=False)
    except OSError:
        resolved = raw

    def _tilde(p: Path) -> str | None:
        try:
            return "~/" + p.relative_to(Path.home()).as_posix()
        except ValueError:
            return None

    roots: list[tuple[Path, str]] = [
        (get_hermes_home(), display_hermes_home()),
    ]
    default_root = get_default_hermes_root()
    roots.append((default_root, _tilde(default_root) or str(default_root)))
    for root, prefix in roots:
        try:
            rel = resolved.relative_to(Path(root).resolve(strict=False))
        except ValueError:
            continue
        if str(rel) == ".":
            return prefix
        return f"{prefix}/{rel.as_posix()}"
    return _tilde(resolved) or str(resolved)


def _dir_bytes_cached(path: Path) -> int:
    """Walk ``path`` at most once per TTL — statusbar polls hardware every 5s."""
    global _storage_cache
    key = str(path)
    now = time.monotonic()
    cached = _storage_cache
    if cached is not None and cached[1] == key and now - cached[0] < _STORAGE_TTL_S:
        return cached[2]
    total = 0
    if path.is_dir():
        for root, _dirs, files in os.walk(path, followlinks=False):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    continue
    _storage_cache = (now, key, total)
    return total


def _volume_bytes(path: Path) -> tuple[int, int]:
    """(free, total) on the volume that holds ``path``. Cheap statvfs."""
    probe = path if path.exists() else path.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return 0, 0
    return int(usage.free), int(usage.total)


def ctx_64k_feasible(total_bytes: int | None) -> bool | None:
    """Whether this card meets the official vLLM 64k catalog floor. None if unknown."""
    if not total_bytes:
        return None
    from hermes_cli.vllm_runtime.recommend import TIERS

    floor = min(t.min_vram_bytes for t in TIERS if t.feasible_at_64k)
    return int(total_bytes) >= floor


def cache_and_runtime_fields(engine: str, device: str | None = None) -> dict[str, Any]:
    """Download cache + runtime tree for the selected engine. One shape, both engines."""
    from hermes_cli.local_runtime import binaries, bootstrap
    from hermes_cli.vllm_runtime.inventory import hf_hub_dir
    from hermes_cli.vllm_runtime.venv import runtimes_root as vllm_runtimes_root

    if is_vllm_engine(engine):
        cache = hf_hub_dir()
        runtime = vllm_runtimes_root(device or engine_to_device(engine))
    else:
        cache = bootstrap.models_dir()
        runtime = binaries.runtimes_root()
    free, total = _volume_bytes(cache)
    return {
        "engine": engine,
        "models_dir": str(cache),
        "models_dir_display": display_user_path(cache),
        "models_storage_bytes": _dir_bytes_cached(cache),
        "disk_free_bytes": free,
        "disk_total_bytes": total,
        "runtime_dir": str(runtime),
        "runtime_dir_display": display_user_path(runtime),
    }


def refuse_llama_only() -> None:
    """400 + plain language when the active engine is vLLM — no GGUF I/O."""
    if is_vllm_engine(configured_engine()):
        raise HTTPException(status_code=400, detail=_LLAMA_ONLY_DETAIL)


def _vllm_last_error(device: str = "gpu") -> str | None:
    from hermes_cli.vllm_runtime.supervisor import read_last_error

    return read_last_error(device)


def classify_vllm_engine_state(
    *,
    ready: bool,
    starting: bool,
    last_error: str | None,
    installed: bool,
) -> str:
    """Ready only after GET /v1/models 200. Starting beats leftover last_error."""
    if ready:
        return "ready"
    if starting:
        return "starting"
    if last_error:
        return "error"
    if installed:
        return "stopped"
    return "not_installed"


def _vllm_start_job_running() -> bool:
    """Quickstart legs that wait on serve — Use/server-start are not jobs."""
    from hermes_cli.web_routers import local_models as lm

    start_kinds = frozenset({"quickstart"})
    start_phases = frozenset({"setting-default", "starting", "starting-server"})
    with lm._JOBS_LOCK:
        return any(
            j.get("status") == "running"
            and j.get("kind") in start_kinds
            and j.get("phase") in start_phases
            for j in lm._JOBS.values()
        )


@contextmanager
def mark_vllm_switch():
    """Use/switch is in flight — status is Starting; Turn on must not spawn."""
    global _switch_in_flight
    with _SWITCH_LOCK:
        _switch_in_flight += 1
    try:
        yield
    finally:
        with _SWITCH_LOCK:
            _switch_in_flight -= 1


def vllm_switch_in_flight() -> bool:
    with _SWITCH_LOCK:
        return _switch_in_flight > 0


def vllm_serve_starting(device: str | None = None) -> bool:
    """In-process start lock / supervisor, Use switch, or a start job.

    The job and Use-switch flags belong to the configured engine. A CPU
    start must not mark GPU as starting, or the GPU chip reads Stopped/Starting
    while the other engine is the one in flight.
    """
    from hermes_cli.vllm_runtime.bootstrap import start_in_flight
    from hermes_cli.vllm_runtime.device import normalize_device

    if device is None:
        return start_in_flight() or _vllm_start_job_running() or vllm_switch_in_flight()
    dev = normalize_device(device)
    if start_in_flight(dev):
        return True
    if dev != vllm_device_from_config(None):
        return False
    return _vllm_start_job_running() or vllm_switch_in_flight()


def refuse_duplicate_vllm_start() -> None:
    """Turn on must not start a second serve of this engine while Use / warmup is live."""
    if is_vllm_engine(configured_engine()) and vllm_serve_starting(vllm_device_from_config(None)):
        raise HTTPException(status_code=409, detail=_ALREADY_STARTING)


def vllm_engine_snapshot(
    config: dict | None = None, *, with_occupancy: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    """Shared starting/ready/stopped for status and hardware. No VRAM.

    ``device`` snapshots that vLLM engine even when the dropdown selects the
    other one. Readiness is GET /v1/models on that engine's state file.
    """
    from hermes_cli.vllm_runtime.bootstrap import get_supervisor
    from hermes_cli.vllm_runtime.device import normalize_device
    from hermes_cli.vllm_runtime.endpoint import _state_endpoint, resolve_vllm_endpoint
    from hermes_cli.vllm_runtime.inventory import running_served_model_name
    from hermes_cli.vllm_runtime.supervisor import (
        openai_base_url, probe_served_model_name, vllm_settings)
    from hermes_cli.vllm_runtime.venv import venv_ready

    cfg = config or {}
    explicit = device is not None
    device = normalize_device(device) if explicit else vllm_device_from_config(cfg)
    settings = vllm_settings(cfg, device=device)
    if explicit:
        running = _state_endpoint(device)
        served = None
        if running:
            served = probe_served_model_name(str(running.get("base_url") or "")) or None
    else:
        running = resolve_vllm_endpoint(cfg, wait_for_boot_s=0)
        served = running_served_model_name() or None
    ready = bool(served)
    occ = occupancy_payload() if with_occupancy else {
        "occupancy": [], "occupancy_message": None,
    }
    last_error = occ["occupancy_message"] or _vllm_last_error(device)
    installed = venv_ready(device)
    sup = get_supervisor(device)
    proc = getattr(sup, "proc", None) if sup is not None else None
    proc_live = proc is not None and getattr(proc, "poll", lambda: 0)() is None
    pid = (running or {}).get("pid")
    if pid is None and proc_live:
        pid = getattr(proc, "pid", None)
    starting = (not ready) and (
        running is not None or proc_live or vllm_serve_starting(device)
    )
    engine_state = classify_vllm_engine_state(
        ready=ready, starting=starting, last_error=last_error, installed=installed,
    )
    return {
        "engine_state": engine_state,
        "installed": installed,
        "last_error": last_error,
        "occupancy": occ,
        "pid": pid,
        "ready": ready,
        "running": running,
        "served": served,
        "settings": settings,
        "start_phase": None if ready else _vllm_log_phase(device),
        "server_base_url": (running or {}).get("base_url") or (
            openai_base_url(settings, device=device) if installed else None),
    }


def occupancy_payload() -> dict[str, Any]:
    from hermes_cli.vllm_runtime.occupancy import (
        discover_occupying_llms, occupancy_stop_message)

    hits = discover_occupying_llms()
    return {
        "occupancy": [
            {"kind": h.kind, "detail": h.detail, "port": h.port, "pid": h.pid}
            for h in hits
        ],
        "occupancy_message": occupancy_stop_message(hits),
    }


def _vllm_log_phase(device: str = "gpu") -> str | None:
    from hermes_cli.vllm_runtime.venv import install_log_read_paths, server_log_read_paths

    for path in (*server_log_read_paths(device), *install_log_read_paths(device)):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[-8000:]
        except OSError:
            continue
        for needle, label in reversed(_LOG_PHASES):
            if needle in text:
                return label
    return None


def vllm_status_fields(config: dict | None = None) -> dict[str, Any]:
    """Status for the selected vLLM engine — not llama tag / GGUF staging."""
    from hermes_cli.vllm_runtime.inventory import catalog_models
    from hermes_cli.vllm_runtime.venv import venv_dir, vllm_version_fields

    cfg = config or {}
    section = cfg.get("local_runtime") or {}
    device = vllm_device_from_config(cfg)
    engine = engine_from_config(cfg)
    snap = vllm_engine_snapshot(cfg)
    installed = snap["installed"]
    ready = snap["ready"]
    served = snap["served"]
    occ = snap["occupancy"]
    versions = vllm_version_fields(device=device)
    inventory = catalog_models(cfg)
    return {
        "engine": engine if is_vllm_engine(engine) else ENGINE_GPU,
        "vllm_device": device,
        "enabled": bool(section.get("enabled")),
        "venv_ready": installed,
        "runtime_installed": installed,
        "runtime_backend": engine if installed else None,
        "venv_path": str(venv_dir(device)) if installed else "",
        "engine_state": snap["engine_state"],
        "pid": snap["pid"],
        "server_running": ready,
        "server_base_url": snap["server_base_url"],
        "active_model_id": served,
        "served_model_name": served,
        "model": str(snap["settings"].get("model") or "") or None,
        "start_phase": snap["start_phase"],
        "last_error": snap["last_error"],
        **occ,
        "tag": versions.get("tag") or "",
        "configured_tag": versions.get("configured_tag") or "",
        "update_available": bool(versions.get("update_available")),
        # Isolated venv only — never PATH / Hermes' own version.
        "vllm_version": (str(versions.get("installed") or versions.get("tag") or "").strip() or None),
        "loaded_models": {served: "ready"} if ready else {},
        "loading": {},
        "placement": {},
        "models": [
            {"id": m["id"], "size_bytes": m.get("size_bytes") or 0,
             "size_label": m.get("size_label") or "—"}
            for m in inventory if m.get("cached") or m.get("active")
        ],
        **{k: v for k, v in cache_and_runtime_fields(
            engine if is_vllm_engine(engine) else ENGINE_GPU,
            device if is_vllm_engine(engine) else None,
        ).items()
           if k in ("models_dir", "models_dir_display", "runtime_dir", "runtime_dir_display")},
        "managed_engines": managed_vllm_engines(cfg),
    }


def managed_vllm_engines(config: dict | None = None) -> list[dict[str, Any]]:
    """GPU and CPU vLLM rows. A running engine is ready here even when the dropdown shows the other.

    ``vllm_version`` is that venv's installed package, not the sibling's.
    Setup may pass one pin to both, but an update upgrades only the selected
    device, so the two numbers are allowed to differ.
    """
    from hermes_cli.vllm_runtime.device import CPU, GPU, device_to_engine
    from hermes_cli.vllm_runtime.venv import installed_vllm_version

    cfg = config or {}
    rows: list[dict[str, Any]] = []
    for device in (GPU, CPU):
        snap = vllm_engine_snapshot(cfg, with_occupancy=False, device=device)
        rows.append({
            "device": device,
            "engine": device_to_engine(device),
            "engine_state": snap["engine_state"],
            "server_running": bool(snap["ready"]),
            "server_base_url": snap["server_base_url"],
            "served_model_name": snap["served"],
            "pid": snap["pid"],
            "runtime_installed": bool(snap["installed"]),
            "last_error": snap["last_error"],
            "start_phase": snap["start_phase"],
            "vllm_version": (installed_vllm_version(device) or "").strip() or None,
        })
    return rows


def recommend_payload() -> dict[str, Any]:
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.recommend import (
        as_vllm_config, recommend_vllm, recommend_vllm_cpu,
    )

    device = vllm_device_from_config(load_config())
    rec = recommend_vllm_cpu() if device == CPU else recommend_vllm()
    return {
        "feasible": rec.feasible,
        "reason": rec.reason,
        "tier": rec.tier.id if rec.tier is not None else None,
        "model": rec.model,
        "served_model_name": rec.served_model_name,
        "tool_call_parser": rec.tool_call_parser,
        "max_model_len": rec.max_model_len,
        "gpu_memory_utilization": rec.gpu_memory_utilization,
        "quantization": rec.quantization,
        "kv_cache_dtype": rec.kv_cache_dtype,
        "config": as_vllm_config(rec),
    }


def set_engine(name: str) -> dict[str, Any]:
    """Persist which Local Models page is configured. Does not stop a server.

    Pages are llama.cpp and vLLM. A legacy ``vllm-cpu`` write folds to the
    vLLM page with ``local_runtime.vllm.device: cpu``.
    """
    from cli import save_config_value
    from hermes_cli.vllm_runtime.device import CPU, ENGINE_CPU

    engine = str(name or "").strip().lower().replace("_", "-")
    if engine == ENGINE_CPU:
        from hermes_cli.vllm_runtime.settings import persist_selected

        save_config_value("local_runtime.engine", ENGINE_GPU)
        persist_selected(CPU)
        return {"ok": True, "engine": ENGINE_GPU, "vllm_device": CPU}
    if engine not in _ENGINE_NAMES:
        raise HTTPException(
            status_code=400,
            detail="engine must be 'llamacpp' or 'vllm'",
        )
    save_config_value("local_runtime.engine", engine)
    return {"ok": True, "engine": engine}


def set_vllm_device(device: str) -> dict[str, Any]:
    """Select which vLLM device chat follows. Does not stop either server."""
    from cli import save_config_value
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.device import CPU, ENGINE_CPU, GPU, normalize_device

    raw = str(device or "").strip().lower()
    if raw not in (GPU, CPU):
        raise HTTPException(status_code=400, detail="device must be 'gpu' or 'cpu'")
    chosen = normalize_device(raw)
    from hermes_cli.vllm_runtime.settings import persist_selected

    persist_selected(chosen)
    section = (load_config().get("local_runtime") or {})
    stored = str(section.get("engine") or "").strip().lower().replace("_", "-")
    if stored == ENGINE_CPU:
        save_config_value("local_runtime.engine", ENGINE_GPU)
    return {"ok": True, "engine": ENGINE_GPU, "vllm_device": chosen}


def stop_active_engine() -> None:
    from hermes_cli.local_engines import stop_configured_engine

    from hermes_cli.web_routers import local_models as lm

    stop_configured_engine(lm._load_config())
    lm._set_runtime_enabled(False)


def _vllm_overlay_from_settings(settings: dict) -> dict[str, Any]:
    return {key: settings.get(key) for key in _VLLM_RESTORE_KEYS}


def _persist_vllm_overlay(overlay: dict[str, Any], device: str | None = None) -> None:
    from hermes_cli.vllm_runtime.settings import persist_device_overlay

    persist_device_overlay(device or vllm_device_from_config(None), overlay)


def recover_vllm_after_failed_start(
    *,
    failed_id: str,
    previous: dict | None,
    was_running: bool,
) -> str | None:
    """Restore the previous serve, else the official recommend. Keep last_error.

    Returns the hid that started, or None when nothing could start. Never
    re-enters ``start_active_engine(recover=True)``.
    """
    from hermes_cli.local_engines import stop_vllm_device
    from hermes_cli.vllm_runtime.inventory import repo_is_cached
    from hermes_cli.vllm_runtime.supervisor import read_last_error, write_last_error

    failure = read_last_error()
    stop_vllm_device(vllm_device_from_config(None))
    prev_model = str((previous or {}).get("model") or "").strip()
    candidates: list[tuple[str, dict[str, Any]]] = []
    if (
        prev_model
        and prev_model != failed_id
        and was_running
        and repo_is_cached(prev_model)
    ):
        candidates.append((prev_model, _vllm_overlay_from_settings(previous or {})))
    hid, _notice, _rec, overlay = _official_setup()
    if hid and hid != failed_id and repo_is_cached(hid):
        if not any(item[0] == hid for item in candidates):
            candidates.append((hid, overlay))

    for cand_id, cand_overlay in candidates:
        _persist_vllm_overlay(cand_overlay)
        try:
            start_active_engine(recover=False)
            activate_vllm()
            if failure:
                write_last_error(failure)
            return cand_id
        except Exception:  # noqa: BLE001 — try the next rung
            stop_vllm_device(vllm_device_from_config(None))

    if prev_model and prev_model != failed_id:
        _persist_vllm_overlay(_vllm_overlay_from_settings(previous or {}))
    elif hid:
        _persist_vllm_overlay(overlay)
    if failure:
        write_last_error(failure)
    return None


def _start_configured_vllm(cfg: dict, settings: dict) -> None:
    from hermes_cli.local_engines import stop_llama_engine, stop_vllm_device
    from hermes_cli.vllm_runtime.occupancy import require_gpu_free
    from hermes_cli.vllm_runtime.supervisor import (
        MODEL_REMOVED_MSG, configured_cache_missing, configured_model_id,
        configured_unservable_reason, disable_auto_start, read_last_error,
        state_served_model_name, write_last_error)

    from hermes_cli.vllm_runtime.settings import ensure_device_model

    device = vllm_device_from_config(cfg)
    settings = ensure_device_model(cfg, device)
    from hermes_cli.config import load_config as _reload_after_pick

    cfg = _reload_after_pick()
    # GPU shares the card with llama.cpp. CPU vLLM does not stop either.
    if device != CPU:
        stop_llama_engine()
    blocked = configured_unservable_reason(settings)
    if not blocked:
        from hermes_cli.vllm_runtime.inventory import cached_model_config
        from hermes_cli.vllm_runtime.serve_compat import incompatible_with_device

        blocked = incompatible_with_device(
            configured_model_id(settings), device,
            config=cached_model_config(configured_model_id(settings)))
    if blocked:
        write_last_error(blocked, device)
        disable_auto_start()
        raise HTTPException(status_code=400, detail=blocked)
    if configured_cache_missing(settings):
        write_last_error(MODEL_REMOVED_MSG, device)
        disable_auto_start()
        raise RuntimeError(MODEL_REMOVED_MSG)
    wanted = str(settings.get("served_model_name") or "").strip()
    got = state_served_model_name(device)
    if wanted and got and wanted != got:
        stop_vllm_device(device)
    if device != CPU:
        require_gpu_free()
    from hermes_cli.vllm_runtime.bootstrap import ensure_vllm_runtime

    sup = ensure_vllm_runtime(cfg, force=True, timeout_s=VLLM_START_TIMEOUT_S)
    if sup is None:
        from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint

        running = resolve_vllm_endpoint(cfg, wait_for_boot_s=0)
        still = state_served_model_name(device)
        leftover = bool(wanted and still and wanted != still)
        if running is None or leftover:
            from hermes_cli.vllm_runtime.venv import server_log_hint

            disable_auto_start()
            raise RuntimeError(
                read_last_error(device)
                or f"managed vLLM did not start — see {server_log_hint(device)}")
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.bootstrap import activate_vllm_provider

    activate_vllm_provider(load_config())


def start_active_engine(*, recover: bool = True) -> None:
    """Start the configured engine. Occupancy is not swallowed.

    GPU vLLM and llama.cpp stop each other. CPU vLLM is left running.
    A failed vLLM start restores the previous serve when the caller asked, else
    the official recommended row — never a crash-loop on the dead id.
    """
    from hermes_cli.local_engines import stop_vllm_device
    from hermes_cli.vllm_runtime.occupancy import OccupyingLlmError
    from hermes_cli.vllm_runtime.supervisor import configured_model_id, vllm_settings

    from hermes_cli.web_routers import local_models as lm

    cfg = lm._set_runtime_enabled(True)
    if is_vllm_engine(configured_engine(cfg)):
        settings = vllm_settings(cfg, device=vllm_device_from_config(cfg))
        failed_id = configured_model_id(settings)
        try:
            _start_configured_vllm(cfg, settings)
        except OccupyingLlmError:
            raise
        except Exception:
            if recover:
                recover_vllm_after_failed_start(
                    failed_id=failed_id, previous=None, was_running=False)
            raise
        return
    stop_vllm_device("gpu")
    lm._start_local_server(cfg, lm._SERVER_START_FAILED)


def _official_setup(rec=None):
    """VRAM-fit public catalog row. Never leftover ``local_runtime.vllm.model``.

    CPU engine plans ``recommend_vllm_cpu`` — not the GPU AWQ catalog.
    """
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.recommend import overlay_for_setup, resolve_public_setup

    device = vllm_device_from_config(load_config())
    hid, notice, picked = resolve_public_setup(rec, device=device)
    return hid, notice, picked, overlay_for_setup(hid, picked)


def vllm_quickstart_plan(model_id: str | None = None) -> dict[str, Any]:
    """Preflight for Set up for me: official ``recommend_vllm()`` id only.

    ``model_id`` and leftover ``local_runtime.vllm.model`` (gated Gemma, a
    search hit, a 401 hollow) are ignored — one-click is the VRAM-fit
    catalog row, never resume-the-last-browse.
    """
    from hermes_cli.vllm_runtime.venv import venv_ready

    _ = model_id  # ignored — leftover Gemma / search hits must not win
    hid, notice, rec, overlay = _official_setup()
    from hermes_cli.config import load_config

    device = vllm_device_from_config(load_config())
    if device == CPU:
        pass
    elif rec.feasible and hid:
        pass
    elif rec.reason == "no_nvidia" and hid:
        # No probe: ship the 16 GB balanced id, not a 14B or stale Hermes-8B.
        pass
    else:
        raise HTTPException(
            status_code=409,
            detail=(
                "this GPU cannot run managed vLLM at the 64k tool-loop floor — "
                "open Local Models to pick a smaller build"
            ),
        )
    return {
        "model": hid,
        "display_name": hid,
        "needs_runtime": not (venv_ready("gpu") and venv_ready("cpu")),
        "needs_download": not repo_is_cached(hid),
        "apply_recommend": True,
        "notice": notice,
    }


def run_vllm_quickstart(job: dict, plan: dict) -> None:
    """Same four legs as llama quickstart: venv → HF weights → serve → default.

    Restore recommended setup must not start Qwen beside leftover Nemotron:
    occupancy (foreign), stop managed leftover, official overlay (clean argv),
    download, then start and wait ``/v1/models``. Kick is off until start.
    """
    from hermes_cli.config import load_config
    from hermes_cli.local_engines import stop_vllm_device
    from hermes_cli.vllm_runtime.inventory import ensure_hf_weights
    from hermes_cli.vllm_runtime.occupancy import require_gpu_free
    from hermes_cli.vllm_runtime.settings import persist_device_overlay
    from hermes_cli.vllm_runtime.supervisor import (
        clear_last_error, disable_auto_start, read_last_error, vllm_settings)
    from hermes_cli.vllm_runtime.venv import ensure_both_vllm_venvs

    from hermes_cli.web_routers import local_models as lm

    # Re-resolve at run time — leftover config / a stale plan.model must not win.
    hid, notice, _rec, overlay = _official_setup()
    if notice:
        job["detail"] = notice
    cfg = load_config()
    device = vllm_device_from_config(cfg)
    # Foreign Ollama/etc. first so we do not kill our leftover then fail.
    # CPU vLLM does not take the GPU — skip occupancy.
    if device != CPU:
        require_gpu_free()
    # Stop this device's leftover BEFORE overlay/enable. Writing Qwen while
    # Nemotron is up lets desktop boot + kick spawn a second serve (SIGKILL).
    # The sibling vLLM server is not that leftover.
    stop_vllm_device(device)
    disable_auto_start()
    clear_last_error(device)
    persist_device_overlay(device, overlay)
    settings = vllm_settings(load_config(), device=device)
    installing = "Installing vLLM on the CPU" if device == CPU else "Installing vLLM on the GPU"
    lm._step(job, "installing-runtime", installing)
    _raise_if_both_venvs_failed(
        ensure_both_vllm_venvs(str(settings.get("python") or "")))
    if not repo_is_cached(hid):
        job["done_bytes"] = 0
        lm._step(job, "downloading", f"Downloading {hid}")
        ensure_hf_weights(hid, job)
    lm._step(job, "starting-server", "Starting vLLM")
    try:
        start_active_engine()
    except Exception:
        err = read_last_error()
        if err:
            raise RuntimeError(err) from None
        raise
    lm._step(job, "setting-default", "Making it your default")
    activate_vllm()
    done = notice or f"{hid} is ready — new chats use it"
    lm._finish(job, done)


def _raise_if_both_venvs_failed(results: dict) -> dict:
    """Setup installs both engines. Fail only when neither venv landed."""
    ok = {k: v for k, v in results.items() if not isinstance(v, Exception)}
    if ok:
        return results
    errors = "; ".join(f"{k}: {v}" for k, v in results.items())
    raise RuntimeError(f"vLLM GPU and CPU installs failed ({errors})")


def apply_recommend_and_install(job: dict | None = None) -> None:
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.inventory import ensure_hf_weights
    from hermes_cli.vllm_runtime.supervisor import vllm_settings
    from hermes_cli.vllm_runtime.venv import ensure_both_vllm_venvs

    hid, notice, _rec, overlay = _official_setup()
    from hermes_cli.vllm_runtime.settings import persist_device_overlay

    device = vllm_device_from_config(load_config())
    persist_device_overlay(device, overlay)
    settings = vllm_settings(load_config(), device=device)
    _raise_if_both_venvs_failed(
        ensure_both_vllm_venvs(str(settings.get("python") or "")))
    if hid:
        if job is not None:
            job["phase"] = "downloading"
            job["detail"] = notice or f"Downloading {hid}"
        ensure_hf_weights(hid, job)


def use_cached_vllm(hf_id: str) -> dict[str, Any]:
    """Switch to an already-cached HF id and start/reload. Never downloads."""
    hid = (hf_id or "").strip()
    if not hid or "/" not in hid:
        raise HTTPException(status_code=400, detail="model must be an org/name Hugging Face id")
    from hermes_cli.config import load_config as _load_for_device
    from hermes_cli.vllm_runtime.inventory import (
        cached_model_config, gated_repo_reason,
    )
    from hermes_cli.vllm_runtime.serve_compat import incompatible_with_device
    from hermes_cli.vllm_runtime.supervisor import disable_auto_start, write_last_error

    device = vllm_device_from_config(_load_for_device())
    blocked = incompatible_with_device(
        hid, device, config=cached_model_config(hid),
    ) or gated_repo_reason(hid)
    if blocked:
        write_last_error(blocked, device)
        disable_auto_start()
        raise HTTPException(status_code=400, detail=blocked)
    from hermes_cli.vllm_runtime.inventory import (
        TOO_BIG_USE_MSG, cached_repo_fit, cpu_fit_ram_bytes,
    )
    from hermes_cli.vllm_runtime.recommend import recommend_vllm
    rec = recommend_vllm()
    if device == CPU:
        from hermes_cli.local_runtime import hardware as hw

        _total, _used, ram_avail = hw._ram_stats()
        tags = cached_repo_fit(
            hid, total_vram=0, total_ram=cpu_fit_ram_bytes(_total, ram_avail),
            device=CPU)
    else:
        tags = cached_repo_fit(hid, total_vram=rec.probe.total_bytes or 0)
    if tags.get("fit") == "too-big":
        detail = str(tags.get("fit_detail") or TOO_BIG_USE_MSG)
        write_last_error(detail, device)
        disable_auto_start()
        raise HTTPException(status_code=400, detail=detail)
    if not repo_is_cached(hid):
        raise HTTPException(
            status_code=409,
            detail=f"{hid} is not downloaded — Download it first",
        )
    from hermes_cli.config import load_config
    from hermes_cli.local_engines import stop_vllm_device
    from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint
    from hermes_cli.vllm_runtime.supervisor import vllm_settings

    with mark_vllm_switch():
        previous = dict(vllm_settings(load_config()))
        current = str(previous.get("model") or "").strip()
        running = resolve_vllm_endpoint(wait_for_boot_s=0) is not None
        set_vllm_model(hid)
        if running and current != hid:
            # New weights need a new serve on THIS device. The sibling stays up.
            stop_vllm_device(device)
        try:
            start_active_engine(recover=False)
            result = activate_vllm()
        except Exception:
            if device != CPU:
                recover_vllm_after_failed_start(
                    failed_id=hid, previous=previous, was_running=running)
            raise
        result["model"] = hid
        result["needs_download"] = False
        result["already_downloaded"] = True
        return result


def activate_vllm() -> dict[str, Any]:
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.bench import verify_tool_calls
    from hermes_cli.vllm_runtime.bootstrap import activate_vllm_provider
    from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint

    url = activate_vllm_provider(load_config())
    state = resolve_vllm_endpoint(wait_for_boot_s=0)
    if state:
        verify_tool_calls(state["base_url"])
    return {"ok": True, "base_url": url}


def vllm_models_payload(config: dict | None = None) -> dict[str, Any]:
    from hermes_cli.vllm_runtime.inventory import catalog_models

    return {"models": catalog_models(config, with_hf_meta=True)}


def set_vllm_model(hf_id: str) -> dict[str, Any]:
    from hermes_cli.vllm_runtime.inventory import apply_vllm_model

    try:
        return apply_vllm_model(hf_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def repo_is_cached(hf_id: str) -> bool:
    from hermes_cli.vllm_runtime.inventory import repo_is_cached as _cached

    return _cached(hf_id)


def setup_download_model(requested: str | None) -> str:
    """Keep an explicit Download id. Empty body is failsafe → official.

    Leftover ``local_runtime.vllm.model`` must not win on Set up for me /
    install. A user clicking Download on a search hit keeps that id.
    """
    hid = (requested or "").strip()
    if hid:
        return hid
    chosen, _notice, _rec, _overlay = _official_setup()
    return chosen


def download_vllm_weights(hf_id: str, job: dict | None = None) -> dict[str, Any]:
    from hermes_cli.vllm_runtime.inventory import ensure_hf_weights

    hid = setup_download_model(hf_id)
    try:
        return ensure_hf_weights(hid, job)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _write_official_vllm_model() -> str:
    """Persist the VRAM-fit catalog row (or the shipped 16 GB id). Never a search hit."""
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.settings import persist_device_overlay

    hid, _notice, _rec, overlay = _official_setup()
    persist_device_overlay(vllm_device_from_config(load_config()), overlay)
    return hid


def delete_vllm_model(hf_id: str) -> dict[str, Any]:
    """Remove HF cache for ``hf_id``. Never downloads, recommends, or starts serve.

    An empty library resets ``local_runtime.vllm.model`` to the official
    recommend — leftover gated search hits (Gemma 401) must not survive a
    clean. An uncached configured leftover deletes as 200, not 404.
    """
    from cli import save_config_value
    from hermes_cli.config import load_config
    from hermes_cli.local_engines import stop_vllm_device
    from hermes_cli.vllm_runtime.inventory import cached_repo_ids, delete_cached_repo
    from hermes_cli.vllm_runtime.supervisor import (
        MODEL_REMOVED_MSG, clear_last_error, configured_model_id, vllm_settings,
        write_last_error)

    hid = (hf_id or "").strip()
    cfg = load_config()
    device = vllm_device_from_config(cfg)
    settings = vllm_settings(cfg, device=device)
    was_configured = configured_model_id(settings) == hid
    try:
        delete_cached_repo(hid)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError:
        if not was_configured:
            raise HTTPException(
                status_code=404, detail=f"{hid} is not in the local Hugging Face cache")
    if was_configured or not cached_repo_ids():
        stop_vllm_device(vllm_device_from_config(load_config()))
    if not cached_repo_ids():
        # Last hub dir (or leftover config with no dir): first-time setup,
        # official default, not Gemma-from-search.
        _write_official_vllm_model()
        save_config_value("local_runtime.enabled", False)
        clear_last_error()
    elif was_configured:
        from hermes_cli.vllm_runtime.settings import persist_device_overlay

        persist_device_overlay(device, {"model": "", "served_model_name": ""})
        shared = (cfg.get("local_runtime") or {}).get("vllm")
        shared = shared if isinstance(shared, dict) else {}
        if str(shared.get("model") or "").strip() == hid:
            save_config_value("local_runtime.vllm.model", "")
            save_config_value("local_runtime.vllm.served_model_name", "")
        write_last_error(MODEL_REMOVED_MSG)
    return {"ok": True}


def search_vllm_models(q: str, limit: int = 20) -> dict[str, Any]:
    from hermes_cli.vllm_runtime.inventory import search_hf_models

    try:
        return {"hits": search_hf_models(q, limit)}
    except Exception as exc:  # noqa: BLE001
        raise_engine_http(exc)


def check_vllm_update() -> dict[str, Any]:
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.venv import vllm_version_fields

    return vllm_version_fields(check=True, device=vllm_device_from_config(load_config()))


def apply_vllm_update() -> None:
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.supervisor import vllm_settings
    from hermes_cli.vllm_runtime.venv import (
        ensure_vllm_venv,
        install_log_path,
        installed_vllm_version,
        latest_vllm_pypi_version,
        write_version_check,
    )

    cfg = load_config()
    settings = vllm_settings(cfg)
    device = vllm_device_from_config(cfg)
    latest = latest_vllm_pypi_version()
    ensure_vllm_venv(
        str(settings.get("python") or ""),
        upgrade=True, version=latest or None, device=device)
    installed = installed_vllm_version(device)
    if not installed:
        raise RuntimeError(
            f"vLLM update finished but no version is installed. See {install_log_path(device)}")
    remembered = write_version_check(installed, latest or installed, device)
    if latest and remembered.get("update_available"):
        raise RuntimeError(
            f"vLLM is still {installed} after update (PyPI has {latest}). "
            f"See {install_log_path(device)}"
        )
