"""Engine-aware helpers for the Local Models HTTP surface.

llama.cpp GGUF routes stay in ``local_models.py``. This sibling owns the
``local_runtime.engine`` contract: status extras, set-engine, start/stop
dispatch, llama-only guards, and vLLM install/use/recommend jobs.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from hermes_cli.local_engines import engine_from_config
from hermes_cli.vllm_runtime.supervisor import READY_TIMEOUT_S

_STORAGE_TTL_S = 30.0
_storage_cache: tuple[float, str, int] | None = None

_LLAMA_ONLY_DETAIL = (
    "This action is for llama.cpp GGUF models. "
    "Switch the local engine to llama.cpp first."
)
_ENGINE_NAMES = frozenset({"llamacpp", "vllm"})
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


def cache_and_runtime_fields(engine: str) -> dict[str, Any]:
    """Download cache + runtime tree for the selected engine. One shape, both engines."""
    from hermes_cli.local_runtime import binaries, bootstrap
    from hermes_cli.vllm_runtime.inventory import hf_hub_dir
    from hermes_cli.vllm_runtime.venv import runtimes_root as vllm_runtimes_root

    if engine == "vllm":
        cache = hf_hub_dir()
        runtime = vllm_runtimes_root()
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
    if configured_engine() == "vllm":
        raise HTTPException(status_code=400, detail=_LLAMA_ONLY_DETAIL)


def _vllm_last_error() -> str | None:
    from hermes_cli.vllm_runtime.supervisor import read_last_error

    return read_last_error()


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


def _vllm_log_phase() -> str | None:
    from hermes_cli.vllm_runtime.venv import install_log_path, runtimes_root

    for path in (runtimes_root() / "vllm-server.log", install_log_path()):
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
    from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint
    from hermes_cli.vllm_runtime.inventory import catalog_models, running_served_model_name
    from hermes_cli.vllm_runtime.supervisor import openai_base_url, vllm_settings
    from hermes_cli.vllm_runtime.venv import venv_dir, venv_ready, vllm_version_fields

    cfg = config or {}
    section = cfg.get("local_runtime") or {}
    settings = vllm_settings(cfg)
    running = resolve_vllm_endpoint(wait_for_boot_s=0)
    # Live serve only after GET /v1/models 200 — spawn-time state is not ready.
    served = running_served_model_name() or None
    ready = bool(served)
    configured = str(settings.get("model") or "") or None
    occ = occupancy_payload()
    versions = vllm_version_fields()
    inventory = catalog_models(cfg)
    return {
        "engine": "vllm",
        "enabled": bool(section.get("enabled")),
        "venv_ready": venv_ready(),
        "runtime_installed": venv_ready(),
        "runtime_backend": "vllm" if venv_ready() else None,
        "venv_path": str(venv_dir()) if venv_ready() else "",
        "server_running": ready,
        "server_base_url": (running or {}).get("base_url") or (
            openai_base_url(settings) if venv_ready() else None),
        "active_model_id": served,
        "served_model_name": served,
        "model": configured,
        "start_phase": None if ready else _vllm_log_phase(),
        "last_error": occ["occupancy_message"] or _vllm_last_error(),
        **occ,
        "tag": versions.get("tag") or "",
        "configured_tag": versions.get("configured_tag") or "",
        "update_available": bool(versions.get("update_available")),
        "loaded_models": {served: "ready"} if ready else {},
        "loading": {},
        "placement": {},
        "models": [
            {"id": m["id"], "size_bytes": m.get("size_bytes") or 0,
             "size_label": m.get("size_label") or "—"}
            for m in inventory if m.get("cached") or m.get("active")
        ],
        **{k: v for k, v in cache_and_runtime_fields("vllm").items()
           if k in ("models_dir", "models_dir_display", "runtime_dir", "runtime_dir_display")},
    }


def recommend_payload() -> dict[str, Any]:
    from hermes_cli.vllm_runtime.recommend import as_vllm_config, recommend_vllm

    rec = recommend_vllm()
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
    """Persist ``local_runtime.engine`` only — a view of which pane is configured.

    Does not stop a running supervisor. Starting the newly selected engine is
    what stops the other (one GPU, one resident weights file).
    """
    from cli import save_config_value

    engine = str(name or "").strip().lower()
    if engine not in _ENGINE_NAMES:
        raise HTTPException(status_code=400, detail="engine must be 'llamacpp' or 'vllm'")
    save_config_value("local_runtime.engine", engine)
    return {"ok": True, "engine": engine}


def stop_active_engine() -> None:
    from hermes_cli.local_engines import stop_configured_engine

    from hermes_cli.web_routers import local_models as lm

    stop_configured_engine(lm._load_config())
    lm._set_runtime_enabled(False)


def _vllm_overlay_from_settings(settings: dict) -> dict[str, Any]:
    return {key: settings.get(key) for key in _VLLM_RESTORE_KEYS}


def _persist_vllm_overlay(overlay: dict[str, Any]) -> None:
    from cli import save_config_value

    for key, value in overlay.items():
        save_config_value(f"local_runtime.vllm.{key}", value)


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
    from hermes_cli.local_engines import stop_vllm_engine
    from hermes_cli.vllm_runtime.inventory import repo_is_cached
    from hermes_cli.vllm_runtime.supervisor import read_last_error, write_last_error

    failure = read_last_error()
    stop_vllm_engine()
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
            stop_vllm_engine()

    if prev_model and prev_model != failed_id:
        _persist_vllm_overlay(_vllm_overlay_from_settings(previous or {}))
    elif hid:
        _persist_vllm_overlay(overlay)
    if failure:
        write_last_error(failure)
    return None


def _start_configured_vllm(cfg: dict, settings: dict) -> None:
    from hermes_cli.local_engines import stop_llama_engine, stop_vllm_engine
    from hermes_cli.vllm_runtime.occupancy import require_gpu_free
    from hermes_cli.vllm_runtime.supervisor import (
        MODEL_REMOVED_MSG, configured_cache_missing,
        configured_unservable_reason, disable_auto_start, read_last_error,
        state_served_model_name, write_last_error)

    stop_llama_engine()
    blocked = configured_unservable_reason(settings)
    if blocked:
        write_last_error(blocked)
        disable_auto_start()
        raise HTTPException(status_code=400, detail=blocked)
    if configured_cache_missing(settings):
        write_last_error(MODEL_REMOVED_MSG)
        disable_auto_start()
        raise RuntimeError(MODEL_REMOVED_MSG)
    wanted = str(settings.get("served_model_name") or "").strip()
    got = state_served_model_name()
    if wanted and got and wanted != got:
        stop_vllm_engine()
    require_gpu_free()
    from hermes_cli.vllm_runtime.bootstrap import ensure_vllm_runtime

    sup = ensure_vllm_runtime(cfg, force=True, timeout_s=VLLM_START_TIMEOUT_S)
    if sup is None:
        from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint

        running = resolve_vllm_endpoint(wait_for_boot_s=0)
        still = state_served_model_name()
        leftover = bool(wanted and still and wanted != still)
        if running is None or leftover:
            disable_auto_start()
            raise RuntimeError(
                read_last_error()
                or "managed vLLM did not start — see runtimes/vllm/vllm-server.log")
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.bootstrap import activate_vllm_provider

    activate_vllm_provider(load_config())


def start_active_engine(*, recover: bool = True) -> None:
    """Start the configured engine; stop the other first. Occupancy is not swallowed.

    A failed vLLM start restores the previous serve when the caller asked, else
    the official recommended row — never a crash-loop on the dead id.
    """
    from hermes_cli.local_engines import stop_vllm_engine
    from hermes_cli.vllm_runtime.occupancy import OccupyingLlmError
    from hermes_cli.vllm_runtime.supervisor import configured_model_id, vllm_settings

    from hermes_cli.web_routers import local_models as lm

    cfg = lm._set_runtime_enabled(True)
    if configured_engine(cfg) == "vllm":
        settings = vllm_settings(cfg)
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
    stop_vllm_engine()
    lm._start_local_server(cfg, lm._SERVER_START_FAILED)


def _official_setup(rec=None):
    """VRAM-fit public catalog row. Never leftover ``local_runtime.vllm.model``."""
    from hermes_cli.vllm_runtime.recommend import overlay_for_setup, resolve_public_setup

    hid, notice, picked = resolve_public_setup(rec)
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
    if rec.feasible and hid:
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
        "display_name": overlay["served_model_name"] or hid.rsplit("/", 1)[-1],
        "needs_runtime": not venv_ready(),
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
    from cli import save_config_value
    from hermes_cli.config import load_config
    from hermes_cli.local_engines import stop_vllm_engine
    from hermes_cli.vllm_runtime.inventory import ensure_hf_weights
    from hermes_cli.vllm_runtime.occupancy import require_gpu_free
    from hermes_cli.vllm_runtime.supervisor import (
        clear_last_error, disable_auto_start, read_last_error, vllm_settings)
    from hermes_cli.vllm_runtime.venv import ensure_vllm_venv

    from hermes_cli.web_routers import local_models as lm

    # Re-resolve at run time — leftover config / a stale plan.model must not win.
    hid, notice, _rec, overlay = _official_setup()
    if notice:
        job["detail"] = notice
    # Foreign Ollama/etc. first so we do not kill our leftover then fail.
    require_gpu_free()
    # Stop leftover BEFORE overlay/enable. Writing Qwen while Nemotron is up
    # lets desktop boot + kick spawn a second serve (SIGKILL / rc=-9).
    stop_vllm_engine()
    disable_auto_start()
    clear_last_error()
    for key, value in overlay.items():
        save_config_value(f"local_runtime.vllm.{key}", value)
    settings = vllm_settings(load_config())
    lm._step(job, "installing-runtime", "Installing vLLM")
    ensure_vllm_venv(str(settings.get("python") or ""))
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
    done = notice or f"{overlay['served_model_name'] or hid.rsplit('/', 1)[-1]} is ready — new chats use it"
    lm._finish(job, done)


def apply_recommend_and_install(job: dict | None = None) -> None:
    from cli import save_config_value
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.inventory import ensure_hf_weights
    from hermes_cli.vllm_runtime.supervisor import vllm_settings
    from hermes_cli.vllm_runtime.venv import ensure_vllm_venv

    hid, notice, _rec, overlay = _official_setup()
    for key, value in overlay.items():
        save_config_value(f"local_runtime.vllm.{key}", value)
    settings = vllm_settings(load_config())
    ensure_vllm_venv(str(settings.get("python") or ""))
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
    from hermes_cli.vllm_runtime.inventory import gated_repo_reason, unservable_reason
    from hermes_cli.vllm_runtime.supervisor import disable_auto_start, write_last_error

    blocked = unservable_reason(hid) or gated_repo_reason(hid)
    if blocked:
        write_last_error(blocked)
        disable_auto_start()
        raise HTTPException(status_code=400, detail=blocked)
    from hermes_cli.vllm_runtime.inventory import TOO_BIG_USE_MSG, cached_repo_fit
    from hermes_cli.vllm_runtime.recommend import recommend_vllm

    rec = recommend_vllm()
    tags = cached_repo_fit(hid, total_vram=rec.probe.total_bytes or 0)
    if tags.get("fit") == "too-big":
        detail = str(tags.get("fit_detail") or TOO_BIG_USE_MSG)
        write_last_error(detail)
        disable_auto_start()
        raise HTTPException(status_code=400, detail=detail)
    if not repo_is_cached(hid):
        raise HTTPException(
            status_code=409,
            detail=f"{hid} is not downloaded — Download it first",
        )
    from hermes_cli.config import load_config
    from hermes_cli.local_engines import stop_vllm_engine
    from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint
    from hermes_cli.vllm_runtime.supervisor import vllm_settings

    previous = dict(vllm_settings(load_config()))
    current = str(previous.get("model") or "").strip()
    running = resolve_vllm_endpoint(wait_for_boot_s=0) is not None
    set_vllm_model(hid)
    if running and current != hid:
        # New weights need a new serve. Same-id reuse leaves a healthy process up.
        stop_vllm_engine()
    try:
        start_active_engine(recover=False)
        result = activate_vllm()
    except Exception:
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
    from cli import save_config_value

    hid, _notice, _rec, overlay = _official_setup()
    for key, value in overlay.items():
        save_config_value(f"local_runtime.vllm.{key}", value)
    return hid


def delete_vllm_model(hf_id: str) -> dict[str, Any]:
    """Remove HF cache for ``hf_id``. Never downloads, recommends, or starts serve.

    An empty library resets ``local_runtime.vllm.model`` to the official
    recommend — leftover gated search hits (Gemma 401) must not survive a
    clean. An uncached configured leftover deletes as 200, not 404.
    """
    from cli import save_config_value
    from hermes_cli.config import load_config
    from hermes_cli.local_engines import stop_vllm_engine
    from hermes_cli.vllm_runtime.inventory import cached_repo_ids, delete_cached_repo
    from hermes_cli.vllm_runtime.supervisor import (
        MODEL_REMOVED_MSG, clear_last_error, configured_model_id, vllm_settings,
        write_last_error)

    hid = (hf_id or "").strip()
    settings = vllm_settings(load_config())
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
        stop_vllm_engine()
    if not cached_repo_ids():
        # Last hub dir (or leftover config with no dir): first-time setup,
        # official default, not Gemma-from-search.
        _write_official_vllm_model()
        save_config_value("local_runtime.enabled", False)
        clear_last_error()
    elif was_configured:
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
    from hermes_cli.vllm_runtime.venv import vllm_version_fields

    return vllm_version_fields(check=True)


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

    settings = vllm_settings(load_config())
    latest = latest_vllm_pypi_version()
    ensure_vllm_venv(str(settings.get("python") or ""), upgrade=True, version=latest or None)
    installed = installed_vllm_version()
    if not installed:
        raise RuntimeError(f"vLLM update finished but no version is installed. See {install_log_path()}")
    remembered = write_version_check(installed, latest or installed)
    if latest and remembered.get("update_available"):
        raise RuntimeError(
            f"vLLM is still {installed} after update (PyPI has {latest}). "
            f"See {install_log_path()}"
        )
