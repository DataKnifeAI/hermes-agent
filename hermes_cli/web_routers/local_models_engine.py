"""Engine-aware helpers for the Local Models HTTP surface.

llama.cpp GGUF routes stay in ``local_models.py``. This sibling owns the
``local_runtime.engine`` contract: status extras, set-engine, start/stop
dispatch, llama-only guards, and vLLM install/use/recommend jobs.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from hermes_cli.local_engines import engine_from_config

_LLAMA_ONLY_DETAIL = (
    "This action is for llama.cpp GGUF models. "
    "Switch the local engine to llama.cpp first."
)
_ENGINE_NAMES = frozenset({"llamacpp", "vllm"})
_LOG_PHASES = (
    ("Downloading", "Downloading model weights"),
    ("download", "Downloading model weights"),
    ("Loading weights", "Loading weights"),
    ("Loading safetensors", "Loading weights"),
    ("Capturing CUDA graph", "Capturing CUDA graphs"),
    ("Application startup complete", "Server ready"),
    ("Uvicorn running", "Server ready"),
)


def configured_engine(config: dict | None = None) -> str:
    if config is None:
        from hermes_cli import config as config_mod

        config = config_mod.load_config()
    return engine_from_config(config)


def refuse_llama_only() -> None:
    """400 + plain language when the active engine is vLLM — no GGUF I/O."""
    if configured_engine() == "vllm":
        raise HTTPException(status_code=400, detail=_LLAMA_ONLY_DETAIL)


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
    from hermes_cli.vllm_runtime.inventory import catalog_models
    from hermes_cli.vllm_runtime.supervisor import openai_base_url, vllm_settings
    from hermes_cli.vllm_runtime.venv import venv_dir, venv_ready, vllm_version_fields

    cfg = config or {}
    section = cfg.get("local_runtime") or {}
    settings = vllm_settings(cfg)
    running = resolve_vllm_endpoint(wait_for_boot_s=0)
    served = str(settings.get("served_model_name") or "") or None
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
        "server_running": running is not None,
        "server_base_url": (running or {}).get("base_url") or (
            openai_base_url(settings) if venv_ready() else None),
        "active_model_id": served or configured,
        "served_model_name": served,
        "model": configured,
        "start_phase": None if running else _vllm_log_phase(),
        "last_error": occ["occupancy_message"],
        **occ,
        "tag": versions.get("tag") or "",
        "configured_tag": versions.get("configured_tag") or "",
        "update_available": bool(versions.get("update_available")),
        "loaded_models": {served: "ready"} if running and served else {},
        "loading": {},
        "placement": {},
        "models": [
            {"id": m["id"], "size_bytes": m.get("size_bytes") or 0,
             "size_label": m.get("size_label") or "—"}
            for m in inventory if m.get("cached") or m.get("active")
        ],
        "models_dir": "",
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


def start_active_engine() -> None:
    """Start the configured engine; stop the other first. Occupancy is not swallowed."""
    from hermes_cli.local_engines import stop_llama_engine, stop_vllm_engine
    from hermes_cli.vllm_runtime.occupancy import OccupyingLlmError, require_gpu_free

    from hermes_cli.web_routers import local_models as lm

    cfg = lm._set_runtime_enabled(True)
    if configured_engine(cfg) == "vllm":
        stop_llama_engine()
        try:
            require_gpu_free()
        except OccupyingLlmError:
            raise
        from hermes_cli.vllm_runtime.bootstrap import ensure_vllm_runtime

        sup = ensure_vllm_runtime(cfg, force=True)
        if sup is None:
            raise RuntimeError(
                "managed vLLM did not start — see runtimes/vllm/vllm-server.log")
        from hermes_cli.config import load_config
        from hermes_cli.vllm_runtime.bootstrap import activate_vllm_provider

        activate_vllm_provider(load_config())
        return
    stop_vllm_engine()
    lm._start_local_server(cfg, lm._SERVER_START_FAILED)


def apply_recommend_and_install(job: dict | None = None) -> None:
    from cli import save_config_value
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.inventory import ensure_hf_weights
    from hermes_cli.vllm_runtime.recommend import as_vllm_config, recommend_vllm
    from hermes_cli.vllm_runtime.supervisor import vllm_settings
    from hermes_cli.vllm_runtime.venv import ensure_vllm_venv

    rec = recommend_vllm()
    if rec.feasible:
        for key, value in as_vllm_config(rec).items():
            save_config_value(f"local_runtime.vllm.{key}", value)
    settings = vllm_settings(load_config())
    ensure_vllm_venv(str(settings.get("python") or ""))
    model = str(settings.get("model") or rec.model or "").strip()
    if model:
        if job is not None:
            job["phase"] = "downloading"
            job["detail"] = f"Downloading {model}"
        ensure_hf_weights(model, job)


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

    return {"models": catalog_models(config)}


def set_vllm_model(hf_id: str) -> dict[str, Any]:
    from hermes_cli.vllm_runtime.inventory import apply_vllm_model

    try:
        return apply_vllm_model(hf_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def repo_is_cached(hf_id: str) -> bool:
    from hermes_cli.vllm_runtime.inventory import repo_is_cached as _cached

    return _cached(hf_id)


def download_vllm_weights(hf_id: str, job: dict | None = None) -> dict[str, Any]:
    from hermes_cli.vllm_runtime.inventory import ensure_hf_weights

    try:
        return ensure_hf_weights(hf_id, job)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def delete_vllm_model(hf_id: str) -> dict[str, Any]:
    from hermes_cli.vllm_runtime.inventory import delete_cached_repo

    try:
        delete_cached_repo(hf_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


def search_vllm_models(q: str, limit: int = 20) -> dict[str, Any]:
    from hermes_cli.vllm_runtime.inventory import search_hf_models

    try:
        return {"hits": search_hf_models(q, limit)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Hugging Face search unavailable: {exc}") from exc


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
