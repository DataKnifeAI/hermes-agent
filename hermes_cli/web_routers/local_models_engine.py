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
    from hermes_cli.vllm_runtime.supervisor import openai_base_url, vllm_settings
    from hermes_cli.vllm_runtime.venv import venv_ready

    cfg = config or {}
    section = cfg.get("local_runtime") or {}
    settings = vllm_settings(cfg)
    running = resolve_vllm_endpoint(wait_for_boot_s=0)
    served = str(settings.get("served_model_name") or "") or None
    occ = occupancy_payload()
    return {
        "engine": "vllm",
        "enabled": bool(section.get("enabled")),
        "venv_ready": venv_ready(),
        "runtime_installed": venv_ready(),
        "runtime_backend": "vllm" if venv_ready() else None,
        "server_running": running is not None,
        "server_base_url": (running or {}).get("base_url") or (
            openai_base_url(settings) if venv_ready() else None),
        "active_model_id": served if running else None,
        "served_model_name": served,
        "model": str(settings.get("model") or "") or None,
        "start_phase": None if running else _vllm_log_phase(),
        "last_error": occ["occupancy_message"],
        **occ,
        # Selected-engine status must not keep llama.cpp widgets as the truth.
        "tag": "",
        "configured_tag": "",
        "update_available": False,
        "loaded_models": {},
        "loading": {},
        "placement": {},
        "models": [],
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
    """Persist ``local_runtime.engine`` then stop the other supervisor."""
    from cli import save_config_value
    from hermes_cli.local_engines import stop_llama_engine, stop_vllm_engine

    engine = str(name or "").strip().lower()
    if engine not in _ENGINE_NAMES:
        raise HTTPException(status_code=400, detail="engine must be 'llamacpp' or 'vllm'")
    save_config_value("local_runtime.engine", engine)
    if engine == "vllm":
        stop_llama_engine()
    else:
        stop_vllm_engine()
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
        return
    stop_vllm_engine()
    lm._start_local_server(cfg, lm._SERVER_START_FAILED)


def apply_recommend_and_install() -> None:
    from cli import save_config_value
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.recommend import as_vllm_config, recommend_vllm
    from hermes_cli.vllm_runtime.supervisor import vllm_settings
    from hermes_cli.vllm_runtime.venv import ensure_vllm_venv

    rec = recommend_vllm()
    if rec.feasible:
        for key, value in as_vllm_config(rec).items():
            save_config_value(f"local_runtime.vllm.{key}", value)
    settings = vllm_settings(load_config())
    ensure_vllm_venv(str(settings.get("python") or ""))


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
