"""``hermes local`` handlers — Phase 2 CLI over the managed llama.cpp / vLLM engines."""

from __future__ import annotations

import argparse
import sys


def cmd_local(args: argparse.Namespace) -> int:
    """Dispatch ``hermes local [subcommand]``. Bare ``hermes local`` is status."""
    sub = getattr(args, "local_command", None)
    handler = _HANDLERS.get(sub)
    if handler is None:
        print(f"Unknown local subcommand: {sub}", file=sys.stderr)
        print("Use one of: status, engine, recommend, install, start, stop, bench, use",
              file=sys.stderr)
        return 2
    from hermes_cli.vllm_runtime.occupancy import OccupyingLlmError

    try:
        return int(handler(args) or 0)
    except OccupyingLlmError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _config():
    from hermes_cli.config import load_config

    return load_config()


def _engine(config: dict | None = None) -> str:
    from hermes_cli.local_engines import engine_from_config

    return engine_from_config(config if config is not None else _config())


def cmd_local_status(args: argparse.Namespace) -> int:  # noqa: ARG001
    from hermes_cli.local_engines import engine_from_config
    from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint
    from hermes_cli.vllm_runtime.bench import last_bench
    from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint
    from hermes_cli.vllm_runtime.recommend import probe_nvidia_vram
    from hermes_cli.vllm_runtime.venv import venv_ready

    cfg = _config()
    engine = engine_from_config(cfg)
    section = cfg.get("local_runtime") or {}
    enabled = bool(section.get("enabled"))
    print(f"engine: {engine}")
    print(f"enabled: {enabled}")
    llama = None
    with _suppress():
        llama = resolve_llamacpp_endpoint(cfg, wait_for_boot_s=0)
    vllm = resolve_vllm_endpoint(wait_for_boot_s=0)
    if engine == "vllm":
        print(f"vllm: {vllm['base_url'] if vllm else 'not running'}")
        print(f"venv: {'ready' if venv_ready() else 'missing — hermes local install'}")
    else:
        print(f"llamacpp: {llama['base_url'] if llama else 'not running'}")
    probe = probe_nvidia_vram()
    if probe.total_bytes > 0:
        gib = probe.total_bytes / (1 << 30)
        print(f"gpu: {gib:.1f} GiB ({probe.source})")
    else:
        print(f"gpu: none ({probe.error or probe.source})")
    bench = last_bench()
    if bench:
        status = "ok" if bench.get("ok") else "fail"
        print(f"last bench: {status} {bench.get('url') or ''}".rstrip())
    return 0


def cmd_local_engine(args: argparse.Namespace) -> int:
    from cli import save_config_value

    name = getattr(args, "engine_name", None)
    if not name:
        print(_engine())
        return 0
    # Persist only. Starting the other engine is what stops this one.
    save_config_value("local_runtime.engine", name)
    print(f"engine: {name}")
    return 0


def cmd_local_recommend(args: argparse.Namespace) -> int:
    from hermes_cli.vllm_runtime.recommend import as_vllm_config, recommend_vllm

    rec = recommend_vllm()
    print(f"feasible: {rec.feasible}")
    print(f"reason: {rec.reason}")
    if rec.tier is not None:
        print(f"tier: {rec.tier.id}")
    print(f"model: {rec.model}")
    print(f"served_model_name: {rec.served_model_name}")
    print(f"max_model_len: {rec.max_model_len}")
    print(f"gpu_memory_utilization: {rec.gpu_memory_utilization}")
    if not getattr(args, "apply", False):
        return 0
    from cli import save_config_value

    for key, value in as_vllm_config(rec).items():
        save_config_value(f"local_runtime.vllm.{key}", value)
    print("wrote local_runtime.vllm.* to config.yaml")
    return 0


def cmd_local_install(args: argparse.Namespace) -> int:  # noqa: ARG001
    cfg = _config()
    if _engine(cfg) == "vllm":
        from hermes_cli.vllm_runtime.supervisor import vllm_settings
        from hermes_cli.vllm_runtime.venv import ensure_vllm_venv

        settings = vllm_settings(cfg)
        exe = ensure_vllm_venv(str(settings.get("python") or ""))
        print(f"vLLM ready: {exe}")
        return 0
    return _install_llamacpp(cfg)


def _install_llamacpp(cfg: dict) -> int:
    from hermes_cli.local_runtime.binaries import (
        BinaryResolutionError, default_tag, ensure_runtime_installed, select_backend)
    from hermes_cli.local_runtime.bootstrap import _detect_gpu_vendor

    section = cfg.get("local_runtime") or {}
    tag = section.get("tag") or default_tag()
    backend = section.get("backend") or "auto"
    if backend == "auto":
        backend = select_backend(_detect_gpu_vendor())
    try:
        dest = ensure_runtime_installed(tag, backend)
    except BinaryResolutionError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Linux NVIDIA often has no llama.cpp CUDA zip. "
            "Switch with `hermes local engine vllm` then `hermes local install`.",
            file=sys.stderr)
        return 1
    print(f"llama.cpp ready: {dest}")
    return 0


def cmd_local_start(args: argparse.Namespace) -> int:  # noqa: ARG001
    cfg = _config()
    if _engine(cfg) == "vllm":
        from hermes_cli.vllm_runtime.bootstrap import start_managed_vllm

        sup = start_managed_vllm(cfg, apply_recommend=False)
        print(f"vLLM listening at {sup.base_url}")
        return 0
    from cli import save_config_value
    from hermes_cli.config import load_config
    from hermes_cli.local_engines import ensure_managed_engine

    save_config_value("local_runtime.enabled", True)
    save_config_value("local_runtime.engine", "llamacpp")
    sup = ensure_managed_engine(load_config(), force=True)
    if sup is None:
        raise RuntimeError("managed llama.cpp did not start")
    url = getattr(sup, "base_url", "") or ""
    print(f"llama.cpp listening at {url}" if url else "llama.cpp started")
    return 0


def cmd_local_stop(args: argparse.Namespace) -> int:  # noqa: ARG001
    from hermes_cli.local_engines import stop_configured_engine

    name = stop_configured_engine(_config())
    print(f"stopped managed {name} (VRAM free; config unchanged)")
    return 0


def cmd_local_bench(args: argparse.Namespace) -> int:  # noqa: ARG001
    from hermes_cli.vllm_runtime.bench import probe_models, verify_tool_calls
    from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint
    from hermes_cli.vllm_runtime.supervisor import openai_base_url, vllm_settings

    cfg = _config()
    if _engine(cfg) != "vllm":
        from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint

        llama = resolve_llamacpp_endpoint(cfg, wait_for_boot_s=0)
        if not llama:
            raise RuntimeError("managed llama.cpp is not running")
        result = probe_models(llama["base_url"])
        if not result["ok"]:
            print(f"bench failed: {result['error']}", file=sys.stderr)
            return 1
        models = ", ".join(result["models"]) or "(none)"
        print(f"ok {result['url']} models={models}")
        return 0
    state = resolve_vllm_endpoint(wait_for_boot_s=0)
    url = (state or {}).get("base_url") or openai_base_url(vllm_settings(cfg))
    result = verify_tool_calls(url)
    models = ", ".join(result.get("models") or []) or "(none)"
    print(f"ok {result['url']} tool_calls={result.get('tool_calls')} models={models}")
    return 0


def cmd_local_use(args: argparse.Namespace) -> int:  # noqa: ARG001
    cfg = _config()
    if _engine(cfg) == "vllm":
        from hermes_cli.vllm_runtime.bench import verify_tool_calls
        from hermes_cli.vllm_runtime.bootstrap import activate_vllm_provider
        from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint

        url = activate_vllm_provider(cfg)
        state = resolve_vllm_endpoint(wait_for_boot_s=0)
        if state:
            verify_tool_calls(state["base_url"])
        print(f"provider: vllm  base_url: {url}")
        return 0
    from cli import save_config_value

    save_config_value("model.provider", "llamacpp")
    print("provider: llamacpp")
    return 0


def _suppress():
    from contextlib import suppress

    return suppress(Exception)


_HANDLERS = {
    **dict.fromkeys((None, "", "status", "list", "ls"), cmd_local_status),
    "engine": cmd_local_engine,
    "recommend": cmd_local_recommend,
    "install": cmd_local_install,
    "start": cmd_local_start,
    "stop": cmd_local_stop,
    "bench": cmd_local_bench,
    "use": cmd_local_use,
}
