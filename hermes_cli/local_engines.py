"""Cross-engine policy for managed local inference.

llama.cpp and GPU vLLM share one GPU, so starting either stops the other.
CPU vLLM does not: it stays up beside GPU vLLM or llama.cpp unless the user
turns that engine off. Persisting ``local_runtime.engine`` (the chat/default
dropdown) does not stop a running supervisor.

CLI, Desktop, and ``hermes serve`` all call these helpers — Desktop is not a
special case.
"""

from __future__ import annotations

from contextlib import suppress
import json
import logging
from pathlib import Path

from hermes_cli.vllm_runtime.device import (
    CPU, ENGINE_CPU, ENGINE_GPU, GPU, is_vllm_engine, normalize_device,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENGINE = "llamacpp"


def engine_from_config(config: dict | None) -> str:
    """Public engine page: ``llamacpp`` or ``vllm``.

    A legacy ``engine: vllm-cpu`` value is the vLLM page, not a third id.
    The device lives in ``local_runtime.vllm.device``.
    """
    raw = ((config or {}).get("local_runtime") or {}).get("engine") or _DEFAULT_ENGINE
    name = str(raw).strip().lower().replace("_", "-")
    if name in (ENGINE_GPU, ENGINE_CPU):
        return ENGINE_GPU
    return _DEFAULT_ENGINE


def vllm_device_from_config(config: dict | None) -> str:
    """``gpu`` or ``cpu``. Legacy ``engine: vllm-cpu`` is cpu until rewritten."""
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    section = (config or {}).get("local_runtime") or {}
    if not isinstance(section, dict):
        section = {}
    vllm = section.get("vllm") if isinstance(section.get("vllm"), dict) else {}
    raw = str(section.get("engine") or "").strip().lower().replace("_", "-")
    # Legacy page id wins over a deep-merged default ``device: gpu`` until
    # the engine id is rewritten to ``vllm``.
    if raw == ENGINE_CPU:
        return CPU
    explicit = str(vllm.get("device") or "").strip().lower()
    if explicit in (GPU, CPU):
        return normalize_device(explicit)
    return GPU


def stop_state_pid(path: Path) -> int | None:
    """SIGTERM the pid recorded in a supervisor state file (and its children).

    Only that pid — never a port scan, never a process-name sweep. Missing or
    stale state is a no-op. Returns the pid that was signalled, or None.
    """
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(state, dict):
        path.unlink(missing_ok=True)
        return None
    pid = int(state.get("pid") or 0)
    path.unlink(missing_ok=True)
    if pid <= 0:
        return None
    _terminate_pid_tree(pid)
    return pid


def _terminate_pid_tree(pid: int) -> None:
    with suppress(Exception):
        import psutil  # type: ignore

        if not psutil.pid_exists(pid):
            return
        proc = psutil.Process(pid)
        children = proc.children(recursive=True)
        with suppress(Exception):
            proc.terminate()
        for child in children:
            with suppress(Exception):
                child.terminate()
        _gone, alive = psutil.wait_procs([proc, *children], timeout=15)
        for leftover in alive:
            with suppress(Exception):
                leftover.kill()


def stop_llama_engine() -> None:
    """Stop in-process llama-server, then a leftover from another Hermes process."""
    from hermes_cli.local_runtime.bootstrap import shutdown_local_runtime
    from hermes_cli.local_runtime.supervisor import state_path

    shutdown_local_runtime()
    stop_state_pid(state_path())


def stop_vllm_device(device: str) -> None:
    """Stop one managed vLLM device. The sibling vLLM server is left running."""
    from hermes_cli.vllm_runtime.bootstrap import shutdown_vllm_runtime
    from hermes_cli.vllm_runtime.device import normalize_device
    from hermes_cli.vllm_runtime.supervisor import state_path

    dev = normalize_device(device)
    shutdown_vllm_runtime(dev)
    stop_state_pid(state_path(dev))


def stop_vllm_engine() -> None:
    """Stop both managed vLLM servers. Process teardown — not a single Turn off."""
    from hermes_cli.vllm_runtime.bootstrap import shutdown_vllm_runtime
    from hermes_cli.vllm_runtime.supervisor import state_path

    shutdown_vllm_runtime()
    stop_state_pid(state_path("gpu"))
    stop_state_pid(state_path("cpu"))


def stop_configured_engine(config: dict | None = None) -> str:
    """Stop the configured engine only. Does not disable auto-start or touch the other engine."""
    name = engine_from_config(config)
    if is_vllm_engine(name):
        stop_vllm_device(vllm_device_from_config(config))
    else:
        stop_llama_engine()
    return name


def ensure_managed_engine(config: dict | None = None, force: bool = False):
    """Boot the configured engine.

    GPU vLLM stops llama.cpp first (one GPU). CPU vLLM stops neither.
    llama.cpp stops GPU vLLM and leaves CPU vLLM running. Returns that
    supervisor or None.
    """
    name = engine_from_config(config)
    if is_vllm_engine(name):
        from hermes_cli.vllm_runtime.bootstrap import ensure_vllm_runtime

        if vllm_device_from_config(config) != CPU:
            stop_llama_engine()
        return ensure_vllm_runtime(config, force=force)
    from hermes_cli.local_runtime.bootstrap import ensure_local_runtime

    stop_vllm_device("gpu")
    return ensure_local_runtime(config, force=force)


def maybe_bind_cpu_compression(base_url: str, model: str = "") -> bool:
    """Point ``auxiliary.compression`` at the CPU serve when the user has not.

    Writes ``config.yaml`` only. A compression block that already names a
    base_url, a model, or a provider other than ``auto`` is left alone.
    Returns True when the file changed.
    """
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return False
    from hermes_cli.config import read_raw_config

    raw = read_raw_config()
    aux = raw.get("auxiliary") if isinstance(raw, dict) else None
    comp = aux.get("compression") if isinstance(aux, dict) else None
    if isinstance(comp, dict) and _compression_target_set(comp):
        return False
    from cli import save_config_value

    # provider auto drops a bare base_url. custom keeps the CPU endpoint.
    save_config_value("auxiliary.compression.provider", "custom")
    save_config_value("auxiliary.compression.base_url", url)
    served = str(model or "").strip()
    if served:
        save_config_value("auxiliary.compression.model", served)
    return True


def _compression_target_set(comp: dict) -> bool:
    if str(comp.get("base_url") or "").strip():
        return True
    if str(comp.get("model") or "").strip():
        return True
    provider = str(comp.get("provider") or "").strip().lower()
    return bool(provider and provider != "auto")


def shutdown_managed_engine() -> None:
    """Stop every managed server (teardown). Order does not matter."""
    stop_llama_engine()
    stop_vllm_engine()
