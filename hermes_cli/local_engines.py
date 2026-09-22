"""Cross-engine policy for managed local inference.

llama.cpp (``hermes_cli.local_runtime``), GPU vLLM, and CPU vLLM each own one
supervised server. This module is the only place that knows all three:
starting an engine stops the others first. Persisting ``local_runtime.engine``
(the settings dropdown) does not stop a running supervisor. One Hermes-managed
serve at a time.

CLI, Desktop, and ``hermes serve`` all call these helpers — Desktop is not a
special case.
"""

from __future__ import annotations

from contextlib import suppress
import json
import logging
from pathlib import Path

from hermes_cli.vllm_runtime.device import (
    VLLM_ENGINES, engine_to_device, is_vllm_engine,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENGINE = "llamacpp"


def engine_from_config(config: dict | None) -> str:
    raw = ((config or {}).get("local_runtime") or {}).get("engine") or _DEFAULT_ENGINE
    name = str(raw).strip().lower().replace("_", "-")
    if name in VLLM_ENGINES:
        return name
    return _DEFAULT_ENGINE


def vllm_device_from_config(config: dict | None) -> str:
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    return engine_to_device(engine_from_config(config))


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


def stop_vllm_engine() -> None:
    """Stop in-process vLLM (GPU and CPU), then leftovers from another Hermes process."""
    from hermes_cli.vllm_runtime.bootstrap import shutdown_vllm_runtime
    from hermes_cli.vllm_runtime.supervisor import state_path

    shutdown_vllm_runtime()
    stop_state_pid(state_path("gpu"))
    stop_state_pid(state_path("cpu"))


def stop_configured_engine(config: dict | None = None) -> str:
    """Stop the configured engine only. Does not disable auto-start or touch the other engine."""
    name = engine_from_config(config)
    if is_vllm_engine(name):
        stop_vllm_engine()
    else:
        stop_llama_engine()
    return name


def ensure_managed_engine(config: dict | None = None, force: bool = False):
    """Boot the configured engine; stop the others first. Returns that supervisor or None."""
    name = engine_from_config(config)
    if is_vllm_engine(name):
        from hermes_cli.vllm_runtime.bootstrap import ensure_vllm_runtime

        stop_llama_engine()
        return ensure_vllm_runtime(config, force=force)
    from hermes_cli.local_runtime.bootstrap import ensure_local_runtime

    stop_vllm_engine()
    return ensure_local_runtime(config, force=force)


def shutdown_managed_engine() -> None:
    """Stop every managed server (teardown). Order does not matter."""
    stop_llama_engine()
    stop_vllm_engine()
