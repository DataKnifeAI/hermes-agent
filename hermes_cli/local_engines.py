"""Cross-engine policy for managed local inference.

llama.cpp (``hermes_cli.local_runtime``) and vLLM (``hermes_cli.vllm_runtime``)
each own one supervised server. This module is the only place that knows both:
switching ``local_runtime.engine`` stops the other process before starting the
new one. One GPU, one resident weights file.

CLI, Desktop, and ``hermes serve`` all call these helpers — Desktop is not a
special case.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEFAULT_ENGINE = "llamacpp"


def engine_from_config(config: dict | None) -> str:
    raw = ((config or {}).get("local_runtime") or {}).get("engine") or _DEFAULT_ENGINE
    name = str(raw).strip().lower()
    return "vllm" if name == "vllm" else _DEFAULT_ENGINE


def ensure_managed_engine(config: dict | None = None, force: bool = False):
    """Boot the configured engine; stop the other first. Returns that supervisor or None."""
    if engine_from_config(config) == "vllm":
        from hermes_cli.local_runtime.bootstrap import shutdown_local_runtime
        from hermes_cli.vllm_runtime.bootstrap import ensure_vllm_runtime

        shutdown_local_runtime()
        return ensure_vllm_runtime(config, force=force)
    from hermes_cli.local_runtime.bootstrap import ensure_local_runtime
    from hermes_cli.vllm_runtime.bootstrap import shutdown_vllm_runtime

    shutdown_vllm_runtime()
    return ensure_local_runtime(config, force=force)


def shutdown_managed_engine() -> None:
    """Stop both managed servers (teardown). Order does not matter; both must free VRAM."""
    from hermes_cli.local_runtime.bootstrap import shutdown_local_runtime
    from hermes_cli.vllm_runtime.bootstrap import shutdown_vllm_runtime

    shutdown_local_runtime()
    shutdown_vllm_runtime()
