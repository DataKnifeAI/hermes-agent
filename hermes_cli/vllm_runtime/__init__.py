"""Managed vLLM local engine.

Isolated PyTorch/CUDA venv, supervised ``vllm serve`` on loopback, VRAM-tier
recommend. llama.cpp stays in ``hermes_cli/local_runtime/``; cross-engine policy
(stop the other server) lives in ``hermes_cli.local_engines``.
"""

from hermes_cli.vllm_runtime.bootstrap import (  # noqa: F401
    activate_vllm_provider, ensure_vllm_runtime, shutdown_vllm_runtime,
    start_managed_vllm)
from hermes_cli.vllm_runtime.recommend import recommend_vllm  # noqa: F401
from hermes_cli.vllm_runtime.supervisor import serve_argv  # noqa: F401
