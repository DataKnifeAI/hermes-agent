"""GPU vs official-CPU managed vLLM — device, ports, runtime leaf names.

One public engine id (``vllm``). Chat / Turn on follow
``local_runtime.vllm.selected`` (``device`` is a write alias). Serve args
live under ``devices.<id>`` — later ``gpu:0`` / ``npu`` are more keys,
not more engines. CUDA wheels stay in ``runtimes/vllm/.venv``, CPU wheels
in ``runtimes/vllm-cpu/.venv``. Never ``--device cpu`` on the CUDA wheel.

``vllm-cpu`` is a legacy stored engine id. Readers fold it to engine ``vllm``
plus device ``cpu``. It is not a third Local Models page.
"""

from __future__ import annotations

GPU = "gpu"
CPU = "cpu"
ENGINE_GPU = "vllm"
ENGINE_CPU = "vllm-cpu"  # legacy stored id; not a public engine page
# Public Local Models pages. ``vllm-cpu`` is accepted only as a legacy read/write fold.
PUBLIC_ENGINES = frozenset({"llamacpp", ENGINE_GPU})
VLLM_ENGINES = frozenset({ENGINE_GPU, ENGINE_CPU})
ALL_ENGINES = PUBLIC_ENGINES

# GPU 18435, CPU 18436; llama.cpp 18434. Never 8000/8080.
GPU_LISTEN_PORT = 18435
CPU_LISTEN_PORT = 18436
LLAMA_CPP_PORT = 18434
USER_SERVER_PORTS = frozenset({8000, 8080})
RESERVED_PORTS = frozenset({LLAMA_CPP_PORT, GPU_LISTEN_PORT, CPU_LISTEN_PORT}) | USER_SERVER_PORTS


def normalize_device(device: str | None) -> str:
    return CPU if str(device or "").strip().lower() == CPU else GPU


def engine_to_device(engine: str | None) -> str:
    return CPU if str(engine or "").strip().lower() == ENGINE_CPU else GPU


def device_to_engine(device: str | None) -> str:
    return ENGINE_CPU if normalize_device(device) == CPU else ENGINE_GPU


def is_vllm_engine(engine: str | None) -> bool:
    return str(engine or "").strip().lower() in VLLM_ENGINES


def runtime_leaf(device: str | None) -> str:
    return "vllm-cpu" if normalize_device(device) == CPU else "vllm"


def default_listen_port(device: str | None) -> int:
    return CPU_LISTEN_PORT if normalize_device(device) == CPU else GPU_LISTEN_PORT


def server_log_name(device: str | None) -> str:
    return "vllm-cpu-server.log" if normalize_device(device) == CPU else "vllm-server.log"


def install_log_name(device: str | None) -> str:
    return "vllm-cpu-install.log" if normalize_device(device) == CPU else "vllm-install.log"


# Picker labels and stable ``providers:`` keys. Chat stays ``provider: custom``
# and points ``model.base_url`` at the selected device. The keys are not the
# legacy engine id, even where the CPU string matches ``vllm-cpu``.
GPU_ENDPOINT_NAME = "vLLM GPU"
CPU_ENDPOINT_NAME = "vLLM CPU"
GPU_ENDPOINT_KEY = "vllm-gpu"
CPU_ENDPOINT_KEY = "vllm-cpu"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def managed_endpoint_name(device: str | None) -> str:
    """User-facing name of the managed serve for *device*."""
    return CPU_ENDPOINT_NAME if normalize_device(device) == CPU else GPU_ENDPOINT_NAME


def managed_endpoint_key(device: str | None) -> str:
    """``providers:`` key that belongs to one device and is never the other."""
    return CPU_ENDPOINT_KEY if normalize_device(device) == CPU else GPU_ENDPOINT_KEY


def _loopback_port(url: str | None) -> int | None:
    from urllib.parse import urlparse

    text = str(url or "").strip()
    if not text:
        return None
    try:
        parsed = urlparse(text)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if host not in _LOOPBACK_HOSTS and not host.startswith("127."):
        return None
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return None


def managed_endpoint_name_for_url(url: str | None) -> str | None:
    """``vLLM GPU`` / ``vLLM CPU`` when *url* is a managed loopback serve.

    Default ports win. A rebound ephemeral port matches the live supervisor
    state for that device. Other hosts and llama.cpp / Ollama ports stay
    unnamed so a user endpoint is not relabeled.
    """
    port = _loopback_port(url)
    if port is None or port in (LLAMA_CPP_PORT, 11434) or port in USER_SERVER_PORTS:
        return None
    if port == CPU_LISTEN_PORT:
        return CPU_ENDPOINT_NAME
    if port == GPU_LISTEN_PORT:
        return GPU_ENDPOINT_NAME
    try:
        from hermes_cli.vllm_runtime.endpoint import _state_endpoint
    except Exception:
        return None
    target = str(url or "").strip().rstrip("/").lower()
    for device in (GPU, CPU):
        try:
            state = _state_endpoint(device)
        except Exception:
            continue
        live = str((state or {}).get("base_url") or "").strip().rstrip("/").lower()
        if live and live == target:
            return managed_endpoint_name(device)
    return None
