"""GPU vs official-CPU managed vLLM — device, ports, runtime leaf names.

One public engine id (``vllm``) with ``local_runtime.vllm.device`` ``gpu`` or
``cpu``. CUDA wheels stay in ``runtimes/vllm/.venv``, CPU wheels in
``runtimes/vllm-cpu/.venv``. Never ``--device cpu`` on the CUDA wheel.

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
