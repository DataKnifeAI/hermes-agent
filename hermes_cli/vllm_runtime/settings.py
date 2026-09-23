"""Device-keyed ``local_runtime.vllm`` settings.

One public engine (``vllm``). Chat and Turn on follow ``selected``
(``device`` is a write alias). Last-used serve args live under
``devices.<id>``. Later ``gpu:0`` / ``cpu:0`` / ``npu`` are more keys,
not more engines.

Old shared ``vllm.model`` / ``served_model_name`` / serve knobs migrate
on read onto the selected device only. A nested ``vllm.cpu`` block
(intermediate schema) becomes ``devices.cpu``; the leftover top-level
model is the GPU checkpoint, not a second copy of the CPU pick.
"""

from __future__ import annotations

from typing import Any

from hermes_cli.vllm_runtime.device import CPU, ENGINE_CPU, GPU, normalize_device

DEVICE_SERVE_KEYS = (
    "model",
    "served_model_name",
    "max_model_len",
    "quantization",
    "tool_call_parser",
    "kv_cache_dtype",
    "gpu_memory_utilization",
)
_PROCESS_KEYS = ("port", "host", "python")


def device_config_prefix(device: str | None) -> str:
    return f"local_runtime.vllm.devices.{normalize_device(device)}"


def _as_dict(value: Any) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _has_model(block: dict | None) -> bool:
    return bool(str((block or {}).get("model") or "").strip())


def _serve_subset(src: dict) -> dict:
    return {key: src[key] for key in DEVICE_SERVE_KEYS if key in src}


def _user_vllm(config: dict | None) -> dict:
    local = (config or {}).get("local_runtime") or {}
    if not isinstance(local, dict):
        return {}
    return _as_dict(local.get("vllm"))


def selected_device(config: dict | None) -> str:
    """Chat / Turn-on device. ``selected``, else ``device``, else GPU.

    Legacy ``engine: vllm-cpu`` is cpu until rewritten. ``selected`` is
    not in DEFAULT_CONFIG so a user ``device: cpu`` is not shadowed by a
    deep-merged default ``selected: gpu``.
    """
    local = (config or {}).get("local_runtime") or {}
    if not isinstance(local, dict):
        local = {}
    raw_engine = str(local.get("engine") or "").strip().lower().replace("_", "-")
    if raw_engine == ENGINE_CPU:
        return CPU
    vllm = _as_dict(local.get("vllm"))
    for key in ("selected", "device"):
        explicit = str(vllm.get(key) or "").strip().lower()
        if explicit in (GPU, CPU):
            return normalize_device(explicit)
    return GPU


def migrate_vllm_devices(vllm: dict | None, *, selected: str) -> dict[str, dict]:
    """Fold shared / legacy keys into ``devices``. In-memory; does not persist.

    * An existing ``devices.<id>.model`` wins for that id.
    * Legacy ``vllm.cpu.model`` becomes ``devices.cpu``.
    * Shared top-level serve keys attach to **one** device:
      - if CPU already has a pick, shared is the GPU checkpoint — unless
        it is only the shipped GPU default (deep-merge), which is not a
        live GPU pick;
      - otherwise attach to ``selected`` only, never both;
      - shipped GPU AWQ on a CPU-selected config is not an explicit CPU
        pick (CPU default stays Qwen3-4B).
    """
    raw = _as_dict(vllm)
    out: dict[str, dict] = {}
    for key, block in _as_dict(raw.get("devices")).items():
        name = str(key or "").strip().lower()
        if name and isinstance(block, dict):
            out[name] = dict(block)

    legacy_cpu = _as_dict(raw.get("cpu"))
    if _has_model(legacy_cpu) and not _has_model(out.get(CPU)):
        merged = dict(out.get(CPU) or {})
        merged.update(_serve_subset(legacy_cpu))
        out[CPU] = merged

    shared = _serve_subset(raw)
    shared_model = str(shared.get("model") or "").strip()
    if not shared_model:
        return out

    from hermes_cli.vllm_runtime.recommend import gpu_shipped_model

    selected = normalize_device(selected)
    cpu_filled = _has_model(out.get(CPU))
    shipped = shared_model == gpu_shipped_model()
    if cpu_filled:
        target = None if shipped else GPU
    elif selected == CPU and shipped:
        target = None
    else:
        target = selected
    if target and not _has_model(out.get(target)):
        merged = dict(out.get(target) or {})
        merged.update(shared)
        out[target] = merged
    return out


def _default_gpu_serve() -> dict:
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    return _serve_subset(DEFAULT_CONFIG["local_runtime"]["vllm"])


def _default_cpu_serve() -> dict:
    from hermes_cli.vllm_runtime.recommend import as_vllm_config, recommend_vllm_cpu

    return as_vllm_config(recommend_vllm_cpu())


def vllm_settings(config: dict | None = None, device: str | None = None) -> dict:
    """Process keys + this device's serve args, defaults applied after migrate.

    ``device`` overrides ``selected`` so a GPU snapshot cannot read the
    CPU pick. Switching ``selected`` does not copy models across devices.
    """
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    defaults = dict(DEFAULT_CONFIG["local_runtime"]["vllm"])
    user = _user_vllm(config)
    selected = selected_device(config)
    want = normalize_device(device) if device is not None else selected
    devices = migrate_vllm_devices(user, selected=selected)
    block = dict(devices.get(want) or {})

    out = {key: defaults.get(key) for key in _PROCESS_KEYS}
    out.update({key: user[key] for key in _PROCESS_KEYS if key in user})
    if not _has_model(block):
        block = _default_cpu_serve() if want == CPU else _default_gpu_serve()
    for key in DEVICE_SERVE_KEYS:
        if key in block:
            out[key] = block[key]
        elif key in defaults and key not in out:
            out[key] = defaults[key]
    return out


def persist_selected(device: str | None) -> None:
    """Write ``selected`` and the ``device`` alias. Does not copy models."""
    from cli import save_config_value

    chosen = normalize_device(device)
    save_config_value("local_runtime.vllm.selected", chosen)
    save_config_value("local_runtime.vllm.device", chosen)


def persist_device_overlay(device: str | None, overlay: dict | None) -> None:
    """Write serve args under ``devices.<id>`` only."""
    from cli import save_config_value

    prefix = device_config_prefix(device)
    for key, value in _as_dict(overlay).items():
        if key in DEVICE_SERVE_KEYS:
            save_config_value(f"{prefix}.{key}", value)


def ensure_device_model(config: dict | None, device: str | None) -> dict:
    """Persist this device's default when it has no explicit pick, then read.

    GPU empty → official AWQ by VRAM class. CPU empty → Qwen3-4B. An
    explicit pick (including Smol on cpu) is left alone.
    """
    want = normalize_device(device)
    user = _user_vllm(config)
    devices = migrate_vllm_devices(user, selected=selected_device(config))
    if _has_model(devices.get(want)):
        return vllm_settings(config, device=want)
    if want == CPU:
        overlay = _default_cpu_serve()
    else:
        from hermes_cli.vllm_runtime.recommend import as_vllm_config, recommend_vllm

        rec = recommend_vllm()
        overlay = as_vllm_config(rec) if rec.feasible else _default_gpu_serve()
    persist_device_overlay(want, overlay)
    from hermes_cli.config import load_config

    return vllm_settings(load_config(), device=want)
