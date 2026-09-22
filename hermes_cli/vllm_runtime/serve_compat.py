"""Which checkpoints each managed vLLM wheel is allowed to start.

CPU vLLM is the official ``+cpu`` wheel (``VLLM_TARGET_DEVICE=cpu`` in
``venv.py`` / ``serve_environ``). That process has no CUDA quant kernels.
Known failures on this wheel: AutoAWQ (``quant_method: awq``), compressed-tensors
AWQ-4bit, FP8 / MXFP4 / NVFP4, and a checkpoint that declares FP8 KV. The
CPU serve argv never passes ``--kv-cache-dtype`` (no GPU KV pool).

GPU vLLM is the CUDA venv. It serves AWQ when ``quant_method`` is awq, and it
passes ``--kv-cache-dtype fp8``. Fit on that list stays the VRAM estimator.
GGUF, EXL2, and DSpark drafts are refused on both devices — ``serve`` already
dies on them before a useful model load.

Incompatible rows are omitted from that device's catalog. They are not
``hide_by_default``: Show is for "too big", and a CUDA quant on the CPU wheel
is not a size problem. The checkpoint currently served on that device stays
listed so an in-use model does not disappear after its own RSS lands.
"""

from __future__ import annotations

from hermes_cli.vllm_runtime.device import CPU, normalize_device

# Quants the CPU wheel cannot execute. int4 is the compressed-tensors / 4-bit
# pack ``_quant_from_config`` and id tags already price as AWQ-class.
_CPU_REFUSED_QUANTS = frozenset({"awq", "gptq", "fp8", "mxfp4", "nvfp4", "int4"})

CPU_CUDA_QUANT_MSG = (
    "The CPU vLLM wheel cannot serve this CUDA quant — "
    "switch to GPU, or use a BF16 instruct model"
)


def _checkpoint_quant(
    hid: str,
    tags: list[str] | None,
    config: dict | None,
    safetensors: dict | None,
) -> str | None:
    from hermes_cli.vllm_runtime.inventory import (
        _quant_from_config, _quant_from_safetensors, parse_quantization,
    )

    # config.json wins (AutoAWQ, compressed-tensors 4-bit → awq). Id tokens
    # still count when that helper returns nothing: parse_quantization blanks
    # a compressed-tensors config before it reads "AWQ" out of the repo id.
    quant = _quant_from_config(config)
    if quant:
        return quant
    packed = _quant_from_safetensors(safetensors)
    if packed:
        return packed
    return parse_quantization(hid, tags)


def _declares_fp8_kv(config: dict | None) -> bool:
    """Checkpoint itself stores KV in FP8. A GPU serve flag is not this."""
    if not isinstance(config, dict):
        return False
    blobs: list[dict] = [config]
    qcfg = config.get("quantization_config")
    if isinstance(qcfg, dict):
        blobs.append(qcfg)
    for blob in blobs:
        for key in ("kv_cache_dtype", "kv_cache_quant_algo"):
            if "fp8" in str(blob.get(key) or "").lower():
                return True
        scheme = blob.get("kv_cache_scheme")
        if isinstance(scheme, str) and "fp8" in scheme.lower():
            return True
        if isinstance(scheme, dict):
            kind = " ".join(
                str(scheme.get(k) or "") for k in ("type", "dtype", "algo")
            )
            if "fp8" in kind.lower():
                return True
    return False


def incompatible_with_device(
    hid: str,
    device: str,
    *,
    tags: list[str] | None = None,
    config: dict | None = None,
    safetensors: dict | None = None,
) -> str | None:
    """Why *hid* must not be started on *device*. None when that wheel can try.

    GPU refusals are only the formats CUDA serve already rejects (GGUF, EXL2,
    DSpark). AWQ / FP8 stay offerable there; VRAM fit is a separate check.
    """
    from hermes_cli.vllm_runtime.inventory import unservable_reason

    blocked = unservable_reason(hid, tags)
    if blocked:
        return blocked
    if normalize_device(device) != CPU:
        return None
    quant = _checkpoint_quant(hid, tags, config, safetensors)
    if quant in _CPU_REFUSED_QUANTS or _declares_fp8_kv(config):
        return CPU_CUDA_QUANT_MSG
    return None
