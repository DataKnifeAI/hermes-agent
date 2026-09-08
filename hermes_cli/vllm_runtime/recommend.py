"""VRAM-tier recommend for the managed vLLM engine.

Probes NVIDIA memory through ``local_runtime.hardware`` (nvidia-smi, then the
CUDA driver via ctypes) so a driver/library mismatch that kills ``nvidia-smi``
still sees the card. Host-independent math takes VRAM as data; callers that
need a live GPU mark the test ``linux_only``.
"""

from __future__ import annotations

from dataclasses import dataclass

_GIB = 1 << 30
MIN_CONTEXT = 65536  # Hermes tool-loop floor; never silently drop below this.
_DEFAULT_MODEL = "solidrust/Hermes-3-Llama-3.1-8B-AWQ"
_DEFAULT_SERVED = "hermes3:8b"


@dataclass(frozen=True)
class VllmTier:
    """One VRAM class. ``feasible_at_64k`` is the 8B-AWQ + 64k contract, not a
    silent ctx shrink: 8–12 GB cards stay infeasible rather than shipping 8k."""

    id: str
    min_vram_bytes: int
    gpu_memory_utilization: float
    feasible_at_64k: bool
    quantization: str
    kv_cache_dtype: str
    model: str = _DEFAULT_MODEL
    served_model_name: str = _DEFAULT_SERVED
    max_model_len: int = MIN_CONTEXT


# Highest matching tier wins. min_vram is a floor: recommend never returns a
# row whose min exceeds probed total.
TIERS: tuple[VllmTier, ...] = (
    VllmTier("8gb", 8 * _GIB, 0.70, False, "awq", "fp8"),
    VllmTier("12gb", 12 * _GIB, 0.70, False, "awq", "fp8"),
    VllmTier("16gb", 16 * _GIB, 0.75, True, "awq", "fp8"),
    VllmTier("24gb", 24 * _GIB, 0.75, True, "awq", "fp8"),
    VllmTier("40gb", 40 * _GIB, 0.80, True, "", ""),  # BF16 alt; no quant flag
)


@dataclass(frozen=True)
class NvidiaProbe:
    total_bytes: int
    free_bytes: int
    source: str  # nvidia-smi | libcuda | none
    error: str = ""


@dataclass(frozen=True)
class VllmRecommendation:
    probe: NvidiaProbe
    tier: VllmTier | None
    feasible: bool
    reason: str

    @property
    def model(self) -> str:
        return self.tier.model if self.tier else _DEFAULT_MODEL

    @property
    def served_model_name(self) -> str:
        return self.tier.served_model_name if self.tier else _DEFAULT_SERVED

    @property
    def max_model_len(self) -> int:
        return self.tier.max_model_len if self.tier else MIN_CONTEXT

    @property
    def gpu_memory_utilization(self) -> float:
        return self.tier.gpu_memory_utilization if self.tier else 0.75

    @property
    def quantization(self) -> str:
        return self.tier.quantization if self.tier else "awq"

    @property
    def kv_cache_dtype(self) -> str:
        return self.tier.kv_cache_dtype if self.tier else "fp8"


def probe_nvidia_vram() -> NvidiaProbe:
    """smi first; ctypes ``libcuda`` when smi is missing or returns empty.

    Tonight's class of failure: kernel module vs NVML library mismatch. smi
    exits non-zero while the 4090 is still there; the driver API still answers.
    """
    from hermes_cli.local_runtime import hardware

    smi = hardware._nvidia_vram()
    if smi is not None:
        total, free = smi
        return NvidiaProbe(total, free, "nvidia-smi")
    cuda = hardware._cuda_driver_pool()
    if cuda is not None:
        total, _integrated = cuda
        return NvidiaProbe(total, total, "libcuda")
    return NvidiaProbe(0, 0, "none", error="no_nvidia")


def _pick_tier(total_bytes: int) -> VllmTier | None:
    chosen = None
    for tier in TIERS:
        if total_bytes >= tier.min_vram_bytes:
            chosen = tier
    return chosen


def recommend_vllm(*, total_bytes: int | None = None,
                   probe: NvidiaProbe | None = None) -> VllmRecommendation:
    """Pick a tier. Pass ``total_bytes`` to skip the live GPU (tests, --apply dry-run)."""
    if probe is None:
        probe = (NvidiaProbe(total_bytes, total_bytes, "data")
                 if total_bytes is not None else probe_nvidia_vram())
    total = probe.total_bytes
    if total <= 0:
        return VllmRecommendation(probe, None, False, probe.error or "no_nvidia")
    tier = _pick_tier(total)
    if tier is None:
        return VllmRecommendation(probe, None, False, "vram_below_smallest_tier")
    if not tier.feasible_at_64k:
        return VllmRecommendation(probe, tier, False, "vram_below_64k_floor")
    return VllmRecommendation(probe, tier, True, "ok")


def as_vllm_config(rec: VllmRecommendation) -> dict:
    """``local_runtime.vllm`` overlay from a recommendation (caller writes config.yaml)."""
    return {
        "model": rec.model,
        "served_model_name": rec.served_model_name,
        "max_model_len": rec.max_model_len,
        "gpu_memory_utilization": rec.gpu_memory_utilization,
        "quantization": rec.quantization,
        "kv_cache_dtype": rec.kv_cache_dtype,
    }
