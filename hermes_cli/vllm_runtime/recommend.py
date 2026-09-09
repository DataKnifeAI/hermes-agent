"""VRAM-tier recommend for the managed vLLM engine.

Probes NVIDIA memory through ``local_runtime.hardware`` (nvidia-smi, then the
CUDA driver via ctypes) so a driver/library mismatch that kills ``nvidia-smi``
still sees the card. Host-independent math takes VRAM as data; callers that
need a live GPU mark the test ``linux_only``.

Recommend is llama.cpp Local Models' idea, retuned for vLLM: a short
official list (not six VRAM-bucket defaults), one hardware-fit
recommended row, everything else via HF search. Official ids are modern
instruct checkpoints vLLM actually serves — AWQ/FP8, a real tool parser,
64k floor. 8–12 GB cards stay ``feasible: false`` at that floor — never
a silent ctx shrink — and do not add a sixth official download. No GGUF.
"""

from __future__ import annotations

from dataclasses import dataclass

_GIB = 1 << 30
MIN_CONTEXT = 65536  # Hermes tool-loop floor; never silently drop below this.
# Shipped / no-probe default: the smallest 64k-feasible catalog row (16 GB).
# 24 GB probe still recommends the 14B tier; this id is first paint + config.yaml.
_DEFAULT_MODEL = "Qwen/Qwen3-8B-AWQ"
_DEFAULT_SERVED = "qwen3:8b"
_DEFAULT_PARSER = "hermes"
# Last-rung public AWQ when every official catalog id 401s/403s/gates.
_PUBLIC_FALLBACK = "Qwen/Qwen2.5-7B-Instruct-AWQ"
_PUBLIC_FALLBACK_SERVED = "qwen2.5:7b"
# Parsers vLLM's OpenAI-compat /v1/chat/completions actually implements.
TOOL_PARSERS = frozenset({"hermes", "llama3_json", "qwen3_xml", "qwen3_coder", "mistral"})


@dataclass(frozen=True)
class VllmTier:
    """One VRAM class. ``feasible_at_64k`` is the catalog row's 64k contract, not a
    silent ctx shrink: 8–12 GB cards stay infeasible rather than shipping 8k.

    ``catalog`` rows are the official Local Models list (llama.cpp-short).
    A non-catalog floor marker reuses the shipped default id so 8–12 GB
    stay infeasible without advertising a unique download.
    """

    id: str
    min_vram_bytes: int
    gpu_memory_utilization: float
    feasible_at_64k: bool
    quantization: str
    kv_cache_dtype: str
    model: str
    served_model_name: str
    tool_call_parser: str = _DEFAULT_PARSER
    max_model_len: int = MIN_CONTEXT
    catalog: bool = True


# Official list first so classify / served-name match the 16 GB 8B floor,
# not the 8 GB infeasible marker that reuses the same id. _pick_tier takes
# the highest min_vram that still fits — walk order must not matter.
# Qwen3.8-27B-FP8 is llama.cpp's top GGUF analog; FP8 + 64k KV is not a
# Q4 file, so it is the 80 GB row only.
TIERS: tuple[VllmTier, ...] = (
    VllmTier(
        "16gb", 16 * _GIB, 0.75, True, "awq", "fp8",
        "Qwen/Qwen3-8B-AWQ", "qwen3:8b",
    ),
    VllmTier(
        "24gb", 24 * _GIB, 0.75, True, "awq", "fp8",
        "Qwen/Qwen3-14B-AWQ", "qwen3:14b",
    ),
    VllmTier(
        "40gb", 40 * _GIB, 0.80, True, "awq", "fp8",
        "Qwen/Qwen3-32B-AWQ", "qwen3:32b",
    ),
    VllmTier(
        # Official FP8 checkpoint — vLLM reads quant from the repo; don't
        # also pass --quantization fp8 (that flag is for on-the-fly casts).
        "80gb", 80 * _GIB, 0.85, True, "", "fp8",
        "Qwen/Qwen3.8-27B-FP8", "qwen3.8:27b", "qwen3_coder",
    ),
    VllmTier(
        "below-64k", 8 * _GIB, 0.70, False, "awq", "fp8",
        _DEFAULT_MODEL, _DEFAULT_SERVED, catalog=False,
    ),
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
    def tool_call_parser(self) -> str:
        return self.tier.tool_call_parser if self.tier else _DEFAULT_PARSER

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


def catalog_tiers() -> tuple[VllmTier, ...]:
    """Official Local Models rows — the short list, not the 64k-floor marker."""
    return tuple(t for t in TIERS if t.catalog)


def tier_for_model(hf_id: str) -> VllmTier | None:
    """Prefer a catalog row when the 64k-floor marker reuses the same id."""
    hid = (hf_id or "").strip()
    if not hid:
        return None
    return next((t for t in TIERS if t.catalog and t.model == hid), None) or next(
        (t for t in TIERS if t.model == hid), None
    )


def _pick_tier(total_bytes: int) -> VllmTier | None:
    matching = [t for t in TIERS if total_bytes >= t.min_vram_bytes]
    if not matching:
        return None
    return max(matching, key=lambda t: t.min_vram_bytes)


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
        "tool_call_parser": rec.tool_call_parser,
    }


def official_catalog_ids() -> frozenset[str]:
    return frozenset(t.model for t in catalog_tiers() if t.model) | {
        _DEFAULT_MODEL, _PUBLIC_FALLBACK,
    }


def overlay_for_setup(hid: str, rec: VllmRecommendation) -> dict:
    """Recommend overlay with *hid* (official row or public fallback) as the model."""
    overlay = as_vllm_config(rec)
    overlay["model"] = hid
    matched = tier_for_model(hid)
    if matched is not None:
        overlay["served_model_name"] = matched.served_model_name
        overlay["quantization"] = matched.quantization
        overlay["tool_call_parser"] = matched.tool_call_parser
        overlay["gpu_memory_utilization"] = matched.gpu_memory_utilization
        overlay["kv_cache_dtype"] = matched.kv_cache_dtype
        overlay["max_model_len"] = matched.max_model_len
    elif hid == _PUBLIC_FALLBACK:
        overlay["served_model_name"] = _PUBLIC_FALLBACK_SERVED
        overlay["quantization"] = "awq"
        overlay["tool_call_parser"] = _DEFAULT_PARSER
    else:
        overlay["served_model_name"] = hid.rsplit("/", 1)[-1]
    return overlay


def resolve_public_setup(rec: VllmRecommendation | None = None) -> tuple[str, str | None, VllmRecommendation]:
    """VRAM-fit official id, or a known-public fallback if Hub rejects that row.

    Leftover ``local_runtime.vllm.model`` is never consulted. Offline / probe
    failure keeps the official id (fail open) so tests and air-gapped boxes
    still plan Qwen3-8B-AWQ.
    """
    picked = rec or recommend_vllm()
    wanted = (picked.model or "").strip() or _DEFAULT_MODEL
    from hermes_cli.vllm_runtime.inventory import hf_repo_access_issue

    candidates: list[str] = []
    for hid in (wanted, *(t.model for t in catalog_tiers()), _PUBLIC_FALLBACK):
        if hid and hid not in candidates:
            candidates.append(hid)
    first_issue = None
    for hid in candidates:
        issue = hf_repo_access_issue(hid)
        if issue is None:
            notice = None
            if hid != wanted and first_issue:
                notice = (
                    f"{wanted} is not publicly downloadable ({first_issue}). "
                    f"Using {hid} instead."
                )
            return hid, notice, picked
        if first_issue is None:
            first_issue = issue
    return wanted, first_issue, picked
