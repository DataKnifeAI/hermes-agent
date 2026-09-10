"""HF-id inventory for the managed vLLM engine.

Cached Hugging Face weights (not GGUF) are the library. Search is the HF
text-generation firehose. Delete removes only that repo's hub cache dir.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HF = "https://huggingface.co"
_TIMEOUT_S = 15
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)[Bb](?:\b|[^a-zA-Z]|$)")
# Qwen-style MoE ids put active experts after the total (``30B-A3B``).
_ACTIVE_EXPERTS_RE = re.compile(r"A\d+(?:\.\d+)?B", re.I)
_CTX_RE = re.compile(r"(?<![a-z0-9])(32|64|128|131)\s*k\b", re.I)
# Bytes/param including a small framework overhead. Missing quant → no guess
# (unknown fit beats a lying "Fits your GPU").
_QUANT_BYTES = {
    "awq": 0.55,
    "gptq": 0.55,
    "int4": 0.55,
    # MXFP4 is 4.25-bit + block scales. Official gpt-oss-20b is ~13 GiB
    # on disk (sized for a 16 GB card), not dense BF16 / Hub usedStorage.
    "mxfp4": 0.55,
    "nvfp4": 0.55,
    "fp8": 1.1,
    "int8": 1.1,
    "bf16": 2.2,
    "fp16": 2.2,
}
_QUANT_TOKENS = (
    ("mxfp4", "mxfp4"),
    ("nvfp4", "nvfp4"),
    ("awq", "awq"),
    ("gptq", "gptq"),
    ("fp8", "fp8"),
    ("int4", "int4"),
    ("4-bit", "int4"),
    ("4bit", "int4"),
    ("int8", "int8"),
    ("8-bit", "int8"),
    ("8bit", "int8"),
    ("bf16", "bf16"),
    ("fp16", "fp16"),
)
_SAFETENSORS_DTYPE = {
    "BF16": "bf16",
    "F16": "fp16",
    "F8": "fp8",
    "F8_E4M3": "fp8",
    "F8_E5M2": "fp8",
    "I8": "int8",
    "U8": "int8",
}
_DTYPE_WIDTH = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "F8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1,
}
# List/search ``expand`` — usedStorage is model_info-only and 400s the list API.
# createdAt is first publish (“released”); lastModified is last push — do not
# substitute one for the other.
_HF_SEARCH_EXPAND = (
    "safetensors", "cardData", "config", "tags",
    "downloads", "likes", "lastModified", "createdAt", "gated",
    "pipeline_tag", "library_name",
)
_HF_JSON_CACHE: dict[str, tuple[float, object]] = {}
_HF_JSON_CACHE_TTL_S = 300
_HF_JSON_CACHE_MAX = 128
_HF_CARD_TIMEOUT_S = 4
_MIN_WEIGHT_BYTES = 100 << 20  # ignore tokenizer-only / empty cache dirs
# Hub leftovers from a gated 401 still have config/tokenizer; weights do not.
_WEIGHT_FILE_SUFFIXES = frozenset({
    ".safetensors", ".bin", ".pt", ".pth", ".npz", ".gguf",
})
_MAX_CAPS = 5
_TOOL_TAGS = frozenset({
    "function calling", "function-calling", "function_calling",
    "tool-use", "tool_use", "tool-calling", "tool_calling",
    "tools", "tool",
})
_CODE_TAGS = frozenset({
    "code", "coding", "coder", "code-generation", "codeqwen", "qwen-coder",
})
_VISION_TAGS = frozenset({
    "vision", "image-text-to-text", "image-text", "multimodal",
    "visual-question-answering",
})
_INSTRUCT_TAGS = frozenset({
    "instruct", "instruction-tuned", "conversational", "chat",
})
_QUANT_CAP_ORDER = ("mxfp4", "nvfp4", "awq", "gptq", "fp8", "int4", "int8", "bf16", "fp16")
_SKIP_DOWNLOAD_SUFFIX = frozenset({
    ".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".html", ".txt",
})


def hf_hub_dir() -> Path:
    """Hub cache root. Honors ``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE`` — not a Hermes env."""
    explicit = (os.environ.get("HUGGINGFACE_HUB_CACHE") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    home = (os.environ.get("HF_HOME") or "").strip()
    if home:
        root = Path(home).expanduser()
        return root if root.name == "hub" else root / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_from_cache_name(name: str) -> str | None:
    if not name.startswith("models--"):
        return None
    return name[len("models--"):].replace("--", "/")


def _cache_name(repo: str) -> str:
    return "models--" + repo.strip().replace("/", "--")


def _dir_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _current_snapshot(path: Path) -> Path | None:
    """Hub ``refs/main`` snapshot, else the only / newest snapshot dir."""
    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        return None
    refs = path / "refs"
    for name in ("main", "master"):
        ref = refs / name
        if not ref.is_file():
            continue
        try:
            rev = ref.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        snap = snapshots / rev
        if rev and snap.is_dir():
            return snap
    try:
        kids = [p for p in snapshots.iterdir() if p.is_dir()]
    except OSError:
        return None
    if not kids:
        return None
    if len(kids) == 1:
        return kids[0]
    return max(kids, key=lambda p: p.stat().st_mtime_ns)


def _snapshot_shard_bytes(snap: Path) -> int:
    """Weight bytes vLLM loads from one snapshot — not original/ or metal/.

    gpt-oss ships ~13 GiB index shards plus duplicate ``original/`` and
    ``metal/`` packs. Summing every weight file (or all hub blobs) is ~38 GiB.
    """
    if not snap.is_dir():
        return 0
    names: set[str] = set()
    idx = snap / "model.safetensors.index.json"
    if idx.is_file():
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            wmap = data.get("weight_map")
            if isinstance(wmap, dict):
                names = {str(v) for v in wmap.values() if v}
    total = 0
    if names:
        for name in names:
            child = snap / name
            try:
                size = child.stat().st_size
            except OSError:
                continue
            if size > 0:
                total += size
        return total
    try:
        children = list(snap.iterdir())
    except OSError:
        return 0
    for child in children:
        if not child.is_file() or child.suffix.lower() not in _WEIGHT_FILE_SUFFIXES:
            continue
        try:
            total += child.stat().st_size
        except OSError:
            continue
    return total


def _hub_repo_bytes(path: Path) -> int:
    """Weight bytes in one HF hub cache dir.

    Hub layout is ``blobs/`` plus ``snapshots/`` symlinks into those blobs.
    ``Path.stat()`` follows the links, so walking the repo root counted
    every shard twice and a 5.7 GiB AWQ looked like 11.4 GiB. ``blobs/``
    also keeps extra snapshot variants — prefer the current snapshot's
    index / top-level shards (gpt-oss-20b ~13 GiB, not ~38 GiB).
    """
    snap = _current_snapshot(path)
    if snap is not None:
        sized = _snapshot_shard_bytes(snap)
        if sized > 0:
            return sized
    blobs = path / "blobs"
    if blobs.is_dir():
        return _dir_bytes(blobs)
    snapshots = path / "snapshots"
    if snapshots.is_dir():
        return _dir_bytes(snapshots)
    return _dir_bytes(path)


def _human_gb(n: int | float) -> str:
    return f"{n / (1 << 30):.1f} GB"


def _cache_has_weights(path: Path) -> bool:
    """True when the hub dir has a real weight file, not tokenizer/README leftovers."""
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            if Path(name).suffix.lower() not in _WEIGHT_FILE_SUFFIXES:
                continue
            try:
                if (Path(root) / name).stat().st_size > 0:
                    return True
            except OSError:
                continue
    return False


def repo_is_gated(payload: dict | None) -> bool:
    """HF ``gated`` is bool | ``auto`` | ``manual`` — anything but explicit false."""
    if not isinstance(payload, dict):
        return False
    gated = payload.get("gated")
    if gated in (False, None, 0, "", "false", "False"):
        return False
    return bool(gated)


def list_cached_repos() -> list[dict[str, Any]]:
    """Repos with weights already in the HF hub cache."""
    root = hf_hub_dir()
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    for child in children:
        if not child.is_dir():
            continue
        repo = _repo_from_cache_name(child.name)
        if not repo or not _cache_has_weights(child):
            continue
        size = _hub_repo_bytes(child)
        rows.append({
            "id": repo,
            "size_bytes": size,
            "size_label": _human_gb(size) if size else "—",
            "cached": True,
        })
    rows.sort(key=lambda r: r["id"].lower())
    return rows


def cached_repo_ids() -> set[str]:
    return {r["id"] for r in list_cached_repos()}


def repo_is_cached(repo: str) -> bool:
    return (repo or "").strip() in cached_repo_ids()


def parse_param_billions(text: str) -> float | None:
    """Last ``Nb`` / ``N.NB`` token — ``Qwen3.8-27B`` → 27, not 3.8.

    Skips MoE active-expert tokens (``30B-A3B`` → 30, not 3).
    """
    cleaned = _ACTIVE_EXPERTS_RE.sub(" ", text or "")
    hits = _PARAM_RE.findall(cleaned)
    if not hits:
        return None
    try:
        return float(hits[-1])
    except ValueError:
        return None


def parse_quantization(repo: str, tags: list[str] | None = None) -> str | None:
    blob = f"{repo} {' '.join(tags or [])}".lower()
    for token, quant in _QUANT_TOKENS:
        if token in blob:
            # HF tags official gpt-oss as both mxfp4 and 8-bit; 8-bit must
            # not win and price a 4.25-bit MoE as dense INT8.
            if quant == "int8" and ("mxfp4" in blob or "nvfp4" in blob or "gpt-oss" in blob):
                continue
            return quant
    # Official OpenAI gpt-oss checkpoints are MXFP4; the id has no quant token.
    if "gpt-oss" in blob:
        return "mxfp4"
    return None


def param_billions_from_safetensors(safetensors: dict | None) -> float | None:
    """``safetensors.total`` / summed ``parameters`` — param count, not file bytes.

    Packed AWQ / compressed-tensors indexes often publish ``total`` far below
    the I32 weight count (a 36B repo with ``total`` ≈ 7B). That is not a
    parameter count — fall through so the id's ``Nb`` token can win.
    """
    if not isinstance(safetensors, dict):
        return None
    params = safetensors.get("parameters")
    total = safetensors.get("total")
    i32 = 0.0
    if isinstance(params, dict):
        raw_i32 = params.get("I32", params.get("i32"))
        if isinstance(raw_i32, (int, float)):
            i32 = float(raw_i32)
        if (
            _awq_packed_safetensors(params)
            and isinstance(total, (int, float))
            and i32 > float(total)
        ):
            return None
    if isinstance(total, (int, float)) and total >= 10_000_000:
        return float(total) / 1_000_000_000
    if isinstance(params, dict):
        summed = sum(v for v in params.values() if isinstance(v, (int, float)))
        if summed >= 10_000_000:
            return float(summed) / 1_000_000_000
    return None


# Llama-3.1-8B BF16 KV at 64k is 8 GiB (2 × 32 × 8 × 128 × 65536 × 2).
# vLLM CUDA graphs / activations add ~3 GiB. A percent of the weight file
# undercounts: 15 GiB on-disk BF16 (params × 2) + 55% was 23 GiB — a lying
# Fits on a 24 GB 4090. KV does not shrink with AWQ/GPTQ.
# Serve still passes --kv-cache-dtype fp8, but fit KV stays BF16 so a
# 15 GiB Nous pack cannot sneak under 24 GB (measured 8B AWQ EngineCore
# was 19.6 GiB on a 4090 — the utilization pool, not a shrinkable KV).
_KV_64K_LLAMA8B = 8 * (1 << 30)
_VLLM_RUNTIME_BYTES = 3 * (1 << 30)
_KV_AND_RUNTIME_64K = _KV_64K_LLAMA8B + _VLLM_RUNTIME_BYTES
# vLLM pre-allocates gpu_memory_utilization of the card, then CUDA graphs
# sit outside that pool. Measured on a 24564 MiB 4090 at util 0.75:
# EngineCore 20082 MiB ≈ 0.75 × card + 1.6 GiB graphs. Fit compares the
# 64k floor against this usable budget, not the sticker 24.000 GiB.
_VLLM_GPU_UTIL = 0.75
_VLLM_GRAPH_BYTES = 2 * (1 << 30)
# Official GLM-4-9B / Z1-9B: 40L × 2 KV heads × 128. HF expand=config
# ships model_type + tokenizer only — no layers/heads — so search used
# the Llama-8B 8 GiB KV floor on ~17.5 GiB BF16 (17.5+8+3 ≈ 28.5, a
# lying Too big). YaRN 64k/128k is official. Do not reuse for 32B / 4.5+.
_GLM4_9B_KV = {
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "hidden_size": 4096,
    "head_dim": 128,
}
_GLM4_9B_ID_RE = re.compile(r"(?:^|/)(?:glm-4-9b|glm-z1-9b)(?:[-_]|$)", re.I)
_GLM45_PLUS_RE = re.compile(r"glm-4\.[5-9]", re.I)


def estimate_min_vram_bytes(
    params_b: float, quant: str, *, config: dict | None = None,
) -> int:
    """Conservative 64k-floor VRAM from param count + known quant. Not a promise."""
    bpp = _QUANT_BYTES.get(quant)
    if bpp is None:
        raise ValueError(f"unknown quant {quant}")
    weight = int(params_b * 1_000_000_000 * bpp)
    return _vram_from_weight_bytes(weight, config=config)


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value == int(value):
        return int(value)
    return None


def _kv_from_config(config: dict | None, *, elem_bytes: int = 2) -> int | None:
    """64k K+V bytes from HF config. None when layers/heads/head_dim are missing."""
    if not isinstance(config, dict) or elem_bytes <= 0:
        return None
    layers = _positive_int(
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("num_layers")
    )
    kv_heads = _positive_int(
        config.get("num_key_value_heads")
        or config.get("num_kv_heads")
        or config.get("multi_query_group_num")
    )
    attn = _positive_int(
        config.get("num_attention_heads") or config.get("n_head")
    )
    if kv_heads is None:
        kv_heads = attn
    hidden = _positive_int(config.get("hidden_size") or config.get("d_model"))
    head_dim = _positive_int(config.get("head_dim") or config.get("kv_channels"))
    if head_dim is None and hidden is not None and attn is not None and attn > 0:
        head_dim = hidden // attn
    if layers is None or kv_heads is None or head_dim is None:
        return None
    return 2 * layers * kv_heads * head_dim * 65536 * int(elem_bytes)


def _known_kv_config(
    hid: str, *, params: float | None = None, config: dict | None = None,
) -> dict | None:
    """Official GLM-4/Z1 9B geometry when HF omitted layers/KV heads."""
    blob = (hid or "").replace("_", "-")
    if _GLM45_PLUS_RE.search(blob):
        return None
    if _GLM4_9B_ID_RE.search(blob):
        return dict(_GLM4_9B_KV)
    model_type = ""
    if isinstance(config, dict):
        model_type = str(config.get("model_type") or "").lower().replace("-", "")
    if model_type == "glm4" and params is not None and 8.0 <= params <= 10.5:
        return dict(_GLM4_9B_KV)
    return None


def _resolve_kv_config(
    hid: str, config: dict | None, params: float | None = None,
) -> dict | None:
    """Prefer published KV geometry; fill official GLM-4/Z1 9B when HF omitted it."""
    if _kv_from_config(config) is not None:
        return config
    known = _known_kv_config(hid, params=params, config=config)
    if known is None:
        return config if isinstance(config, dict) else None
    if isinstance(config, dict):
        merged = dict(config)
        merged.update(known)
        return merged
    return known


def _vram_from_weight_bytes(weight: int, *, config: dict | None = None) -> int:
    """weights + 64k KV + vLLM working set. KV is not a fraction of the file.

    The 8 GiB Llama-8B floor is only the fallback when KV geometry is
    unknown. 2-head GQA (GLM-4-9B) is ~2.5 GiB at 64k — do not pad it.
    """
    kv = _kv_from_config(config)
    reserve = (kv + _VLLM_RUNTIME_BYTES) if kv else _KV_AND_RUNTIME_64K
    return int(weight) + reserve


def usable_vllm_bytes(total_vram: int, *, util: float = _VLLM_GPU_UTIL) -> int:
    """Bytes vLLM can actually hold at the default 64k serve flags.

    ``gpu_memory_utilization`` (0.75) is the engine pool; CUDA graphs add a
    couple of GiB outside it. A 24564 MiB 4090 is 12 MiB under 24 GiB — using
    the sticker tier floor as VRAM was a lying Too big for 14B AWQ.
    """
    if total_vram <= 0:
        return 0
    pool = int(total_vram * util)
    return min(int(total_vram), pool + _VLLM_GRAPH_BYTES)


def _fit_on_probe(min_vram: int, total_vram: int) -> str:
    """Fits when the 64k floor is at or under the card — not the 0.75 pool.

    Qwen3-14B-AWQ (9.3 GiB weights + 64k KV) started on a 24564 MiB 4090 at
    64k. Comparing against ``usable_vllm_bytes`` (pool + 2 GiB graphs) called
    that Too big by 0.3 GiB. The card is the budget; utilization is a serve
    flag, not a shrink-to-fit.
    """
    if total_vram <= 0 or min_vram <= 0:
        return "unknown"
    return "fits-gpu" if min_vram <= total_vram else "too-big"


def _format_param_label(params: float | None) -> str | None:
    if params is None:
        return None
    if abs(params - round(params)) < 0.05:
        return f"{int(round(params))}B"
    return f"{params:g}B"


def _fit_why_label(hid: str, quant: str | None, params: float | None, min_vram: int) -> str:
    """Hover fragment: ``BF16 8B + 64k KV needs ~27.0 GB``."""
    dtype = (quant or "").upper() or None
    size = _format_param_label(params) or _format_param_label(parse_param_billions(hid))
    if dtype and size:
        head = f"{dtype} {size} + 64k KV"
    elif dtype:
        head = f"{dtype} + 64k KV"
    elif size:
        head = f"{size} + 64k KV"
    else:
        head = "64k KV"
    if min_vram:
        return f"{head} needs ~{_human_gb(min_vram)}"
    return head


def _base_model_blob(card_data: dict | None) -> str:
    if not isinstance(card_data, dict):
        return ""
    raw = card_data.get("base_model")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return " ".join(str(x) for x in raw if x)
    return ""


def _compressed_tensors_bits(qcfg: dict) -> int | None:
    """llmcompressor W4 lives in ``config_groups.*.weights.num_bits``, not ``bits``."""
    bits = qcfg.get("bits")
    if isinstance(bits, int) and bits > 0:
        return bits
    groups = qcfg.get("config_groups")
    if not isinstance(groups, dict):
        return None
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        weights = group.get("weights")
        if isinstance(weights, dict):
            n = weights.get("num_bits")
            if isinstance(n, int) and n > 0:
                return n
    return None


def _quant_from_config(config: dict | None) -> str | None:
    if not isinstance(config, dict):
        return None
    qcfg = config.get("quantization_config")
    if not isinstance(qcfg, dict):
        return None
    method = str(qcfg.get("quant_method") or "").lower().replace("_", "-")
    bits = qcfg.get("bits")
    if method in {"awq", "gptq", "fp8", "mxfp4", "nvfp4"}:
        return method
    if method == "compressed-tensors":
        n = _compressed_tensors_bits(qcfg)
        fmt = str(qcfg.get("format") or "").lower()
        if n == 4 or "pack-quantized" in fmt:
            return "awq"
        if n == 8:
            return "int8"
    if bits == 4:
        return "int4"
    if bits == 8:
        return "int8"
    return None


def _awq_packed_safetensors(params: dict) -> bool:
    """AWQ shards store packed int32 + a BF16/F16 scale — not a unique dtype."""
    keys = {str(k).upper() for k in params}
    return "I32" in keys and bool(keys & {"BF16", "F16"})


def _packed_awq_file_bytes(params: dict) -> int:
    """On-disk bytes for HF's I32+BF16 AWQ index.

    Hub counts 4-bit weights as I32 *elements*, not int32 values. Each is
    0.5 bytes; scales stay BF16/F16. ``total × 0.55`` on a compressed-tensors
    index under-prices a 21 GiB 36B pack as ~3.6 GiB (the ~7B ``total``).
    """
    total = 0
    for key, value in params.items():
        if not isinstance(value, (int, float)):
            continue
        kind = str(key).upper()
        if kind == "I32":
            total += int(value * 0.5)
        else:
            total += int(value) * _DTYPE_WIDTH.get(kind, 0)
    return total


def _quant_from_safetensors(safetensors: dict | None) -> str | None:
    """Single-dtype packs, or the I32+BF16 AWQ layout Hugging Face publishes."""
    if not isinstance(safetensors, dict):
        return None
    params = safetensors.get("parameters")
    if not isinstance(params, dict) or not params:
        return None
    if _awq_packed_safetensors(params):
        return "awq"
    mapped = {_SAFETENSORS_DTYPE.get(str(k).upper()) for k in params}
    mapped.discard(None)
    if len(mapped) == 1:
        return next(iter(mapped))
    return None


def _context_label(blob: str, config: dict | None) -> str | None:
    ranks: list[tuple[int, str]] = []
    max_pos = config.get("max_position_embeddings") if isinstance(config, dict) else None
    if isinstance(max_pos, int) and max_pos >= 24_000:
        if max_pos >= 100_000:
            ranks.append((128, "128k"))
        elif max_pos >= 48_000:
            ranks.append((64, "64k"))
        else:
            ranks.append((32, "32k"))
    for match in _CTX_RE.finditer(blob):
        n = int(match.group(1))
        ranks.append((128, "128k") if n >= 128 else (n, f"{n}k"))
    if not ranks:
        return None
    return max(ranks, key=lambda item: item[0])[1]


def _has_token(blob: str, token: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", blob))


def _capability_tags(
    *,
    hid: str,
    tag_list: list[str],
    quant: str | None,
    matched: Any,
    config: dict | None,
    pipeline_tag: str,
) -> list[str]:
    """HF tags / id tokens only — never a model-name hardcode (no 'hermes' → tools)."""
    tags_l = {t.lower() for t in tag_list}
    blob = f"{hid} {' '.join(tag_list)} {pipeline_tag}".lower()
    caps: list[str] = []
    if quant in _QUANT_CAP_ORDER:
        caps.append(quant)
    instruct = (
        matched is not None
        or bool(tags_l & _INSTRUCT_TAGS)
        or "instruct" in blob
        or "-chat" in blob
        or "chat-" in blob
    )
    if instruct:
        caps.append("instruct")
    tools = matched is not None or bool(tags_l & _TOOL_TAGS)
    if tools:
        caps.append("tools")
    vision = (
        pipeline_tag.lower() in _VISION_TAGS
        or bool(tags_l & _VISION_TAGS)
        or "-vl-" in blob
        or "-vision-" in blob
    )
    omni = _has_token(blob, "omni")
    if omni:
        caps.append("omni")
    elif vision:
        caps.append("vision")
    moe = False
    if isinstance(config, dict):
        experts = config.get("num_experts") or config.get("num_local_experts")
        if isinstance(experts, int) and experts > 1:
            moe = True
        elif str(config.get("model_type") or "").lower().replace("-", "_") == "gpt_oss":
            moe = True
    if not moe and any("moe" in t.lower().replace("-", "_").split("_") for t in tag_list):
        moe = True
    elif not moe and ("-moe-" in blob or blob.endswith("-moe")):
        moe = True
    if moe:
        caps.append("moe")
    ctx = _context_label(blob, config)
    if ctx:
        caps.append(ctx)
    coding = bool(tags_l & _CODE_TAGS) or "-coder-" in blob or _has_token(blob, "coder")
    if coding:
        caps.append("coding")
    return caps[:_MAX_CAPS]


# Shared with Start/Use so the pane and the supervisor say the same thing.
UNSERVABLE_FORMAT_MSG = (
    "vLLM cannot serve this format — Download an AWQ or FP8 instruct model"
)
# DSpark / DFlash HF ids are speculative-decode drafts. Serving one as the
# target dies in qwen3_dspark.py: speculative_config is None.
UNSERVABLE_DSPARK_MSG = (
    "This is a DSpark speculative draft, not a standalone model — "
    "Use the matching instruct checkpoint (without DSpark), or an AWQ / FP8 instruct model"
)
GATED_DOWNLOAD_MSG = (
    "This Hugging Face repo is gated — sign in at huggingface.co and request "
    "access. Hermes will not download it unsigned."
)
TOO_BIG_USE_MSG = "This model is too big for this GPU at the 64k tool-loop floor"
NEEDS_AWQ_MSG = (
    "This full-precision checkpoint is too big for this GPU at the 64k "
    "tool-loop floor — Download an AWQ or FP8 instruct model"
)
MISSING_REPO_MSG = (
    "Hugging Face has no such repo — or a required file is missing"
)
HF_BAD_REQUEST_MSG = (
    "Hugging Face rejected this repo (gated, invalid id, or a missing file)"
)
DOWNLOAD_VERIFY_PHASE = "verifying"
DOWNLOAD_VERIFY_DETAIL = "Finishing download"
_HTTP_ERROR_RE = re.compile(r"HTTP Error (\d{3})", re.I)
_CLIENT_ERROR_RE = re.compile(r"\b([45]\d{2}) Client Error", re.I)


def _hf_origin(exc: BaseException, blob: str) -> bool:
    """True when *exc* is a Hugging Face Hub failure, not local vLLM 4xx."""
    url = str(getattr(exc, "url", "") or "")
    resp = getattr(exc, "response", None)
    if resp is not None:
        url = url or str(getattr(resp, "url", "") or "")
    req = getattr(exc, "request", None)
    if req is not None:
        url = url or str(getattr(req, "url", "") or req)
    hay = f"{url} {blob} {type(exc).__name__}".lower()
    return (
        "huggingface.co" in hay
        or "hf.co" in hay
        or "hfhub" in hay
        or "huggingface" in type(exc).__name__.lower()
    )


def hf_http_status_and_detail(exc: BaseException) -> tuple[int, str] | None:
    """Map Hugging Face Hub HTTP errors to ``(FastAPI status, human detail)``.

    Local vLLM 400s (leftover weights, tool-call bench) are not HF rejections
    and must not become ``HTTP Error 400: Bad Request`` nested under 502.
    """
    code = None
    blob = str(exc) or ""
    if isinstance(exc, urllib.error.HTTPError):
        code = int(exc.code)
    else:
        raw = getattr(exc, "status_code", None)
        if raw is None:
            raw = getattr(exc, "code", None)
        if isinstance(raw, int):
            code = raw
    if code is None:
        match = _HTTP_ERROR_RE.search(blob) or _CLIENT_ERROR_RE.search(blob)
        if match:
            code = int(match.group(1))
    if code is None:
        return None
    if not _hf_origin(exc, blob):
        return None
    lower = blob.lower()
    if code in (401, 403) or "gated" in lower:
        return 400, GATED_DOWNLOAD_MSG
    if code == 404 or "entry not found" in lower or "repository not found" in lower:
        return 400, MISSING_REPO_MSG
    if 400 <= code < 500:
        return 400, HF_BAD_REQUEST_MSG
    if code >= 500:
        return 502, f"Hugging Face is unavailable (HTTP {code})"
    return None


def job_failure_detail(exc: BaseException) -> str:
    """Job / toast copy: HF 4xx as gated/missing/rejected; never urllib's Bad Request."""
    mapped = hf_http_status_and_detail(exc)
    if mapped:
        return mapped[1]
    blob = str(exc).strip() or type(exc).__name__
    match = _HTTP_ERROR_RE.search(blob)
    if match:
        code = int(match.group(1))
        if 400 <= code < 500:
            return "the local server rejected the request"
    return blob


def hf_repo_access_issue(hid: str) -> str | None:
    """Human reason if Hub will refuse *hid*. None = public, unknown, or offline."""
    repo = (hid or "").strip()
    if not repo or "/" not in repo:
        return MISSING_REPO_MSG
    try:
        info = _hf_json(
            f"{_HF}/api/models/{urllib.parse.quote(repo, safe='')}",
            timeout=_HF_CARD_TIMEOUT_S,
        )
    except urllib.error.HTTPError as exc:
        mapped = hf_http_status_and_detail(exc)
        return mapped[1] if mapped else HF_BAD_REQUEST_MSG
    except Exception:
        return None
    if isinstance(info, dict) and repo_is_gated(info):
        return GATED_DOWNLOAD_MSG
    return None


def unservable_reason(hid: str, tags: list[str] | None = None) -> str | None:
    """vLLM cannot load GGUF, EXL2, or a DSpark draft — say so instead of a lying Fits badge."""
    blob = f"{hid} {' '.join(tags or [])}".lower()
    if "gguf" in blob or "exl2" in blob or "exllamav2" in blob:
        return UNSERVABLE_FORMAT_MSG
    if "dspark" in blob or "dflash" in blob:
        return UNSERVABLE_DSPARK_MSG
    return None


def gated_repo_reason(hid: str) -> str | None:
    """HF gated / 401 / 403 — Use and Download share this copy."""
    try:
        _refuse_if_gated(hid)
    except ValueError as exc:
        return str(exc) or GATED_DOWNLOAD_MSG
    return None


def weight_bytes_from_safetensors(safetensors: dict | None, quant: str | None = None) -> int:
    """Published weight file bytes from the safetensors index — not param count as GB."""
    if not isinstance(safetensors, dict):
        return 0
    params = safetensors.get("parameters")
    total = safetensors.get("total")
    count = float(total) if isinstance(total, (int, float)) and total >= 10_000_000 else 0.0
    if isinstance(params, dict) and params:
        if not count:
            count = float(sum(v for v in params.values() if isinstance(v, (int, float))))
        # Packed AWQ must win before unique-dtype: I32 is unmapped, so
        # I32+BF16 would otherwise look like a BF16-only pack (~4× too large).
        # ``total × 0.55`` is wrong when total is the compressed index
        # (36B / 70B cyankiwi packs); price the I32+scale file bytes.
        if _awq_packed_safetensors(params):
            packed = _packed_awq_file_bytes(params)
            if packed >= _MIN_WEIGHT_BYTES:
                return packed
            if count >= 10_000_000:
                return int(count * _QUANT_BYTES["awq"])
        # MXFP4 publishes U8 + BF16. Summing U8×1 + BF16×2 is ~21 GiB for
        # gpt-oss-20b; the snapshot is ~13 GiB. Price as MXFP4 bpp.
        if count >= 10_000_000 and quant in {"mxfp4", "nvfp4"}:
            return int(count * _QUANT_BYTES[quant])
        mapped = {_SAFETENSORS_DTYPE.get(str(k).upper()) for k in params}
        mapped.discard(None)
        if len(mapped) == 1:
            kind = next(iter(mapped))
            # Unique BF16/FP16 shards are params × 2 (GLM-4-9B ~17.5 GiB),
            # not the 2.2 ID-only bpp. That extra 10% plus the Llama KV
            # floor was a lying Too big on a 24 GB card.
            if kind in {"bf16", "fp16"} and count >= 10_000_000:
                return int(count * 2)
            bpp = _QUANT_BYTES.get(kind)
            if bpp and count >= 10_000_000:
                return int(count * bpp)
        summed = sum(
            int(v) * _DTYPE_WIDTH.get(str(k).upper(), 0)
            for k, v in params.items() if isinstance(v, (int, float))
        )
        if summed >= _MIN_WEIGHT_BYTES:
            return summed
    if count >= 10_000_000 and quant in _QUANT_BYTES:
        return int(count * _QUANT_BYTES[quant])
    return 0


def resolve_weight_bytes(
    *,
    hid: str,
    tags: list[str] | None = None,
    card_data: dict | None = None,
    config: dict | None = None,
    safetensors: dict | None = None,
    disk_bytes: int = 0,
    used_storage: int = 0,
    quant: str | None = None,
) -> tuple[int, int]:
    """``(actual, listing)`` weight file bytes. VRAM is ``actual`` plus 64k KV.

    Listing prefers usedStorage when it matches the published pack, then
    safetensors file bytes, then params × quant only when the quant is on
    the repo id / config / index — a loose tag plus an ``8B`` token is not
    enough to badge Fits. Hub ``usedStorage`` and a full hub-cache walk
    can count extra revisions / ``original/`` + ``metal/`` packs
    (gpt-oss-20b lists ~38 GiB; the MXFP4 shards are ~13 GiB) — prefer
    the priced listing when those figures are far larger. Other quants
    still trust a larger on-disk pack (AWQ sideload).
    """
    tag_list = [str(t) for t in (tags or [])]
    st_bytes = weight_bytes_from_safetensors(safetensors, quant)
    used = int(used_storage or 0)
    strong_quant = (
        parse_quantization(hid, None)
        or _quant_from_config(config)
        or _quant_from_safetensors(safetensors)
    )
    params = (
        param_billions_from_safetensors(safetensors)
        or parse_param_billions(hid)
        or parse_param_billions(" ".join(tag_list))
        or parse_param_billions(_base_model_blob(card_data))
    )
    priced = 0
    if st_bytes >= _MIN_WEIGHT_BYTES:
        priced = st_bytes
    elif params is not None and strong_quant in _QUANT_BYTES:
        priced = int(params * 1_000_000_000 * _QUANT_BYTES[strong_quant])
    # Named 36B/70B AWQ must not price from a ~7B/13B packed ``total``.
    # Dense BF16/FP16 and MXFP4 already have their own shard math.
    if params is not None and strong_quant in {"awq", "gptq", "int4", "int8", "fp8"}:
        floor = int(params * 1_000_000_000 * _QUANT_BYTES[strong_quant])
        if priced < floor:
            priced = floor
    listing = 0
    if used >= _MIN_WEIGHT_BYTES:
        # gpt-oss usedStorage counts extra revisions; MXFP4 shards are
        # smaller. Other quants (AWQ sideload / compressed-tensors) trust
        # the larger listing — throwing it away was a lying Fits for 36B.
        inflated_mxfp = (
            priced >= _MIN_WEIGHT_BYTES
            and used > int(priced * 1.5)
            and strong_quant in {"mxfp4", "nvfp4"}
        )
        listing = priced if inflated_mxfp else used
    elif priced >= _MIN_WEIGHT_BYTES:
        listing = priced
    disk = int(disk_bytes or 0)
    actual = listing
    if disk >= _MIN_WEIGHT_BYTES:
        inflated_mxfp = (
            priced >= _MIN_WEIGHT_BYTES
            and disk > int(priced * 1.5)
            and strong_quant in {"mxfp4", "nvfp4"}
        )
        actual = priced if inflated_mxfp else disk
    return actual, listing


def classify_vllm_repo(
    repo: str,
    *,
    tags: list[str] | None = None,
    total_vram: int = 0,
    recommended_id: str = "",
    safetensors: dict | None = None,
    card_data: dict | None = None,
    config: dict | None = None,
    weight_bytes: int = 0,
    used_storage: int = 0,
    pipeline_tag: str = "",
) -> dict[str, Any]:
    """Fit / capability tags for a catalog row or HF search hit.

    ``fit`` is ``fits-gpu``, ``too-big``, or ``unknown``. Search and cache
    both use weight-file bytes + 64k KV (``_vram_from_weight_bytes``) —
    never raw GB on disk as VRAM, never an ``8B`` token over a larger pack,
    never a percent of the weight file as KV.
    """
    from hermes_cli.vllm_runtime.recommend import tier_for_model

    hid = (repo or "").strip()
    tag_list = [str(t) for t in (tags or [])]
    matched = tier_for_model(hid)
    quant = (
        (matched.quantization if matched and matched.quantization else None)
        or parse_quantization(hid, tag_list)
        or _quant_from_config(config)
        or _quant_from_safetensors(safetensors)
    )
    params = (
        param_billions_from_safetensors(safetensors)
        or parse_param_billions(hid)
        or parse_param_billions(" ".join(tag_list))
        or parse_param_billions(_base_model_blob(card_data))
    )
    kv_config = _resolve_kv_config(hid, config, params)
    disk = int(weight_bytes or 0)
    actual, listing = resolve_weight_bytes(
        hid=hid, tags=tag_list, card_data=card_data, config=config,
        safetensors=safetensors, disk_bytes=disk, used_storage=int(used_storage or 0),
        quant=quant,
    )
    blocked = unservable_reason(hid, tag_list)
    min_vram = 0
    fit = "unknown"
    detail = ""

    def _detail(*, priced_quant: str | None, fullprec_too_big: bool = False) -> str:
        why = _fit_why_label(hid, priced_quant, params, min_vram)
        if fullprec_too_big:
            return f"{NEEDS_AWQ_MSG} ({why})" if why else NEEDS_AWQ_MSG
        if min_vram:
            return f"Needs ~{_human_gb(min_vram)} GPU memory ({why})"
        return ""

    if blocked and matched is None:
        detail = blocked
    elif actual >= _MIN_WEIGHT_BYTES:
        min_vram = _vram_from_weight_bytes(actual, config=kv_config)
        if total_vram > 0:
            fit = _fit_on_probe(min_vram, total_vram)
        fullprec = not quant or quant in {"bf16", "fp16"}
        detail = _detail(priced_quant=quant or ("bf16" if fullprec else None),
                         fullprec_too_big=fit == "too-big" and fullprec)
        if (
            fit == "too-big"
            and disk >= _MIN_WEIGHT_BYTES
            and listing >= _MIN_WEIGHT_BYTES
            and disk > int(listing * 1.15)
        ):
            detail += (
                f" — downloaded weights are {_human_gb(disk)}, "
                f"larger than the {_human_gb(listing)} Hugging Face listing"
            )
    else:
        # No listing/disk bytes: price from the *id* only. A loose AWQ tag
        # plus an 8B token must not badge Fits (the index may be a bigger
        # BF16 pack). Named full-precision sizes that miss the 64k floor
        # become too-big + NEEDS_AWQ so Use is a 400, not a 502 wrapping HF.
        id_quant = parse_quantization(hid, None)
        tag_quant = parse_quantization("", tag_list)
        if params is not None and id_quant in _QUANT_BYTES and total_vram > 0:
            min_vram = estimate_min_vram_bytes(params, id_quant, config=kv_config)
            fit = _fit_on_probe(min_vram, total_vram)
            detail = _detail(
                priced_quant=id_quant,
                fullprec_too_big=fit == "too-big" and id_quant in {"bf16", "fp16"},
            )
        elif (
            params is not None
            and total_vram > 0
            and not id_quant
            and not tag_quant
        ):
            min_vram = estimate_min_vram_bytes(params, "bf16", config=kv_config)
            fit = _fit_on_probe(min_vram, total_vram)
            if fit == "too-big":
                detail = _detail(priced_quant="bf16", fullprec_too_big=True)
    capabilities = _capability_tags(
        hid=hid, tag_list=tag_list, quant=quant, matched=matched,
        config=config, pipeline_tag=pipeline_tag or "",
    )
    return {
        "fit": fit,
        "fits": None if fit == "unknown" else fit == "fits-gpu",
        "min_vram_bytes": min_vram,
        "quantization": quant or "",
        "capabilities": capabilities,
        "recommended": bool(recommended_id and hid == recommended_id),
        "fit_detail": detail,
        "size_bytes": actual,
    }


def cached_repo_fit(hf_id: str, *, total_vram: int = 0) -> dict[str, Any]:
    """Fit for a hub-cached id using the same weights+64k formula as search."""
    hid = (hf_id or "").strip()
    disk = 0
    for row in list_cached_repos():
        if row["id"] == hid:
            disk = int(row.get("size_bytes") or 0)
            break
    return classify_vllm_repo(hid, total_vram=total_vram, weight_bytes=disk)


def ensure_hf_weights(repo: str, job: dict | None = None) -> dict[str, Any]:
    """Pull ``repo`` into the HF hub cache when missing. Tests patch ``download_hf_repo``."""
    hid = (repo or "").strip()
    if not hid or "/" not in hid:
        raise ValueError("model must be an org/name Hugging Face id")
    blocked = unservable_reason(hid)
    if blocked:
        raise ValueError(blocked)
    if hid in cached_repo_ids():
        return {"id": hid, "already_downloaded": True}
    download_hf_repo(hid, job)
    return {"id": hid, "already_downloaded": False}


def _job_tqdm_class(job: dict):
    """tqdm stand-in so ``snapshot_download`` writes done_bytes, not just a phase.

    huggingface_hub always reads and assigns ``bar.total`` (even when it
    constructed the bar with ``total=0`` / omitted it). A missing attribute
    aborts the download.
    """

    class _JobTqdm:
        def __init__(self, *args, **kwargs):
            iterable = args[0] if args else kwargs.get("iterable")
            self._iter = iter(iterable) if iterable is not None else None
            self._unit = kwargs.get("unit")
            self.n = int(kwargs.get("initial") or 0)
            # HF's snapshot bars start at 0; per-file wrappers omit total.
            self._total = kwargs["total"] if "total" in kwargs else None
            self._publish_total(self._total)

        @property
        def total(self):
            return self._total

        @total.setter
        def total(self, value):
            self._total = value
            self._publish_total(value)

        @property
        def format_dict(self):
            return {"n": self.n, "total": self._total, "rate": None}

        def _publish_total(self, value):
            if job.get("total_bytes"):
                return
            if self._unit not in (None, "B"):
                return
            try:
                nbytes = int(value)
            except (TypeError, ValueError):
                return
            if nbytes <= 0:
                return
            # hf_thread_map uses a file-count bar (no unit, tiny total).
            if self._unit != "B" and nbytes < 256:
                return
            job["total_bytes"] = nbytes

        def __iter__(self):
            return self

        def __next__(self):
            item = next(self._iter)
            self.update(1)
            return item

        def update(self, n=1):
            try:
                step = int(n or 0)
            except (TypeError, ValueError):
                step = 0
            self.n = int(self.n or 0) + max(step, 0)
            if step <= 0:
                return
            # Several HF bars share this class; keep the largest n so a
            # file-count bar cannot clobber byte progress (or vice versa).
            job["done_bytes"] = max(int(job.get("done_bytes") or 0), self.n)
            total = job.get("total_bytes") or 0
            if total:
                job["done_bytes"] = min(int(job["done_bytes"]), int(total))
                if int(job["done_bytes"]) >= int(total) and (
                    self._unit == "B" or int(total) >= 256
                ):
                    _mark_finishing_download(job)

        def close(self):
            return None

        def clear(self):
            return None

        def refresh(self):
            return None

        def set_description(self, *args, **kwargs):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

        def set_postfix_str(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return _JobTqdm


def _prime_job_bytes(repo: str, job: dict) -> None:
    """Denominator from HF siblings so the bar is not a phase-only string."""
    try:
        info = _hf_json(
            f"{_HF}/api/models/{urllib.parse.quote(repo, safe='')}?blobs=true",
            timeout=_HF_CARD_TIMEOUT_S,
        )
    except Exception:
        return
    if not isinstance(info, dict):
        return
    siblings = [s for s in (info.get("siblings") or []) if isinstance(s, dict) and s.get("rfilename")]
    files = [
        s for s in siblings
        if Path(str(s["rfilename"])).suffix.lower() not in _SKIP_DOWNLOAD_SUFFIX
    ]
    total = sum(int(s.get("size") or 0) for s in files)
    if total:
        job["total_bytes"] = total
        job["done_bytes"] = int(job.get("done_bytes") or 0)


def _hub_snapshot_download(hid: str, job: dict | None) -> None:
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {"repo_id": hid, "cache_dir": str(hf_hub_dir())}
    if job is not None:
        kwargs["tqdm_class"] = _job_tqdm_class(job)
    try:
        snapshot_download(**kwargs)
    except TypeError:
        kwargs.pop("tqdm_class", None)
        snapshot_download(**kwargs)


def _mark_finishing_download(job: dict | None) -> None:
    """Bytes are in; snapshot_download is still hashing/moving into the hub cache."""
    if not job or job.get("phase") in {DOWNLOAD_VERIFY_PHASE, "done", "error"}:
        return
    job["phase"] = DOWNLOAD_VERIFY_PHASE
    job["detail"] = DOWNLOAD_VERIFY_DETAIL


def _is_gated_failure(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
        return True
    code = getattr(exc, "status_code", None)
    if code in (401, 403):
        return True
    blob = f"{type(exc).__name__} {exc}".lower()
    name = type(exc).__name__.lower()
    blob = str(exc).lower()
    return "gated" in name or "gated" in blob or "unauthorized" in blob or "forbidden" in blob


def _remove_fresh_cache(repo: str) -> None:
    """Drop a hub dir we just created so a failed/gated pull cannot become a library row."""
    try:
        delete_cached_repo(repo)
    except (FileNotFoundError, ValueError, OSError):
        root = hf_hub_dir() / _cache_name(repo)
        shutil.rmtree(root, ignore_errors=True)


def _refuse_if_gated(hid: str) -> None:
    try:
        info = _hf_json(
            f"{_HF}/api/models/{urllib.parse.quote(hid, safe='')}",
            timeout=_HF_CARD_TIMEOUT_S,
        )
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ValueError(GATED_DOWNLOAD_MSG) from exc
        return
    except Exception:
        return
    if isinstance(info, dict) and repo_is_gated(info):
        raise ValueError(GATED_DOWNLOAD_MSG)


def download_hf_repo(repo: str, job: dict | None = None) -> None:
    """Write ``repo`` into the hub cache. Suite always monkeypatches this."""
    hid = (repo or "").strip()
    existed = hid in cached_repo_ids()
    _refuse_if_gated(hid)
    if job is not None:
        job["phase"] = "downloading"
        job["detail"] = f"Downloading {hid}"
        _prime_job_bytes(hid, job)
    try:
        _hub_snapshot_download(hid, job)
        if job is not None and job.get("total_bytes"):
            job["done_bytes"] = job["total_bytes"]
        return
    except ImportError:
        pass
    except Exception as exc:
        if not existed:
            _remove_fresh_cache(hid)
        mapped = hf_http_status_and_detail(exc)
        if mapped and mapped[0] < 500:
            raise ValueError(mapped[1]) from exc
        if _is_gated_failure(exc):
            raise ValueError(GATED_DOWNLOAD_MSG) from exc
        raise
    try:
        _download_hf_via_api(hid, job)
    except Exception as exc:
        if not existed:
            _remove_fresh_cache(hid)
        mapped = hf_http_status_and_detail(exc)
        if mapped and mapped[0] < 500:
            raise ValueError(mapped[1]) from exc
        if _is_gated_failure(exc):
            raise ValueError(GATED_DOWNLOAD_MSG) from exc
        raise


def _download_hf_via_api(repo: str, job: dict | None = None) -> None:
    """Hub-API fallback when ``huggingface_hub`` is not installed."""
    info = _hf_json(f"{_HF}/api/models/{urllib.parse.quote(repo, safe='')}?blobs=true")
    if not isinstance(info, dict):
        raise RuntimeError(f"Hugging Face did not describe {repo}")
    siblings = [s for s in (info.get("siblings") or []) if isinstance(s, dict) and s.get("rfilename")]
    files = [
        s for s in siblings
        if Path(str(s["rfilename"])).suffix.lower() not in _SKIP_DOWNLOAD_SUFFIX
    ]
    if not files:
        raise RuntimeError(f"{repo} has no downloadable model files")
    total = sum(int(s.get("size") or 0) for s in files)
    if job is not None:
        job["total_bytes"] = total or None
        job["done_bytes"] = 0
    dest_root = hf_hub_dir() / _cache_name(repo) / "snapshots" / "main"
    dest_root.mkdir(parents=True, exist_ok=True)
    done = 0
    for s in files:
        name = str(s["rfilename"])
        dest = dest_root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            done += int(s.get("size") or dest.stat().st_size)
            if job is not None:
                job["done_bytes"] = done
            continue
        url = f"{_HF}/{repo}/resolve/main/{urllib.parse.quote(name)}"
        _http_download(url, dest, job, done)
        done += int(s.get("size") or dest.stat().st_size)
        if job is not None:
            job["done_bytes"] = done


def _http_download(url: str, dest: Path, job: dict | None, base_done: int) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-local-models"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                if job is not None:
                    job["done_bytes"] = base_done + fh.tell()
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def delete_cached_repo(repo: str) -> None:
    """Remove one HF repo's hub cache. Refuses paths outside the hub root."""
    repo = (repo or "").strip()
    if not repo or "/" not in repo:
        raise ValueError("model id must be an org/name Hugging Face id")
    root = hf_hub_dir().resolve()
    target = (root / _cache_name(repo)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("refusing to delete a path outside the Hugging Face hub cache") from exc
    if not target.is_dir():
        raise FileNotFoundError(f"{repo} is not in the local Hugging Face cache")
    shutil.rmtree(target)


def served_name_for(hf_id: str) -> str:
    from hermes_cli.vllm_runtime.recommend import tier_for_model

    hid = (hf_id or "").strip()
    matched = tier_for_model(hid)
    if matched is not None:
        return matched.served_model_name
    return hid.rsplit("/", 1)[-1] if hid else hid


def running_served_model_name() -> str:
    """Id the live server advertised after GET /v1/models 200. Empty if down.

    Supervisor state is written at spawn, before CUDA graphs finish — do not
    treat that as In use. Config ``served_model_name`` is written on Use click.
    """
    from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint
    from hermes_cli.vllm_runtime.supervisor import probe_served_model_name

    try:
        endpoint = resolve_vllm_endpoint(wait_for_boot_s=0)
        if endpoint is None:
            return ""
        return probe_served_model_name(str(endpoint.get("base_url") or ""))
    except Exception:  # noqa: BLE001 — status/catalog must not 500 on a probe
        return ""


def hide_catalog_row_by_default(row: dict[str, Any]) -> bool:
    """Official catalog rows the probe says cannot run at 64k stay in the catalog.

    User-added / cached extras stay visible even when Too big — the user
    already knows they are there. Search hits are not catalog rows.
    """
    if row.get("added_by_you"):
        return False
    return row.get("fit") == "too-big" or row.get("fits") is False


def visible_catalog_models(
    rows: list[dict[str, Any]], *, show_unfitting: bool = False,
) -> list[dict[str, Any]]:
    """Same hide set for Desktop and ``hermes local ls``."""
    if show_unfitting:
        return list(rows)
    return [r for r in rows if not r.get("hide_by_default")]


def catalog_models(config: dict | None = None, *, with_hf_meta: bool = False) -> list[dict[str, Any]]:
    """Official short list + extra cached / configured HF ids. Not six defaults.

    ``with_hf_meta`` pulls HF ``createdAt`` (first publish) for official rows —
    status polls skip this; the models list does not.
    """
    from hermes_cli.vllm_runtime.recommend import catalog_tiers, recommend_vllm
    from hermes_cli.vllm_runtime.supervisor import vllm_settings

    rec = recommend_vllm()
    cached = {r["id"]: r for r in list_cached_repos()}
    settings = vllm_settings(config)
    configured = str(settings.get("model") or "").strip()
    served = str(settings.get("served_model_name") or "").strip()
    live_served = running_served_model_name()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    recommended_id = rec.tier.model if rec.feasible and rec.tier else ""
    vram = rec.probe.total_bytes or 0
    official = [t.model for t in catalog_tiers() if t.model]
    listing = hf_listing_meta(official) if with_hf_meta else {}

    def _row(hf_id: str, *, display: str, recommended: bool, extra: dict | None = None) -> dict[str, Any]:
        hit = cached.get(hf_id) or {}
        disk = int(hit.get("size_bytes") or 0)
        meta = listing.get(hf_id) or {}
        used = int(meta.get("size_bytes") or 0)
        tags = classify_vllm_repo(
            hf_id, total_vram=vram, recommended_id=recommended_id,
            weight_bytes=disk, used_storage=used,
        )
        size = disk or used or int(tags.get("size_bytes") or 0)
        advertised = served_name_for(hf_id)
        out = {
            "id": hf_id,
            "display_name": display,
            "served_model_name": advertised,
            "recommended": recommended,
            "cached": hf_id in cached,
            "size_bytes": size,
            "size_label": ("—" if not size else _human_gb(size)),
            "active": bool(live_served and (
                hf_id == live_served or advertised == live_served
            )),
            "fits": tags["fits"],
            "fit": tags["fit"],
            "fit_detail": tags["fit_detail"],
            "min_vram_bytes": tags["min_vram_bytes"],
            "quantization": tags["quantization"],
            "capabilities": tags["capabilities"],
            "hide_by_default": False,
        }
        created = meta.get("created_at")
        if created:
            out["created_at"] = created
        if extra:
            out.update(extra)
        out["hide_by_default"] = hide_catalog_row_by_default(out)
        return out

    for tier in catalog_tiers():
        if not tier.model or tier.model in seen:
            continue
        seen.add(tier.model)
        rows.append(_row(
            tier.model,
            display=tier.served_model_name or tier.model.rsplit("/", 1)[-1],
            recommended=bool(recommended_id and tier.model == recommended_id),
            extra={"added_by_you": False},
        ))

    extras = list(cached)
    # Uncached search hits left in config (gated Gemma, 401 leftovers) are
    # not library rows — official ids already walked above.
    official_ids = set(official)
    if configured and configured not in extras and configured in official_ids:
        extras.append(configured)
    for hf_id in extras:
        if hf_id in seen:
            continue
        seen.add(hf_id)
        label = served if hf_id == configured and served else hf_id.rsplit("/", 1)[-1]
        rows.append(_row(hf_id, display=label, recommended=False, extra={"added_by_you": True}))
    return rows


def apply_vllm_model(hf_id: str) -> dict[str, Any]:
    """Persist ``local_runtime.vllm.model`` (+ served name). Does not start or stop a server."""
    from cli import save_config_value
    from hermes_cli.vllm_runtime.recommend import as_vllm_config, recommend_vllm, tier_for_model

    hid = (hf_id or "").strip()
    if not hid or "/" not in hid:
        raise ValueError("model must be an org/name Hugging Face id")
    blocked = unservable_reason(hid)
    if blocked:
        raise ValueError(blocked)
    matched = tier_for_model(hid)
    if matched is not None:
        rec = recommend_vllm()
        # Write the tier's model/parser knobs; keep the live recommend's GPU util
        # when this row is the current recommendation, else the tier's own values.
        overlay = {
            "model": matched.model,
            "served_model_name": matched.served_model_name,
            "max_model_len": matched.max_model_len,
            "gpu_memory_utilization": (
                rec.gpu_memory_utilization if rec.tier and rec.tier.model == hid
                else matched.gpu_memory_utilization
            ),
            "quantization": matched.quantization,
            "kv_cache_dtype": matched.kv_cache_dtype,
            "tool_call_parser": matched.tool_call_parser,
        }
        if rec.tier and rec.tier.model == hid:
            overlay = as_vllm_config(rec)
        for key, value in overlay.items():
            save_config_value(f"local_runtime.vllm.{key}", value)
    else:
        save_config_value("local_runtime.vllm.model", hid)
        save_config_value("local_runtime.vllm.served_model_name", served_name_for(hid))
        # Recommend writes --quantization awq. A search hit that is not AWQ
        # (Dolphin BF16, etc.) then dies with "Cannot find the config file for awq".
        parsed = parse_quantization(hid)
        save_config_value(
            "local_runtime.vllm.quantization",
            parsed if parsed in {"awq", "gptq"} else "",
        )
    return {"ok": True, "model": hid, "served_model_name": served_name_for(hid)}


def created_at_from_hf(payload: dict | None) -> str | None:
    """HF ``createdAt`` (first publish / released). Never ``lastModified``."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get("createdAt")
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    return text or None


def _used_storage_bytes(payload: dict | None) -> int:
    if not isinstance(payload, dict):
        return 0
    used = int(payload.get("usedStorage") or 0)
    if used >= _MIN_WEIGHT_BYTES:
        return used
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        return 0
    total = 0
    for sib in siblings:
        if not isinstance(sib, dict) or not sib.get("rfilename"):
            continue
        if Path(str(sib["rfilename"])).suffix.lower() in _SKIP_DOWNLOAD_SUFFIX:
            continue
        total += int(sib.get("size") or 0)
    return total if total >= _MIN_WEIGHT_BYTES else 0


def hf_listing_meta(repos: list[str]) -> dict[str, dict[str, Any]]:
    """``createdAt`` + download size for official catalog ids. Fail-soft."""
    out: dict[str, dict[str, Any]] = {}
    for repo in dict.fromkeys(r.strip() for r in repos if r and r.strip()):
        try:
            info = _hf_json(
                f"{_HF}/api/models/{urllib.parse.quote(repo, safe='')}",
                timeout=_HF_CARD_TIMEOUT_S,
            )
        except Exception:
            continue
        if not isinstance(info, dict):
            continue
        rec: dict[str, Any] = {}
        created = created_at_from_hf(info)
        if created:
            rec["created_at"] = created
        size = _used_storage_bytes(info)
        if size:
            rec["size_bytes"] = size
        if rec:
            out[repo] = rec
    return out


def _hf_json(url: str, timeout: float = _TIMEOUT_S) -> object:
    now = time.monotonic()
    hit = _HF_JSON_CACHE.get(url)
    if hit and now - hit[0] < _HF_JSON_CACHE_TTL_S:
        return hit[1]
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-local-models"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    if len(_HF_JSON_CACHE) >= _HF_JSON_CACHE_MAX:
        _HF_JSON_CACHE.pop(min(_HF_JSON_CACHE, key=lambda k: _HF_JSON_CACHE[k][0]))
    _HF_JSON_CACHE[url] = (now, data)
    return data


def search_hf_models(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Full-text HF search for weights vLLM can serve — not the GGUF firehose."""
    q = (query or "").strip()
    if not q:
        return []
    n = max(1, min(int(limit), 50))
    expand = "".join(f"&expand={name}" for name in _HF_SEARCH_EXPAND)
    url = (
        f"{_HF}/api/models?search={urllib.parse.quote(q)}"
        f"&pipeline_tag=text-generation&sort=downloads&direction=-1&limit={n}"
        f"{expand}"
    )
    raw = _hf_json(url)
    if not isinstance(raw, list):
        return []
    cached_rows = {r["id"]: r for r in list_cached_repos()}
    from hermes_cli.vllm_runtime.recommend import recommend_vllm

    rec = recommend_vllm()
    recommended_id = rec.tier.model if rec.feasible and rec.tier else ""
    vram = rec.probe.total_bytes or 0
    hits: list[dict[str, Any]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        repo = str(m.get("id") or "").strip()
        if not repo or unservable_reason(repo):
            continue
        tags = [str(t).lower() for t in (m.get("tags") or [])] if isinstance(m.get("tags"), list) else []
        card = m.get("cardData") if isinstance(m.get("cardData"), dict) else {}
        extra = card.get("tags") if isinstance(card.get("tags"), list) else []
        tags.extend(str(t).lower() for t in extra)
        if unservable_reason(repo, tags):
            continue
        safetensors = m.get("safetensors") if isinstance(m.get("safetensors"), dict) else None
        config = m.get("config") if isinstance(m.get("config"), dict) else None
        disk = int((cached_rows.get(repo) or {}).get("size_bytes") or 0)
        used = int(m.get("usedStorage") or 0)
        classified = classify_vllm_repo(
            repo,
            tags=tags,
            total_vram=vram,
            recommended_id=recommended_id,
            safetensors=safetensors,
            card_data=card,
            config=config,
            weight_bytes=disk,
            used_storage=used,
            pipeline_tag=str(m.get("pipeline_tag") or ""),
        )
        size = disk or used or int(classified.get("size_bytes") or 0)
        hit = {
            "repo": repo,
            "downloads": int(m.get("downloads") or 0),
            "likes": int(m.get("likes") or 0),
            "updated": str(m.get("lastModified") or ""),
            "gated": bool(m.get("gated")),
            "cached": repo in cached_rows,
            "fit": classified["fit"],
            "recommended": classified["recommended"],
            "capabilities": classified["capabilities"],
            "quantization": classified["quantization"],
            "fit_detail": classified["fit_detail"],
            "size_bytes": size,
            "size_label": _human_gb(size) if size else "",
        }
        created = created_at_from_hf(m)
        if created:
            hit["created_at"] = created
        hits.append(hit)
    return hits
