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
    "fp8": 1.1,
    "int8": 1.1,
    "bf16": 2.2,
    "fp16": 2.2,
}
_QUANT_TOKENS = (
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
# List/search ``expand`` — usedStorage is model_info-only and 400s the list API.
_HF_SEARCH_EXPAND = (
    "safetensors", "cardData", "config", "tags",
    "downloads", "likes", "lastModified", "gated",
    "pipeline_tag", "library_name",
)
_MIN_WEIGHT_BYTES = 100 << 20  # ignore tokenizer-only / empty cache dirs
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
_QUANT_CAP_ORDER = ("awq", "gptq", "fp8", "int4", "int8", "bf16", "fp16")
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


def _human_gb(n: int | float) -> str:
    return f"{n / (1 << 30):.1f} GB"


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
        if not repo:
            continue
        size = _dir_bytes(child)
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
            return quant
    return None


def param_billions_from_safetensors(safetensors: dict | None) -> float | None:
    """``safetensors.total`` / summed ``parameters`` — param count, not file bytes."""
    if not isinstance(safetensors, dict):
        return None
    total = safetensors.get("total")
    if isinstance(total, (int, float)) and total >= 10_000_000:
        return float(total) / 1_000_000_000
    params = safetensors.get("parameters")
    if isinstance(params, dict):
        summed = sum(v for v in params.values() if isinstance(v, (int, float)))
        if summed >= 10_000_000:
            return float(summed) / 1_000_000_000
    return None


def estimate_min_vram_bytes(params_b: float, quant: str) -> int:
    """Conservative 64k-floor VRAM from param count + known quant. Not a promise."""
    bpp = _QUANT_BYTES.get(quant)
    if bpp is None:
        raise ValueError(f"unknown quant {quant}")
    weight = int(params_b * 1_000_000_000 * bpp)
    return _vram_from_weight_bytes(weight)


def _vram_from_weight_bytes(weight: int) -> int:
    # KV + activations at the tool-loop floor: ~25% of weights, 2 GiB minimum.
    reserve = max(2 * (1 << 30), int(weight * 0.25))
    return weight + reserve


def _base_model_blob(card_data: dict | None) -> str:
    if not isinstance(card_data, dict):
        return ""
    raw = card_data.get("base_model")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return " ".join(str(x) for x in raw if x)
    return ""


def _quant_from_config(config: dict | None) -> str | None:
    if not isinstance(config, dict):
        return None
    qcfg = config.get("quantization_config")
    if not isinstance(qcfg, dict):
        return None
    method = str(qcfg.get("quant_method") or "").lower()
    bits = qcfg.get("bits")
    if method in {"awq", "gptq", "fp8"}:
        return method
    if bits == 4:
        return "int4"
    if bits == 8:
        return "int8"
    return None


def _quant_from_safetensors(safetensors: dict | None) -> str | None:
    """Single-dtype packs only. I32+BF16 AWQ shards are not a unique dtype."""
    if not isinstance(safetensors, dict):
        return None
    params = safetensors.get("parameters")
    if not isinstance(params, dict) or not params:
        return None
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
    if isinstance(config, dict) and isinstance(config.get("num_experts"), int) and config["num_experts"] > 1:
        moe = True
    elif any("moe" in t.lower().replace("-", "_").split("_") for t in tag_list):
        moe = True
    elif "-moe-" in blob or blob.endswith("-moe"):
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
    pipeline_tag: str = "",
) -> dict[str, Any]:
    """Fit / capability tags for a catalog row or HF search hit.

    ``fit`` is ``fits-gpu``, ``too-big``, or ``unknown``. Unknown when param
    size *and* weight bytes are missing, or quant cannot be read — never a
    green Fits badge on a guess.
    """
    from hermes_cli.vllm_runtime.recommend import TIERS

    hid = (repo or "").strip()
    tag_list = [str(t) for t in (tags or [])]
    matched = next((t for t in TIERS if t.model == hid), None)
    quant = (
        (matched.quantization if matched and matched.quantization else None)
        or parse_quantization(hid, tag_list)
        or _quant_from_config(config)
        or _quant_from_safetensors(safetensors)
    )
    params_b = (
        param_billions_from_safetensors(safetensors)
        or parse_param_billions(hid)
        or parse_param_billions(" ".join(tag_list))
        or parse_param_billions(_base_model_blob(card_data))
    )
    min_vram = 0
    fit = "unknown"
    if matched is not None:
        min_vram = matched.min_vram_bytes
        if total_vram > 0:
            fit = "fits-gpu" if total_vram >= min_vram else "too-big"
    elif params_b is not None and quant is not None:
        min_vram = estimate_min_vram_bytes(params_b, quant)
        if total_vram > 0:
            fit = "fits-gpu" if total_vram >= min_vram else "too-big"
    elif int(weight_bytes or 0) >= _MIN_WEIGHT_BYTES:
        min_vram = _vram_from_weight_bytes(int(weight_bytes))
        if total_vram > 0:
            fit = "fits-gpu" if total_vram >= min_vram else "too-big"
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
        "fit_detail": (
            f"Needs ~{_human_gb(min_vram)} GPU memory"
            if min_vram else ""
        ),
    }


def ensure_hf_weights(repo: str, job: dict | None = None) -> dict[str, Any]:
    """Pull ``repo`` into the HF hub cache when missing. Tests patch ``download_hf_repo``."""
    hid = (repo or "").strip()
    if not hid or "/" not in hid:
        raise ValueError("model must be an org/name Hugging Face id")
    if hid in cached_repo_ids():
        return {"id": hid, "already_downloaded": True}
    download_hf_repo(hid, job)
    return {"id": hid, "already_downloaded": False}


def download_hf_repo(repo: str, job: dict | None = None) -> None:
    """Write ``repo`` into the hub cache. Suite always monkeypatches this."""
    hid = (repo or "").strip()
    if job is not None:
        job["phase"] = "downloading"
        job["detail"] = f"Downloading {hid}"
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        snapshot_download = None
    if snapshot_download is not None:
        snapshot_download(repo_id=hid, cache_dir=str(hf_hub_dir()))
        return
    _download_hf_via_api(hid, job)


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
    from hermes_cli.vllm_runtime.recommend import TIERS

    hid = (hf_id or "").strip()
    for tier in TIERS:
        if tier.model == hid:
            return tier.served_model_name
    return hid.rsplit("/", 1)[-1] if hid else hid


def catalog_models(config: dict | None = None) -> list[dict[str, Any]]:
    """Curated VRAM-tier rows + extra cached / configured HF ids."""
    from hermes_cli.vllm_runtime.recommend import TIERS, recommend_vllm
    from hermes_cli.vllm_runtime.supervisor import vllm_settings

    rec = recommend_vllm()
    cached = {r["id"]: r for r in list_cached_repos()}
    settings = vllm_settings(config)
    configured = str(settings.get("model") or "").strip()
    served = str(settings.get("served_model_name") or "").strip()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    recommended_id = rec.tier.model if rec.tier else ""
    vram = rec.probe.total_bytes or 0

    def _row(hf_id: str, *, display: str, recommended: bool, extra: dict | None = None) -> dict[str, Any]:
        hit = cached.get(hf_id) or {}
        size = int(hit.get("size_bytes") or 0)
        tags = classify_vllm_repo(
            hf_id, total_vram=vram, recommended_id=recommended_id,
            weight_bytes=size,
        )
        out = {
            "id": hf_id,
            "display_name": display,
            "served_model_name": served_name_for(hf_id),
            "recommended": recommended,
            "cached": hf_id in cached,
            "size_bytes": size,
            "size_label": hit.get("size_label") or ("—" if not size else _human_gb(size)),
            "active": bool(configured and hf_id == configured),
            "fits": tags["fits"],
            "fit": tags["fit"],
            "fit_detail": tags["fit_detail"],
            "min_vram_bytes": tags["min_vram_bytes"],
            "quantization": tags["quantization"],
            "capabilities": tags["capabilities"],
        }
        if extra:
            out.update(extra)
        return out

    for tier in TIERS:
        if tier.model in seen:
            continue
        seen.add(tier.model)
        rows.append(_row(
            tier.model,
            display=tier.served_model_name or tier.model.rsplit("/", 1)[-1],
            recommended=bool(rec.tier and rec.tier.model == tier.model),
            extra={"added_by_you": False},
        ))

    extras = list(cached)
    if configured and configured not in extras:
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
    from hermes_cli.vllm_runtime.recommend import TIERS, as_vllm_config, recommend_vllm

    hid = (hf_id or "").strip()
    if not hid or "/" not in hid:
        raise ValueError("model must be an org/name Hugging Face id")
    matched = next((t for t in TIERS if t.model == hid), None)
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
    return {"ok": True, "model": hid, "served_model_name": served_name_for(hid)}


def _hf_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "hermes-local-models"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
        return json.load(r)


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
    cached = cached_repo_ids()
    from hermes_cli.vllm_runtime.recommend import recommend_vllm

    rec = recommend_vllm()
    recommended_id = rec.tier.model if rec.tier else ""
    vram = rec.probe.total_bytes or 0
    hits: list[dict[str, Any]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        repo = str(m.get("id") or "").strip()
        if not repo or "gguf" in repo.lower():
            continue
        tags = [str(t).lower() for t in (m.get("tags") or [])] if isinstance(m.get("tags"), list) else []
        card = m.get("cardData") if isinstance(m.get("cardData"), dict) else {}
        extra = card.get("tags") if isinstance(card.get("tags"), list) else []
        tags.extend(str(t).lower() for t in extra)
        if any("gguf" in t for t in tags):
            continue
        safetensors = m.get("safetensors") if isinstance(m.get("safetensors"), dict) else None
        config = m.get("config") if isinstance(m.get("config"), dict) else None
        classified = classify_vllm_repo(
            repo,
            tags=tags,
            total_vram=vram,
            recommended_id=recommended_id,
            safetensors=safetensors,
            card_data=card,
            config=config,
            weight_bytes=int(m.get("usedStorage") or 0),
            pipeline_tag=str(m.get("pipeline_tag") or ""),
        )
        hits.append({
            "repo": repo,
            "downloads": int(m.get("downloads") or 0),
            "likes": int(m.get("likes") or 0),
            "updated": str(m.get("lastModified") or ""),
            "gated": bool(m.get("gated")),
            "cached": repo in cached,
            "fit": classified["fit"],
            "recommended": classified["recommended"],
            "capabilities": classified["capabilities"],
            "quantization": classified["quantization"],
            "fit_detail": classified["fit_detail"],
        })
    return hits
