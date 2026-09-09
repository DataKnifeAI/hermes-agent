"""HF-id inventory for the managed vLLM engine.

Cached Hugging Face weights (not GGUF) are the library. Search is the HF
text-generation firehose. Delete removes only that repo's hub cache dir.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HF = "https://huggingface.co"
_TIMEOUT_S = 15


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

    def _row(hf_id: str, *, display: str, recommended: bool, extra: dict | None = None) -> dict[str, Any]:
        hit = cached.get(hf_id) or {}
        size = int(hit.get("size_bytes") or 0)
        out = {
            "id": hf_id,
            "display_name": display,
            "served_model_name": served_name_for(hf_id),
            "recommended": recommended,
            "cached": hf_id in cached,
            "size_bytes": size,
            "size_label": hit.get("size_label") or ("—" if not size else _human_gb(size)),
            "active": bool(configured and hf_id == configured),
        }
        if extra:
            out.update(extra)
        return out

    for tier in TIERS:
        if tier.model in seen:
            continue
        seen.add(tier.model)
        fits = rec.probe.total_bytes >= tier.min_vram_bytes if rec.probe.total_bytes else None
        rows.append(_row(
            tier.model,
            display=tier.served_model_name or tier.model.rsplit("/", 1)[-1],
            recommended=bool(rec.tier and rec.tier.model == tier.model),
            extra={
                "min_vram_bytes": tier.min_vram_bytes,
                "quantization": tier.quantization,
                "fits": fits,
                "added_by_you": False,
            },
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
    url = (
        f"{_HF}/api/models?search={urllib.parse.quote(q)}"
        f"&pipeline_tag=text-generation&sort=downloads&direction=-1&limit={n}"
    )
    raw = _hf_json(url)
    if not isinstance(raw, list):
        return []
    cached = cached_repo_ids()
    hits: list[dict[str, Any]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        repo = str(m.get("id") or "").strip()
        if not repo or "gguf" in repo.lower():
            continue
        tags = [str(t).lower() for t in (m.get("tags") or [])] if isinstance(m.get("tags"), list) else []
        if any("gguf" in t for t in tags):
            continue
        hits.append({
            "repo": repo,
            "downloads": int(m.get("downloads") or 0),
            "likes": int(m.get("likes") or 0),
            "updated": str(m.get("lastModified") or ""),
            "gated": bool(m.get("gated")),
            "cached": repo in cached,
        })
    return hits
