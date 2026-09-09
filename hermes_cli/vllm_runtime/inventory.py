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
    """Last ``Nb`` / ``N.NB`` token in an HF id — ``Qwen3.8-27B`` → 27, not 3.8."""
    hits = _PARAM_RE.findall(text or "")
    if not hits:
        return None
    try:
        return float(hits[-1])
    except ValueError:
        return None


def parse_quantization(repo: str, tags: list[str] | None = None) -> str | None:
    blob = f"{repo} {' '.join(tags or [])}".lower()
    for name in ("awq", "gptq", "fp8", "int4", "int8", "bf16", "fp16"):
        if name in blob:
            return name
    return None


def estimate_min_vram_bytes(params_b: float, quant: str) -> int:
    """Conservative 64k-floor VRAM from param count + known quant. Not a promise."""
    bpp = _QUANT_BYTES.get(quant)
    if bpp is None:
        raise ValueError(f"unknown quant {quant}")
    weight = int(params_b * 1_000_000_000 * bpp)
    # KV + activations at the tool-loop floor: ~25% of weights, 2 GiB minimum.
    reserve = max(2 * (1 << 30), int(weight * 0.25))
    return weight + reserve


def classify_vllm_repo(
    repo: str,
    *,
    tags: list[str] | None = None,
    total_vram: int = 0,
    recommended_id: str = "",
) -> dict[str, Any]:
    """Fit / capability tags for a catalog row or HF search hit.

    ``fit`` is ``fits-gpu``, ``too-big``, or ``unknown``. Unknown when param
    size or quant cannot be read — never a green Fits badge on a guess.
    """
    from hermes_cli.vllm_runtime.recommend import TIERS

    hid = (repo or "").strip()
    tag_list = [str(t) for t in (tags or [])]
    matched = next((t for t in TIERS if t.model == hid), None)
    quant = (matched.quantization if matched and matched.quantization else None) or parse_quantization(hid, tag_list)
    params_b = parse_param_billions(hid)
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
    blob = f"{hid} {' '.join(tag_list)}".lower()
    capabilities: list[str] = []
    if quant in {"awq", "gptq", "fp8"}:
        capabilities.append(quant)
    if "instruct" in blob or "-chat" in blob or "chat-" in blob:
        capabilities.append("instruct")
    if matched is not None or "hermes" in blob or "tool" in blob:
        if "instruct" not in capabilities and matched is not None:
            capabilities.append("instruct")
        if "tools" not in capabilities and (matched is not None or "hermes" in blob):
            capabilities.append("tools")
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
    url = (
        f"{_HF}/api/models?search={urllib.parse.quote(q)}"
        f"&pipeline_tag=text-generation&sort=downloads&direction=-1&limit={n}"
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
        if any("gguf" in t for t in tags):
            continue
        classified = classify_vllm_repo(
            repo, tags=tags, total_vram=vram, recommended_id=recommended_id,
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
