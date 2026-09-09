"""vLLM inventory contracts: cache short-circuit, honest fit tags, no live HF."""

from __future__ import annotations

import urllib.error
from io import BytesIO

from hermes_cli.vllm_runtime.inventory import (
    DOWNLOAD_VERIFY_DETAIL,
    DOWNLOAD_VERIFY_PHASE,
    GATED_DOWNLOAD_MSG,
    HF_BAD_REQUEST_MSG,
    NEEDS_AWQ_MSG,
    hf_http_status_and_detail,
    _cache_name,
    _job_tqdm_class,
    _vram_from_weight_bytes,
    classify_vllm_repo,
    created_at_from_hf,
    download_hf_repo,
    ensure_hf_weights,
    estimate_min_vram_bytes,
    hf_hub_dir,
    list_cached_repos,
    parse_param_billions,
    parse_quantization,
    repo_is_cached,
    resolve_weight_bytes,
)


_GIB = 1 << 30


def test_parse_param_billions_takes_last_size_token():
    assert parse_param_billions("Qwen/Qwen3.8-27B-FP8") == 27
    assert parse_param_billions("meta-llama/Llama-3.1-8B-Instruct") == 8
    assert parse_param_billions("Qwen/Qwen3-Coder-30B-A3B-Instruct") == 30
    assert parse_param_billions("someone/mystery-weights") is None


def test_parse_quantization_from_id_or_tags():
    assert parse_quantization("Qwen/Qwen3-8B-AWQ") == "awq"
    assert parse_quantization("org/model", ["fp8", "text-generation"]) == "fp8"
    assert parse_quantization("org/mystery", ["4-bit", "text-generation"]) == "int4"
    assert parse_quantization("org/mystery", ["text-generation"]) is None


def test_classify_unknown_when_quant_or_size_missing():
    tags = classify_vllm_repo("someone/mystery-weights", total_vram=24 * _GIB)
    assert tags["fit"] == "unknown"
    assert tags["fits"] is None
    assert tags["capabilities"] == []


def test_classify_nous_8b_bf16_is_too_big_needs_awq():
    tags = classify_vllm_repo("NousResearch/Hermes-3-Llama-3.1-8B", total_vram=24 * _GIB)
    assert tags["fit"] == "too-big"
    assert tags["fit_detail"] == NEEDS_AWQ_MSG


def test_classify_nous_awq_8b_fits_24gb():
    tags = classify_vllm_repo(
        "solidrust/Hermes-3-Llama-3.1-8B-AWQ", total_vram=24 * _GIB)
    assert tags["fit"] == "fits-gpu"


def test_hf_http_400_is_plain_language_not_urllib_string():
    err = urllib.error.HTTPError(
        "https://huggingface.co/api/models/x", 400, "Bad Request",
        hdrs=None, fp=BytesIO())
    mapped = hf_http_status_and_detail(err)
    assert mapped is not None
    status, detail = mapped
    assert status == 400
    assert detail == HF_BAD_REQUEST_MSG
    assert "Bad Request" not in detail
    wrapped = hf_http_status_and_detail(
        RuntimeError("HTTP Error 400: Bad Request for url: https://huggingface.co/api/models/x"))
    assert wrapped == (400, HF_BAD_REQUEST_MSG)


def test_local_vllm_400_is_not_hf_rejection():
    """Leftover-server tool-call 400 must not toast as Hugging Face rejected."""
    from hermes_cli.vllm_runtime.inventory import job_failure_detail

    local = urllib.error.HTTPError(
        "http://127.0.0.1:40689/v1/chat/completions", 400, "Bad Request",
        hdrs=None, fp=BytesIO())
    assert hf_http_status_and_detail(local) is None
    assert hf_http_status_and_detail(RuntimeError("HTTP Error 400: Bad Request")) is None
    detail = job_failure_detail(RuntimeError("HTTP Error 400: Bad Request"))
    assert "Hugging Face" not in detail
    assert "Bad Request" not in detail


def test_classify_uses_tier_floor_not_a_guess():
    # Curated 32B row is the 40 GB tier — 24 GB must read too-big, not Fits.
    tags = classify_vllm_repo("Qwen/Qwen3-32B-AWQ", total_vram=24 * _GIB)
    assert tags["fit"] == "too-big"
    assert tags["fits"] is False
    small = classify_vllm_repo("Qwen/Qwen3-8B-AWQ", total_vram=24 * _GIB,
                               recommended_id="Qwen/Qwen3-14B-AWQ")
    assert small["fit"] == "fits-gpu"
    assert small["recommended"] is False
    assert "awq" in small["capabilities"]


def test_classify_8b_awq_fits_24gb_from_safetensors_not_id():
    """Repo name has no Nb token; HF safetensors.total + AWQ tag is enough."""
    tags = classify_vllm_repo(
        "org/finetune-awq",
        tags=["awq", "4-bit", "instruct"],
        total_vram=24 * _GIB,
        safetensors={"parameters": {"I32": 7_000_000_000, "BF16": 1_000_000_000},
                     "total": 8_000_000_000},
    )
    assert tags["fit"] == "fits-gpu"
    assert tags["fits"] is True
    assert "awq" in tags["capabilities"]
    assert "instruct" in tags["capabilities"]


def test_classify_bf16_from_safetensors_dtype():
    tags = classify_vllm_repo(
        "org/plain-instruct",
        tags=["conversational"],
        total_vram=24 * _GIB,
        safetensors={"parameters": {"BF16": 8_000_000_000}, "total": 8_000_000_000},
    )
    assert tags["fit"] == "too-big"
    assert tags["fits"] is False
    assert tags["quantization"] == "bf16"
    assert "bf16" in tags["capabilities"]
    assert "instruct" in tags["capabilities"]


def test_classify_unknown_without_size_even_with_quant_tag():
    tags = classify_vllm_repo(
        "org/mystery-awq-build",
        tags=["awq"],
        total_vram=24 * _GIB,
    )
    assert tags["fit"] == "unknown"
    assert tags["fits"] is None


def test_classify_weight_bytes_when_card_has_no_params():
    tags = classify_vllm_repo(
        "org/sideload",
        total_vram=24 * _GIB,
        weight_bytes=5 * _GIB,
    )
    assert tags["fit"] == "fits-gpu"
    tiny = classify_vllm_repo("org/sideload", total_vram=24 * _GIB, weight_bytes=64)
    assert tiny["fit"] == "unknown"


def test_classify_capabilities_from_hf_tags_not_model_names():
    named = classify_vllm_repo("NousResearch/Hermes-3-Llama-3.1-8B", total_vram=24 * _GIB)
    assert named["fit"] == "too-big"
    assert named["fit_detail"] == NEEDS_AWQ_MSG
    assert "tools" not in named["capabilities"]
    tagged = classify_vllm_repo(
        "acme/custom-weights",
        tags=["vision", "qwen3_moe", "code", "128k", "function-calling", "instruct", "awq"],
        total_vram=24 * _GIB,
        safetensors={"total": 8_000_000_000},
    )
    assert tagged["fit"] == "fits-gpu"
    caps = tagged["capabilities"]
    assert "awq" in caps
    assert "instruct" in caps
    assert "tools" in caps
    assert "vision" in caps
    assert "moe" in caps
    # Density cap — not every tag becomes a badge.
    assert len(caps) <= 5
    assert "coding" not in caps or "128k" not in caps or len(caps) <= 5


def test_classify_context_and_coding_from_tags():
    tags = classify_vllm_repo(
        "org/coder-32k",
        tags=["coder", "32k", "awq"],
        total_vram=24 * _GIB,
        safetensors={"total": 8_000_000_000},
    )
    assert "coding" in tags["capabilities"]
    assert "32k" in tags["capabilities"]


def test_ensure_hf_weights_skips_cached_and_calls_download(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    cached = hub / "models--acme--already"
    cached.mkdir(parents=True)
    (cached / "w.bin").write_bytes(b"x")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: pulled.append(repo))

    skip = ensure_hf_weights("acme/already")
    assert skip["already_downloaded"] is True
    assert pulled == []

    missing = ensure_hf_weights("acme/fresh")
    assert missing["already_downloaded"] is False
    assert pulled == ["acme/fresh"]


def test_created_at_from_hf_is_created_at_not_last_modified():
    assert created_at_from_hf({
        "createdAt": "2025-03-15T00:00:00.000Z",
        "lastModified": "2026-08-01T00:00:00.000Z",
    }) == "2025-03-15T00:00:00.000Z"
    assert created_at_from_hf({"lastModified": "2026-08-01T00:00:00.000Z"}) is None
    assert created_at_from_hf({"createdAt": ""}) is None
    assert created_at_from_hf(None) is None


def test_search_payload_created_at_and_size(monkeypatch):
    from hermes_cli.vllm_runtime.inventory import search_hf_models
    from hermes_cli.vllm_runtime.recommend import NvidiaProbe, TIERS, VllmRecommendation

    tier = next(t for t in TIERS if t.id == "24gb")
    rec = VllmRecommendation(NvidiaProbe(24 * _GIB, 24 * _GIB, "data"), tier, True, "ok")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory.list_cached_repos", lambda: [])
    seen: list[str] = []

    def _fake(url, timeout=15):
        seen.append(url)
        return [
            {"id": "Qwen/Qwen3-8B-AWQ", "downloads": 9, "likes": 2,
             "lastModified": "2026-08-01T00:00:00.000Z",
             "createdAt": "2025-04-29T00:00:00.000Z",
             "gated": False, "tags": ["awq", "instruct"],
             "usedStorage": 5 * _GIB},
            {"id": "someone/mystery-weights", "downloads": 1, "likes": 0,
             "lastModified": "2026-01-01", "gated": False, "tags": []},
        ]

    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory._hf_json", _fake)
    hits = search_hf_models("qwen")
    assert seen and "expand=createdAt" in seen[0]
    by_repo = {h["repo"]: h for h in hits}
    assert by_repo["Qwen/Qwen3-8B-AWQ"]["created_at"] == "2025-04-29T00:00:00.000Z"
    assert by_repo["Qwen/Qwen3-8B-AWQ"]["size_bytes"] == 5 * _GIB
    assert "GB" in by_repo["Qwen/Qwen3-8B-AWQ"]["size_label"]
    assert "created_at" not in by_repo["someone/mystery-weights"]


def test_catalog_payload_created_at_and_size(monkeypatch):
    from hermes_cli.vllm_runtime.inventory import catalog_models
    from hermes_cli.vllm_runtime.recommend import (
        NvidiaProbe, VllmRecommendation, catalog_tiers,
    )

    official = catalog_tiers()
    pick = next(t for t in official if t.min_vram_bytes <= 24 * _GIB)
    rec = VllmRecommendation(NvidiaProbe(24 * _GIB, 24 * _GIB, "data"), pick, True, "ok")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory.list_cached_repos", lambda: [])
    monkeypatch.setattr("hermes_cli.vllm_runtime.supervisor.vllm_settings", lambda cfg=None: {})

    def _fake(url, timeout=15):
        if "Qwen3-8B-AWQ" in url:
            return {"id": "Qwen/Qwen3-8B-AWQ",
                    "createdAt": "2025-04-29T00:00:00.000Z",
                    "lastModified": "2026-08-01T00:00:00.000Z",
                    "usedStorage": 5 * _GIB}
        return {"id": "other", "lastModified": "2026-01-01"}

    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory._hf_json", _fake)
    rows = catalog_models({}, with_hf_meta=True)
    eight = next(r for r in rows if r["id"] == "Qwen/Qwen3-8B-AWQ")
    assert eight["created_at"] == "2025-04-29T00:00:00.000Z"
    assert eight["size_bytes"] == 5 * _GIB
    assert "GB" in eight["size_label"]
    missing = [r for r in rows if r["id"] != "Qwen/Qwen3-8B-AWQ"]
    assert missing and all("created_at" not in r for r in missing)


def test_catalog_omits_uncached_configured_search_hit(monkeypatch):
    """Gated Gemma left in config.yaml is not a hollow library row."""
    from hermes_cli.vllm_runtime.inventory import catalog_models
    from hermes_cli.vllm_runtime.recommend import (
        NvidiaProbe, VllmRecommendation, catalog_tiers,
    )

    pick = next(t for t in catalog_tiers() if t.id == "24gb")
    rec = VllmRecommendation(NvidiaProbe(24 * _GIB, 24 * _GIB, "data"), pick, True, "ok")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory.list_cached_repos", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.vllm_settings",
        lambda cfg=None: {"model": "google/gemma-3-27b-it", "served_model_name": "gemma"},
    )
    rows = catalog_models({})
    ids = {r["id"] for r in rows}
    assert "google/gemma-3-27b-it" not in ids
    assert "Qwen/Qwen3-8B-AWQ" in ids
    assert all(r["id"].count("/") == 1 for r in rows)


def test_fit_same_formula_search_and_cached():
    """Search listing bytes and the same bytes on disk must agree."""
    hid = "org/dolphin-8b-awq"
    listing = classify_vllm_repo(
        hid, tags=["awq", "instruct"], total_vram=24 * _GIB, used_storage=5 * _GIB)
    cached = classify_vllm_repo(
        hid, tags=["awq", "instruct"], total_vram=24 * _GIB, weight_bytes=5 * _GIB)
    assert listing["fit"] == cached["fit"] == "fits-gpu"
    assert listing["min_vram_bytes"] == cached["min_vram_bytes"]
    assert listing["min_vram_bytes"] == _vram_from_weight_bytes(5 * _GIB)
    assert listing["min_vram_bytes"] != 5 * _GIB  # disk GB is not VRAM


def test_fit_does_not_trust_8b_token_over_larger_pack():
    """An 8B token must not badge Fits when the index is a bigger BF16 pack."""
    hid = "org/Dolphin-Llama-8B"
    from_name = classify_vllm_repo(hid, tags=["awq"], total_vram=24 * _GIB)
    assert from_name["fit"] == "unknown"
    big = classify_vllm_repo(
        hid, tags=["awq"], total_vram=24 * _GIB,
        safetensors={"parameters": {"BF16": 70_000_000_000}, "total": 70_000_000_000},
    )
    assert big["fit"] == "too-big"
    assert big["min_vram_bytes"] == _vram_from_weight_bytes(
        int(70_000_000_000 * 2.2))


def test_fit_flip_only_when_cache_is_larger():
    search = classify_vllm_repo(
        "org/dolphin-8b-awq", tags=["instruct"], total_vram=24 * _GIB,
        used_storage=5 * _GIB)
    cached = classify_vllm_repo(
        "org/dolphin-8b-awq", tags=["instruct"], total_vram=24 * _GIB,
        weight_bytes=40 * _GIB, used_storage=5 * _GIB)
    assert search["fit"] == "fits-gpu"
    assert cached["fit"] == "too-big"
    assert "larger" in cached["fit_detail"]
    same = classify_vllm_repo(
        "org/dolphin-8b-awq", tags=["instruct"], total_vram=24 * _GIB,
        weight_bytes=5 * _GIB, used_storage=5 * _GIB)
    assert same["fit"] == search["fit"]


def test_exl2_is_not_fits():
    tags = classify_vllm_repo(
        "org/Dolphin-8B-exl2", tags=["exl2"], total_vram=24 * _GIB,
        used_storage=5 * _GIB)
    assert tags["fit"] == "unknown"
    assert "cannot serve" in (tags["fit_detail"] or "").lower()
    from hermes_cli.vllm_runtime.inventory import UNSERVABLE_FORMAT_MSG, unservable_reason

    assert unservable_reason("org/Dolphin-8B-exl2") == UNSERVABLE_FORMAT_MSG
    assert unservable_reason("someone/Qwen-GGUF") == UNSERVABLE_FORMAT_MSG
    assert unservable_reason("dphn/dolphin-2.9.1-llama-3-8b") is None


def test_download_job_reports_bytes(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda url, timeout=15: {"siblings": [
            {"rfilename": "model.safetensors", "size": 1000},
            {"rfilename": "README.md", "size": 12},
        ]},
    )

    def _snap(hid, job):
        bar = _job_tqdm_class(job)(total=1000)
        bar.update(400)
        bar.close()

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hub_snapshot_download", _snap)
    job = {"phase": "", "detail": "", "done_bytes": 0, "total_bytes": None}
    download_hf_repo("org/weights", job)
    assert job["total_bytes"] == 1000
    assert job["done_bytes"] == 1000
    assert job["phase"] == "downloading"


def test_download_completes_when_hub_tqdm_omits_total(monkeypatch, tmp_path):
    """snapshot_download reads bar.total; HF often constructs the bar without it."""
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda url, timeout=15: {"siblings": [{"rfilename": "model.safetensors"}]},
    )

    def _snap(hid, job):
        # HF may construct the bar with no total kwarg, then still read .total.
        bare = _job_tqdm_class(job)()
        bar = _job_tqdm_class(job)(total=0, unit="B")
        bar.total = (bare.total or 0) + (bar.total or 0)
        bar.update(400)
        bar.close()

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hub_snapshot_download", _snap)
    job = {"phase": "", "detail": "", "done_bytes": 0, "total_bytes": None}
    download_hf_repo("org/weights", job)
    assert job["done_bytes"] == 400
    assert job["total_bytes"] is None


def test_estimate_min_vram_matches_weight_plus_kv():
    assert estimate_min_vram_bytes(8, "awq") == _vram_from_weight_bytes(
        int(8 * 1_000_000_000 * 0.55))
    actual, listing = resolve_weight_bytes(hid="org/Qwen3-8B-AWQ", used_storage=5 * _GIB)
    assert actual == listing == 5 * _GIB


def test_nous_full_precision_too_big_on_24gb():
    """Search listing and the same bytes on disk must both refuse 8B BF16 on 24 GB."""
    hid = "NousResearch/Hermes-3-Llama-3.1-8B"
    st = {"parameters": {"BF16": 8_000_000_000}, "total": 8_000_000_000}
    search = classify_vllm_repo(hid, total_vram=24 * _GIB, safetensors=st)
    listed = classify_vllm_repo(hid, total_vram=24 * _GIB, used_storage=16 * _GIB)
    cached = classify_vllm_repo(hid, total_vram=24 * _GIB, weight_bytes=16 * _GIB)
    assert search["fit"] == listed["fit"] == cached["fit"] == "too-big"
    assert search["fits"] is False
    assert listed["min_vram_bytes"] == cached["min_vram_bytes"]
    assert listed["min_vram_bytes"] == _vram_from_weight_bytes(16 * _GIB)


def test_large_awq_too_big_on_24gb_without_catalog_id():
    tags = classify_vllm_repo(
        "org/Hermes-32B-AWQ", tags=["awq"], total_vram=24 * _GIB,
        used_storage=18 * _GIB)
    assert tags["fit"] == "too-big"
    assert tags["fits"] is False


def test_empty_or_tokenizer_cache_is_not_a_library_row(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    hollow = hub / "models--google--gemma-3-27b-it"
    hollow.mkdir(parents=True)
    (hollow / "README.md").write_text("gated leftover", encoding="utf-8")
    (hollow / "config.json").write_text("{}", encoding="utf-8")
    (hollow / "tokenizer.json").write_bytes(b"x" * 2048)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    assert list_cached_repos() == []
    assert repo_is_cached("google/gemma-3-27b-it") is False
    tags = classify_vllm_repo(
        "google/gemma-3-27b-it", total_vram=24 * _GIB, weight_bytes=0)
    # Hollow cache is not a library row (above). Search/Use of the id still
    # prices 27B BF16 as too-big so the toast is needs-AWQ, not a nested 400.
    assert tags["fit"] == "too-big"
    assert tags["fit_detail"] == NEEDS_AWQ_MSG
    assert tags["fits"] is False


def test_gated_download_refuses_without_cache_dir(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda url, timeout=15: {"id": "google/gemma-3-27b-it", "gated": True},
    )
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hub_snapshot_download",
        lambda hid, job=None: pulled.append(hid),
    )
    job = {"phase": "", "detail": "", "done_bytes": 0, "total_bytes": None}
    try:
        download_hf_repo("google/gemma-3-27b-it", job)
        raise AssertionError("gated download must refuse")
    except ValueError as exc:
        assert "gated" in str(exc).lower()
        assert str(exc) == GATED_DOWNLOAD_MSG
    assert pulled == []
    assert not (hf_hub_dir() / _cache_name("google/gemma-3-27b-it")).exists()
    assert list_cached_repos() == []


def test_download_hf_400_is_plain_language(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda url, timeout=15: {"id": "org/missing", "gated": False},
    )

    def _snap(hid, job=None):
        raise urllib.error.HTTPError(
            "https://huggingface.co/org/missing", 400, "Bad Request",
            hdrs=None, fp=BytesIO())

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hub_snapshot_download", _snap)
    try:
        download_hf_repo("org/missing")
        raise AssertionError("400 must refuse")
    except ValueError as exc:
        assert str(exc) == HF_BAD_REQUEST_MSG
        assert "Bad Request" not in str(exc)


def test_gated_http_error_scrubs_incomplete_cache(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda url, timeout=15: {"id": "google/gemma-3-27b-it", "gated": False},
    )

    def _snap(hid, job):
        dest = hf_hub_dir() / _cache_name(hid)
        dest.mkdir(parents=True)
        (dest / "README.md").write_text("no weights", encoding="utf-8")
        raise urllib.error.HTTPError(
            "https://huggingface.co/google/gemma-3-27b-it", 403, "Forbidden",
            hdrs={}, fp=BytesIO())

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hub_snapshot_download", _snap)
    try:
        download_hf_repo("google/gemma-3-27b-it")
        raise AssertionError("403 must refuse")
    except ValueError as exc:
        assert "gated" in str(exc).lower()
    assert not (hf_hub_dir() / _cache_name("google/gemma-3-27b-it")).exists()
    assert list_cached_repos() == []


def test_download_job_verifying_after_bytes_complete(monkeypatch, tmp_path):
    """100% bytes is not done — snapshot_download still hashes/moves into the hub."""
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda url, timeout=15: {"siblings": [
            {"rfilename": "model.safetensors", "size": 1000},
        ]},
    )
    seen: list[str] = []

    def _snap(hid, job):
        bar = _job_tqdm_class(job)(total=1000, unit="B")
        bar.update(1000)
        seen.append(job["phase"])
        assert job["detail"] == DOWNLOAD_VERIFY_DETAIL
        bar.close()

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hub_snapshot_download", _snap)
    job = {"phase": "", "detail": "", "done_bytes": 0, "total_bytes": None}
    download_hf_repo("org/weights", job)
    assert seen == [DOWNLOAD_VERIFY_PHASE]
    assert job["phase"] == DOWNLOAD_VERIFY_PHASE
    assert job["done_bytes"] == 1000
    assert "install" not in (job.get("detail") or "").lower()
