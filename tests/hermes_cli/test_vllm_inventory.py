"""vLLM inventory contracts: cache short-circuit, honest fit tags, no live HF."""

from __future__ import annotations

from hermes_cli.vllm_runtime.inventory import (
    _vram_from_weight_bytes,
    classify_vllm_repo,
    created_at_from_hf,
    download_hf_repo,
    ensure_hf_weights,
    estimate_min_vram_bytes,
    parse_param_billions,
    parse_quantization,
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
    assert tags["fit"] == "fits-gpu"
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
    assert named["fit"] == "unknown"  # 8B in id, no quant
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
        bar = __import__(
            "hermes_cli.vllm_runtime.inventory", fromlist=["_job_tqdm_class"]
        )._job_tqdm_class(job)(total=1000)
        bar.update(400)
        bar.close()

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hub_snapshot_download", _snap)
    job = {"phase": "", "detail": "", "done_bytes": 0, "total_bytes": None}
    download_hf_repo("org/weights", job)
    assert job["total_bytes"] == 1000
    assert job["done_bytes"] == 1000
    assert job["phase"] == "downloading"


def test_estimate_min_vram_matches_weight_plus_kv():
    assert estimate_min_vram_bytes(8, "awq") == _vram_from_weight_bytes(
        int(8 * 1_000_000_000 * 0.55))
    actual, listing = resolve_weight_bytes(hid="org/Qwen3-8B-AWQ", used_storage=5 * _GIB)
    assert actual == listing == 5 * _GIB
