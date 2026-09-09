"""vLLM inventory contracts: cache short-circuit, honest fit tags, no live HF."""

from __future__ import annotations

from hermes_cli.vllm_runtime.inventory import (
    classify_vllm_repo,
    ensure_hf_weights,
    parse_param_billions,
    parse_quantization,
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
