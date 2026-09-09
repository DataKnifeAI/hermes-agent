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
    assert parse_param_billions("someone/mystery-weights") is None


def test_parse_quantization_from_id_or_tags():
    assert parse_quantization("Qwen/Qwen3-8B-AWQ") == "awq"
    assert parse_quantization("org/model", ["fp8", "text-generation"]) == "fp8"
    assert parse_quantization("org/mystery", ["text-generation"]) is None


def test_classify_unknown_when_quant_or_size_missing():
    tags = classify_vllm_repo("someone/mystery-weights", total_vram=24 * _GIB)
    assert tags["fit"] == "unknown"
    assert tags["fits"] is None


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
