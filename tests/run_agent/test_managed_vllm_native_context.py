"""Managed GPU 14B may chat at native 40960; compression/aux still needs 64k.

Qwen3-14B-AWQ max_position_embeddings is 40960. Serving 64k CUDA-OOBs.
Agent init must not reject that honest window, must not fake context_length,
and must not switch the selected 14B to CPU 4B. Aux below 64k still fails.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


GPU_14B_URL = "http://127.0.0.1:18435/v1"
CPU_4B_URL = "http://127.0.0.1:18436/v1"
NATIVE_14B = 40960


def _gpu_14b_cfg(**extra):
    cfg = {
        "model": {
            "provider": "custom",
            "default": "qwen3:14b",
            "base_url": GPU_14B_URL,
        },
        "local_runtime": {
            "enabled": True,
            "engine": "vllm",
            "vllm": {"device": "gpu", "served_model_name": "qwen3:14b"},
        },
    }
    cfg.update(extra)
    return cfg


def _init_agent(cfg, *, context_length, base_url, model, provider="custom"):
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=context_length),
        patch("agent.context_compressor.get_model_context_length", return_value=context_length),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        return AIAgent(
            api_key="local",
            base_url=base_url,
            model=model,
            provider=provider,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_managed_gpu_14b_native_40k_does_not_fail_agent_init():
    """Selected GPU 14B at honest 40960 must init; window stays 40960."""
    agent = _init_agent(
        _gpu_14b_cfg(),
        context_length=NATIVE_14B,
        base_url=GPU_14B_URL,
        model="qwen3:14b",
    )
    assert agent.model == "qwen3:14b"
    assert agent.base_url.rstrip("/") == GPU_14B_URL
    assert agent.context_compressor.context_length == NATIVE_14B
    assert agent.context_compressor.context_length < 64_000


def test_unmanaged_local_below_floor_still_fails_and_does_not_suggest_faking_64k():
    """A non-managed local 32k window still fails init; no 'set context_length to 64k'."""
    with pytest.raises(ValueError) as exc_info:
        _init_agent(
            {
                "model": {
                    "provider": "custom",
                    "default": "tiny-local",
                    "base_url": "http://127.0.0.1:9999/v1",
                },
            },
            context_length=32_768,
            base_url="http://127.0.0.1:9999/v1",
            model="tiny-local",
        )
    err = str(exc_info.value)
    assert "tiny-local" in err
    assert "32,768" in err
    assert "64,000" in err
    assert "this must be at least" not in err
    assert "do not set model.context_length" in err


def test_gpu_14b_init_binds_live_cpu_as_aux_without_switching_main(tmp_path, monkeypatch):
    """CPU 4B live + compression unset → aux pin only; main stays GPU 14B."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    from hermes_cli.vllm_runtime.supervisor import state_path

    cpu_state = state_path("cpu")
    cpu_state.parent.mkdir(parents=True, exist_ok=True)
    cpu_state.write_text(
        json.dumps({
            "base_url": CPU_4B_URL,
            "pid": os.getpid(),
            "served_model_name": "qwen3:4b",
        }),
        encoding="utf-8",
    )

    agent = _init_agent(
        _gpu_14b_cfg(),
        context_length=NATIVE_14B,
        base_url=GPU_14B_URL,
        model="qwen3:14b",
    )
    assert agent.model == "qwen3:14b"
    assert "18435" in agent.base_url
    assert "18436" not in agent.base_url
    assert agent.context_compressor.context_length == NATIVE_14B

    from hermes_cli.config import read_raw_config

    raw = read_raw_config()
    comp = (raw.get("auxiliary") or {}).get("compression") or {}
    assert comp.get("base_url") == CPU_4B_URL
    assert comp.get("model") == "qwen3:4b"
    assert comp.get("provider") == "custom"


@patch("agent.model_metadata.get_model_context_length", return_value=NATIVE_14B)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_aux_at_14b_native_still_requires_64k(mock_get_client, mock_ctx_len):
    """Compression/aux at 40960 still hard-fails; do not fake a 64k override."""
    from agent.context_compressor import ContextCompressor

    agent = AIAgent.__new__(AIAgent)
    agent.model = "qwen3:14b"
    agent.provider = "custom"
    agent.base_url = GPU_14B_URL
    agent.api_key = ""
    agent.api_mode = "chat_completions"
    agent.auth_mode = ""
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent.compression_enabled = True
    agent._print_fn = None
    agent.suppress_status_output = False
    agent._stream_consumers = []
    agent._executing_tools = False
    agent._mute_post_response = False
    agent.status_callback = None
    agent.tool_progress_callback = None
    agent._compression_warning = None
    agent._aux_compression_context_length_config = None
    agent._custom_providers = []
    agent.tools = []
    compressor = MagicMock(spec=ContextCompressor)
    compressor.context_length = 200_000
    compressor.threshold_tokens = 100_000
    compressor.summary_target_ratio = 0.20
    compressor.tail_token_budget = 20_000
    agent.context_compressor = compressor
    mock_client = SimpleNamespace(base_url=GPU_14B_URL, api_key="")
    mock_get_client.return_value = (mock_client, "qwen3:14b")
    agent._emit_status = lambda msg: None

    with pytest.raises(ValueError) as exc_info:
        agent._check_compression_model_feasibility()
    err = str(exc_info.value)
    assert "qwen3:14b" in err
    assert "40,960" in err
    assert "64,000" in err
    assert "below the minimum" in err
    assert "this must be at least" not in err
