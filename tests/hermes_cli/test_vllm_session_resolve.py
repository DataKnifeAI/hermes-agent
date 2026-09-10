"""Session resolve + occupancy for managed vLLM (issues #4 / #5).

``provider: vllm`` must boot or raise — never fall through to OpenRouter.
Temp HERMES_HOME, real imports, fake HTTP / mocked occupancy. No live GPU.
"""

from __future__ import annotations

import json
import os
import time

import pytest
import yaml

from hermes_cli.vllm_runtime.occupancy import OccupyingLlmError


def _home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    (home / "config.yaml").write_text(
        yaml.dump({
            "local_runtime": {"enabled": True, "engine": "vllm"},
            "model": {"provider": "vllm"},
        }),
        encoding="utf-8")
    return home


def test_vllm_runtime_fake_endpoint_is_not_openrouter(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: {"base_url": "http://127.0.0.1:18435/v1", "api_key": ""})
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: {
        "local_runtime": {"enabled": True, "engine": "vllm"},
        "model": {"provider": "vllm"},
        "providers": {},
    })

    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested="vllm")
    assert runtime is not None
    host = (runtime.get("base_url") or "").lower()
    assert "openrouter" not in host
    assert "127.0.0.1" in host
    assert runtime["source"] == "local-runtime"


def test_vllm_missing_server_raises_not_cloud(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: {
        "local_runtime": {"enabled": True, "engine": "vllm"},
        "model": {"provider": "vllm"},
        "providers": {},
    })

    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(ValueError, match="isn't running"):
        resolve_runtime_provider(requested="vllm")


def test_vllm_occupancy_surfaces_on_session_resolve(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: None)

    def _occupied():
        raise OccupyingLlmError(
            "Another LLM is already running (Ollama on http://127.0.0.1:11434). "
            "Stop it so managed vLLM can use the GPU.")

    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", _occupied)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: {
        "local_runtime": {"enabled": True, "engine": "vllm"},
        "model": {"provider": "vllm"},
        "providers": {},
    })

    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(OccupyingLlmError, match="Stop it so managed vLLM"):
        resolve_runtime_provider(requested="vllm")


def test_vllm_remote_url_is_not_rewritten_to_loopback(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    kicked = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: kicked.append("resolve") or None)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: {
        "local_runtime": {"enabled": True, "engine": "vllm"},
        "model": {"provider": "vllm", "base_url": "http://gpu-box.example:8000/v1"},
        "providers": {"vllm": {"base_url": "http://gpu-box.example:8000/v1"}},
    })

    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested="vllm")
    assert "gpu-box.example" in runtime["base_url"]
    assert "openrouter" not in (runtime.get("base_url") or "").lower()
    assert kicked == []


def test_ensure_vllm_runtime_occupancy_does_not_start(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    import hermes_cli.vllm_runtime.bootstrap as boot

    monkeypatch.setattr(boot, "_SUPERVISOR", None)

    def _occupied():
        raise OccupyingLlmError(
            "Another LLM is already running (vllm API on http://127.0.0.1:8000). "
            "Stop it so managed vLLM can use the GPU.")

    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", _occupied)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint._state_endpoint", lambda: None)

    with pytest.raises(OccupyingLlmError, match="Stop it so managed vLLM"):
        boot.ensure_vllm_runtime(
            {"local_runtime": {"enabled": True, "engine": "vllm"}}, force=True)
    assert boot.get_supervisor() is None
    assert not (home / "runtimes" / "vllm" / "server.json").exists()


def test_vllm_endpoint_kicks_boot_and_waits(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    from hermes_cli.vllm_runtime import endpoint as ep
    from hermes_cli.vllm_runtime.supervisor import state_path

    monkeypatch.setattr(ep, "_boot_in_flight", lambda config: True)
    monkeypatch.setattr(ep, "_pid_alive", lambda pid: True)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)

    def _fake_ensure(config=None, force=False):
        time.sleep(0.3)
        state_path().parent.mkdir(parents=True, exist_ok=True)
        state_path().write_text(json.dumps({
            "base_url": "http://127.0.0.1:59997/v1",
            "pid": os.getpid(),
        }), encoding="utf-8")

    monkeypatch.setattr("hermes_cli.local_engines.ensure_managed_engine", _fake_ensure)

    resolved = ep.resolve_vllm_endpoint(config={"local_runtime": {"enabled": True, "engine": "vllm"}},
                                        wait_for_boot_s=5.0)
    assert resolved is not None
    assert "59997" in resolved["base_url"]
    assert "openrouter" not in resolved["base_url"]


def test_vllm_boot_in_flight_real_gate(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    from hermes_cli.vllm_runtime import endpoint as ep
    from hermes_cli.vllm_runtime.venv import vllm_executable

    enabled = {"local_runtime": {"enabled": True, "engine": "vllm"}}
    assert ep._boot_in_flight(enabled) is False
    exe = vllm_executable()
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("", encoding="utf-8")
    assert ep._boot_in_flight(enabled) is True
    assert ep._boot_in_flight({"local_runtime": {"enabled": False, "engine": "vllm"}}) is False
    assert ep._boot_in_flight({"local_runtime": {"enabled": True, "engine": "llamacpp"}}) is False
    assert home  # HERMES_HOME isolation


def test_stale_loopback_custom_pin_follows_live_vllm(tmp_path, monkeypatch):
    """A session pinned to 18435 after an ephemeral rebind must hit the live port."""
    home = _home(tmp_path, monkeypatch)
    from hermes_cli.vllm_runtime.supervisor import state_path

    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "base_url": "http://127.0.0.1:53351/v1",
        "pid": os.getpid(),
        "served_model_name": "qwen3:14b",
    }), encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.load_config",
        lambda: {
            "local_runtime": {"enabled": True, "engine": "vllm"},
            "model": {"provider": "vllm", "default": "qwen3:14b",
                      "base_url": "http://127.0.0.1:53351/v1"},
            "providers": {},
        })

    from hermes_cli.runtime_provider import _resolve_named_custom_runtime

    runtime = _resolve_named_custom_runtime(
        requested_provider="custom",
        explicit_base_url="http://127.0.0.1:18435/v1",
        target_model="qwen3:8b")
    assert runtime is not None
    assert "53351" in runtime["base_url"]
    assert "18435" not in runtime["base_url"]
    assert runtime["source"] == "local-runtime"
    assert home


def test_follow_live_managed_vllm_leaves_foreign_and_remote(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    from hermes_cli.vllm_runtime import endpoint as ep
    from hermes_cli.vllm_runtime.supervisor import state_path

    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "base_url": "http://127.0.0.1:53351/v1",
        "pid": os.getpid(),
        "served_model_name": "qwen3:14b",
    }), encoding="utf-8")

    assert ep.follow_live_managed_vllm("http://127.0.0.1:18434/v1", "qwen3:8b") is None
    assert ep.follow_live_managed_vllm("http://127.0.0.1:11434/v1", "qwen3:8b") is None
    assert ep.follow_live_managed_vllm("http://gpu-box.example:8000/v1", "qwen3:8b") is None
    followed = ep.follow_live_managed_vllm("http://127.0.0.1:18435/v1", "qwen3:8b")
    assert followed is not None
    assert followed["base_url"] == "http://127.0.0.1:53351/v1"
    assert followed["served_model_name"] == "qwen3:14b"
    assert home


def test_session_override_does_not_restamp_stale_vllm_port(tmp_path, monkeypatch):
    """``_resolve_agent_model_runtime`` re-applies persisted base_url — must not undo follow."""
    _home(tmp_path, monkeypatch)
    from hermes_cli.vllm_runtime.supervisor import state_path

    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps({
        "base_url": "http://127.0.0.1:53351/v1",
        "pid": os.getpid(),
        "served_model_name": "qwen3:14b",
    }), encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **k: {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "http://127.0.0.1:53351/v1",
            "api_key": "no-key-required",
            "source": "local-runtime",
        })

    from tui_gateway.server import _resolve_agent_model_runtime

    model, runtime = _resolve_agent_model_runtime(
        {"model": "qwen3:8b", "provider": "custom",
         "base_url": "http://127.0.0.1:18435/v1"},
        None)
    assert "53351" in runtime["base_url"]
    assert "18435" not in runtime["base_url"]
    assert model == "qwen3:14b"


def test_vllm_endpoint_wait_zero_does_not_kick(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    from hermes_cli.vllm_runtime import endpoint as ep

    kicked = []
    monkeypatch.setattr(ep, "_boot_in_flight", lambda config: True)
    monkeypatch.setattr(ep, "_kick_managed_boot", lambda config: kicked.append("kick"))
    assert ep.resolve_vllm_endpoint(wait_for_boot_s=0) is None
    assert kicked == []
