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

    monkeypatch.setattr(boot, "_SUPERVISORS", {"gpu": None, "cpu": None})

    def _occupied():
        raise OccupyingLlmError(
            "Another LLM is already running (vllm API on http://127.0.0.1:8000). "
            "Stop it so managed vLLM can use the GPU.")

    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", _occupied)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint._state_endpoint", lambda *a, **k: None)

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
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.probe_served_model_name",
        lambda url, timeout_s=1.5: "qwen3:14b" if url else "")

    resolved = ep.resolve_vllm_endpoint(config={"local_runtime": {"enabled": True, "engine": "vllm"}},
                                        wait_for_boot_s=5.0)
    assert resolved is not None
    assert "59997" in resolved["base_url"]
    assert "openrouter" not in resolved["base_url"]


def test_vllm_boot_in_flight_real_gate(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    # Isolate the hub so a developer cache cannot satisfy the weight gate.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    from hermes_cli.vllm_runtime import endpoint as ep
    from hermes_cli.vllm_runtime.supervisor import configured_model_id, vllm_settings
    from hermes_cli.vllm_runtime.venv import vllm_executable

    enabled = {"local_runtime": {"enabled": True, "engine": "vllm"}}
    assert ep._boot_in_flight(enabled) is False
    exe = vllm_executable()
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("", encoding="utf-8")
    # The venv alone does not kick boot when the configured weights are absent.
    assert ep._boot_in_flight(enabled) is False
    hid = configured_model_id(vllm_settings(enabled))
    cache = tmp_path / "hf" / "hub" / ("models--" + hid.replace("/", "--"))
    cache.mkdir(parents=True)
    (cache / "model.safetensors").write_bytes(b"x")
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


def _managed_chat_config(*, device: str, enabled: bool, base_url: str,
                         provider: str = "custom", vllm_block: dict | None = None) -> dict:
    """Chat on custom (or an explicit request) with the hidden legacy vLLM slot."""
    legacy = {"name": "vLLM", "base_url": "", "enabled": False}
    if vllm_block is not None:
        legacy = vllm_block
    return {
        "model": {"provider": provider, "default": "qwen3:4b", "base_url": base_url},
        "providers": {
            "vllm": legacy,
            "vllm-gpu": {
                "name": "vLLM GPU",
                "base_url": "http://127.0.0.1:18435/v1",
                "model": "qwen3:14b",
            },
            "vllm-cpu": {
                "name": "vLLM CPU",
                "base_url": "http://127.0.0.1:18436/v1",
                "model": "qwen3:4b",
            },
        },
        "local_runtime": {
            "enabled": enabled,
            "engine": "vllm",
            "vllm": {
                "device": device,
                "model": "Qwen/Qwen3-4B" if device == "cpu" else "Qwen/Qwen3-14B-AWQ",
                "served_model_name": "qwen3:4b" if device == "cpu" else "qwen3:14b",
            },
        },
    }


def _install_managed_chat(home, monkeypatch, cfg: dict) -> list:
    """Write *cfg* and record which device an on-demand boot would start.

    The boot gate is real (``local_runtime.enabled`` + engine vllm). The
    serve itself is fake: it records the device and writes that device's
    state file so GET /v1/models can succeed without a GPU.
    """
    (home / "config.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
    started: list[str] = []

    def _fake_ensure(config=None, force=False):
        from hermes_cli.local_engines import vllm_device_from_config
        from hermes_cli.vllm_runtime.supervisor import state_path

        device = vllm_device_from_config(config)
        started.append(device)
        port = 18436 if device == "cpu" else 18435
        path = state_path(device)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "base_url": f"http://127.0.0.1:{port}/v1",
            "pid": os.getpid(),
            "served_model_name": "qwen3:4b" if device == "cpu" else "qwen3:14b",
        }), encoding="utf-8")
        return object()

    monkeypatch.setattr("hermes_cli.local_engines.ensure_managed_engine", _fake_ensure)
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.venv_ready", lambda device=None: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.configured_cache_missing", lambda settings: False)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.probe_served_model_name",
        lambda url, timeout_s=1.5: "ready" if url else "")
    return started


def test_disabled_legacy_vllm_does_not_fail_custom_managed_chat(tmp_path, monkeypatch):
    """Hidden ``providers.vllm`` must not fail init when chat is custom on either device URL."""
    home = _home(tmp_path, monkeypatch)
    from hermes_cli.runtime_provider import resolve_runtime_provider

    for device, url in (("gpu", "http://127.0.0.1:18435/v1"), ("cpu", "http://127.0.0.1:18436/v1")):
        started = _install_managed_chat(home, monkeypatch, _managed_chat_config(
            device=device, enabled=True, base_url=url))
        runtime = resolve_runtime_provider(requested="custom")
        assert runtime["provider"] == "custom"
        assert url.rstrip("/") in runtime["base_url"]
        assert "openrouter" not in (runtime.get("base_url") or "").lower()
        assert started == [device]

        started_alias = _install_managed_chat(home, monkeypatch, _managed_chat_config(
            device=device, enabled=True, base_url=url))
        # Drop the state file the previous boot wrote so this request starts again.
        from hermes_cli.vllm_runtime.supervisor import state_path
        state_path(device).unlink(missing_ok=True)
        alias = resolve_runtime_provider(requested="vllm")
        assert alias["provider"] == "custom"
        assert url.rstrip("/") in alias["base_url"]
        assert started_alias == [device]


def test_explicit_vllm_off_does_not_spawn(tmp_path, monkeypatch):
    """``local_runtime.enabled: false`` is Turn off — do not start either device."""
    home = _home(tmp_path, monkeypatch)
    started = _install_managed_chat(home, monkeypatch, _managed_chat_config(
        device="cpu", enabled=False, base_url="http://127.0.0.1:18436/v1"))
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(ValueError, match="vLLM is offline") as exc:
        resolve_runtime_provider(requested="custom")
    assert "Turn on" in str(exc.value)
    assert "disabled in config" not in str(exc.value)
    assert started == []


def test_selected_device_starts_when_engine_is_on(tmp_path, monkeypatch):
    """Only the device in ``local_runtime.vllm.device`` starts, and only when enabled."""
    home = _home(tmp_path, monkeypatch)
    started = _install_managed_chat(home, monkeypatch, _managed_chat_config(
        device="gpu", enabled=True, base_url="http://127.0.0.1:18435/v1"))
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider()
    assert runtime["provider"] == "custom"
    assert "18435" in runtime["base_url"]
    assert started == ["gpu"]


def test_other_managed_device_url_does_not_spawn(tmp_path, monkeypatch):
    """Chat pinned at the sibling serve must not start the selected device or the sibling."""
    home = _home(tmp_path, monkeypatch)
    started = _install_managed_chat(home, monkeypatch, _managed_chat_config(
        device="cpu", enabled=True, base_url="http://127.0.0.1:18435/v1"))
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(requested="custom")
    assert runtime["provider"] == "custom"
    assert "18435" in runtime["base_url"]
    assert started == []


def _write_device_state(device: str, *, port: int, served: str) -> None:
    from hermes_cli.vllm_runtime.supervisor import state_path

    path = state_path(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "base_url": f"http://127.0.0.1:{port}/v1",
        "pid": os.getpid(),
        "served_model_name": served,
    }), encoding="utf-8")


def test_gpu_chat_keeps_14b_when_cpu_4b_is_live(tmp_path, monkeypatch):
    """device=gpu + GPU pin must stay on 14b even when CPU 4B is healthy.

    The send-time writer was ``follow_live_managed_vllm`` (and
    ``_resolve_agent_model_runtime`` applying its served name): it followed
    whichever managed serve ``resolve_vllm_endpoint`` returned — the selected
    device, or a last-ready CPU record — and replaced qwen3:14b @ 18435.
    """
    home = _home(tmp_path, monkeypatch)
    _write_device_state("gpu", port=18435, served="qwen3:14b")
    _write_device_state("cpu", port=18436, served="qwen3:4b")
    cfg = _managed_chat_config(
        device="gpu", enabled=True, base_url="http://127.0.0.1:18435/v1")
    cfg["model"]["default"] = "qwen3:14b"
    (home / "config.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: cfg)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    started = _install_managed_chat(home, monkeypatch, cfg)

    from hermes_cli.runtime_provider import (
        _resolve_named_custom_runtime, resolve_runtime_provider,
    )
    from hermes_cli.vllm_runtime import endpoint as ep

    # Even if resolution claims the CPU serve is the live managed endpoint,
    # a GPU pin must not follow it.
    real_resolve = ep.resolve_vllm_endpoint
    monkeypatch.setattr(
        ep, "resolve_vllm_endpoint",
        lambda *a, **k: {
            "base_url": "http://127.0.0.1:18436/v1",
            "served_model_name": "qwen3:4b",
            "api_key": "",
            "pid": os.getpid(),
        })
    assert ep.follow_live_managed_vllm(
        "http://127.0.0.1:18435/v1", "qwen3:14b") is None
    monkeypatch.setattr(ep, "resolve_vllm_endpoint", real_resolve)

    runtime = resolve_runtime_provider(requested="custom")
    assert runtime["provider"] == "custom"
    assert "18435" in runtime["base_url"]
    assert "18436" not in runtime["base_url"]

    named = _resolve_named_custom_runtime(
        requested_provider="custom",
        explicit_base_url="http://127.0.0.1:18435/v1",
        target_model="qwen3:14b")
    assert named is not None
    assert "18435" in named["base_url"]
    assert "18436" not in named["base_url"]

    from tui_gateway.server import _resolve_agent_model_runtime

    model, resolved = _resolve_agent_model_runtime(
        {"model": "qwen3:14b", "provider": "custom",
         "base_url": "http://127.0.0.1:18435/v1"},
        None)
    assert model == "qwen3:14b"
    assert "18435" in resolved["base_url"]
    assert "18436" not in resolved["base_url"]
    assert started == [] or started == ["gpu"]
    assert home


def test_disabled_remote_vllm_still_raises(tmp_path, monkeypatch):
    """A remote ``providers.vllm`` the user turned off stays disabled."""
    home = _home(tmp_path, monkeypatch)
    _install_managed_chat(home, monkeypatch, _managed_chat_config(
        device="gpu", enabled=True, base_url="http://gpu-box.example:8000/v1",
        provider="vllm",
        vllm_block={
            "name": "GPU box",
            "base_url": "http://gpu-box.example:8000/v1",
            "enabled": False,
        },
    ))
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(ValueError, match="disabled in config"):
        resolve_runtime_provider(requested="vllm")
