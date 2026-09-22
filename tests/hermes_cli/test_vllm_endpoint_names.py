"""Managed vLLM serves keep distinct display names.

Routing stays ``provider: custom`` / ``provider: vllm``. Only the label saved
and shown for the two loopback serves changes.
"""

from hermes_cli.inventory import ConfigContext, _vllm_runtime_row
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli.vllm_runtime.device import (
    CPU_ENDPOINT_NAME,
    GPU_ENDPOINT_NAME,
    managed_endpoint_name,
    managed_endpoint_name_for_url,
)


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def test_device_names_are_unique():
    assert managed_endpoint_name("gpu") == GPU_ENDPOINT_NAME == "vLLM GPU"
    assert managed_endpoint_name("cpu") == CPU_ENDPOINT_NAME == "vLLM CPU"
    assert managed_endpoint_name("gpu") != managed_endpoint_name("cpu")


def test_managed_loopback_ports_map_to_device_names():
    assert managed_endpoint_name_for_url("http://127.0.0.1:18435/v1") == "vLLM GPU"
    assert managed_endpoint_name_for_url("http://localhost:18436/v1/") == "vLLM CPU"
    assert managed_endpoint_name_for_url("http://[::1]:18436/v1") == "vLLM CPU"
    assert managed_endpoint_name_for_url("http://127.0.0.1:11434/v1") is None
    assert managed_endpoint_name_for_url("http://127.0.0.1:18434/v1") is None
    assert managed_endpoint_name_for_url("http://127.0.0.1:1234/v1") is None
    assert managed_endpoint_name_for_url("https://gpu.example:18435/v1") is None


def test_rebound_loopback_follows_live_device_state(monkeypatch):
    def _state(device=None, config=None):
        if device == "cpu":
            return {"base_url": "http://127.0.0.1:53351/v1"}
        return None

    monkeypatch.setattr("hermes_cli.vllm_runtime.endpoint._state_endpoint", _state)
    assert managed_endpoint_name_for_url("http://127.0.0.1:53351/v1") == "vLLM CPU"
    assert managed_endpoint_name_for_url("http://10.0.0.4:53351/v1") is None


def test_auto_provider_name_uses_device_labels():
    from hermes_cli.main_provider_setup import _auto_provider_name

    assert _auto_provider_name("http://127.0.0.1:18435/v1") == "vLLM GPU"
    assert _auto_provider_name("http://127.0.0.1:18436/v1") == "vLLM CPU"
    assert _auto_provider_name("http://localhost:11434/v1") == "Local (localhost:11434)"
    assert _auto_provider_name("http://127.0.0.1:1234/v1") == "Local (127.0.0.1:1234)"


def test_bare_custom_picker_row_names_the_cpu_serve(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: None)

    providers = list_authenticated_providers(
        current_provider="custom",
        current_base_url="http://127.0.0.1:18436/v1",
        current_model="qwen3:4b",
        user_providers={},
        custom_providers=[],
        max_models=50,
    )

    bare = next(p for p in providers if p["slug"] == "custom")
    assert bare["name"] == "vLLM CPU"
    assert bare["slug"] == "custom"
    assert bare["models"] == ["qwen3:4b"]

    foreign = list_authenticated_providers(
        current_provider="custom",
        current_base_url="https://www.ccsub.net/v1",
        current_model="gpt-4o",
        user_providers={},
        custom_providers=[],
        max_models=50,
    )
    other = next(p for p in foreign if p["slug"] == "custom")
    assert other["name"] == "Custom endpoint"


def test_vllm_runtime_row_names_the_configured_device(monkeypatch):
    cfg = {
        "local_runtime": {
            "engine": "vllm",
            "vllm": {"device": "cpu", "served_model_name": "qwen3:4b"},
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.vllm_settings",
        lambda _cfg=None: {"served_model_name": "qwen3:4b"},
    )

    row = _vllm_runtime_row(ConfigContext(
        current_provider="vllm",
        current_model="qwen3:4b",
        current_base_url="http://127.0.0.1:18436/v1",
        user_providers={},
        custom_providers=[],
    ))

    assert row["slug"] == "vllm"
    assert row["name"] == "vLLM CPU"
    assert row["models"] == ["qwen3:4b"]


def test_switch_label_names_managed_cpu_without_changing_provider(monkeypatch):
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="qwen3:4b",
        current_provider="custom",
        current_model="qwen3:4b",
        current_base_url="http://127.0.0.1:18436/v1",
        current_api_key="local",
        explicit_provider="custom",
        user_providers={},
        custom_providers=[],
    )

    assert result.success is True
    assert result.target_provider == "custom"
    assert result.provider_label == "vLLM CPU"
    assert result.base_url == "http://127.0.0.1:18436/v1"
