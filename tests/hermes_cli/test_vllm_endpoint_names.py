"""Managed vLLM serves keep distinct, stable custom endpoints.

Chat stays ``provider: custom`` and points ``model.base_url`` at the selected
device. Starting GPU does not replace the CPU record, and the reverse.
"""

import yaml

from hermes_cli.inventory import ConfigContext, _vllm_runtime_row
from hermes_cli.model_switch import list_authenticated_providers, switch_model
from hermes_cli.vllm_runtime.device import (
    CPU_ENDPOINT_NAME,
    GPU_ENDPOINT_NAME,
    managed_endpoint_key,
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
    assert managed_endpoint_key("gpu") == "vllm-gpu"
    assert managed_endpoint_key("cpu") == "vllm-cpu"
    assert managed_endpoint_key("gpu") != managed_endpoint_key("cpu")


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
        lambda _cfg=None, device=None: {"served_model_name": "qwen3:4b"},
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


class _Serve:
    def __init__(self, url: str):
        self.base_url = url


def _isolate_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)
    return home


def _supervisor(device=None):
    if device == "cpu":
        return _Serve("http://127.0.0.1:18436/v1")
    if device == "gpu":
        return _Serve("http://127.0.0.1:18435/v1")
    return None


def test_starting_one_device_does_not_overwrite_the_other_endpoint(tmp_path, monkeypatch):
    """CPU start leaves the GPU record; GPU start leaves the CPU record.

    The old writer stored both serves in ``providers.vllm`` and renamed that
    one slot to whichever device had just started.
    """
    home = _isolate_home(tmp_path, monkeypatch)
    monkeypatch.setattr("hermes_cli.vllm_runtime.bootstrap.get_supervisor", _supervisor)
    gpu = {
        "name": "vLLM GPU",
        "base_url": "http://127.0.0.1:53351/v1",
        "model": "hermes3:8b",
    }
    cpu = {
        "name": "vLLM CPU",
        "base_url": "http://127.0.0.1:18436/v1",
        "model": "qwen3:4b",
    }
    remote = {"name": "GPU box", "base_url": "http://gpu-box.example:8000/v1"}
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {
            "provider": "custom",
            "base_url": "http://127.0.0.1:18436/v1",
            "default": "qwen3:4b",
        },
        "providers": {"vllm": dict(remote), "vllm-gpu": dict(gpu), "vllm-cpu": dict(cpu)},
        "auxiliary": {
            "compression": {
                "provider": "custom",
                "model": "qwen3:4b",
                "base_url": "http://127.0.0.1:18436/v1",
            },
        },
        "local_runtime": {
            "enabled": True,
            "engine": "vllm",
            "vllm": {
                "device": "cpu",
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "served_model_name": "qwen3:4b",
            },
        },
    }), encoding="utf-8")

    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.bootstrap import activate_vllm_provider
    from hermes_cli.web_routers.config_env import _custom_endpoint_response

    activate_vllm_provider(load_config())
    after_cpu = load_config()
    assert after_cpu["providers"]["vllm-gpu"]["name"] == "vLLM GPU"
    assert after_cpu["providers"]["vllm-gpu"]["base_url"] == gpu["base_url"]
    assert after_cpu["providers"]["vllm-gpu"]["model"] == "hermes3:8b"
    assert after_cpu["providers"]["vllm-cpu"]["name"] == "vLLM CPU"
    assert after_cpu["providers"]["vllm-cpu"]["base_url"] == "http://127.0.0.1:18436/v1"
    assert after_cpu["providers"]["vllm"] == remote
    assert after_cpu["model"]["provider"] == "custom"
    assert after_cpu["model"]["base_url"] == "http://127.0.0.1:18436/v1"
    assert after_cpu["auxiliary"]["compression"]["provider"] == "custom"
    assert after_cpu["auxiliary"]["compression"]["base_url"] == "http://127.0.0.1:18436/v1"

    listed = _custom_endpoint_response(after_cpu)["endpoints"]
    by_id = {row["id"]: row for row in listed}
    assert set(by_id) >= {"vllm-gpu", "vllm-cpu"}
    assert "custom" not in by_id
    assert by_id["vllm-cpu"]["is_current"] is True
    assert by_id["vllm-gpu"]["is_current"] is False
    assert by_id["vllm-gpu"]["name"] == "vLLM GPU"
    assert by_id["vllm-cpu"]["name"] == "vLLM CPU"

    from cli import save_config_value

    save_config_value("local_runtime.vllm.device", "gpu")
    cpu_frozen = {
        "name": after_cpu["providers"]["vllm-cpu"]["name"],
        "base_url": after_cpu["providers"]["vllm-cpu"]["base_url"],
        "model": after_cpu["providers"]["vllm-cpu"]["model"],
    }
    activate_vllm_provider(load_config())
    after_gpu = load_config()
    assert after_gpu["providers"]["vllm-cpu"]["name"] == cpu_frozen["name"]
    assert after_gpu["providers"]["vllm-cpu"]["base_url"] == cpu_frozen["base_url"]
    assert after_gpu["providers"]["vllm-cpu"]["model"] == cpu_frozen["model"]
    assert after_gpu["providers"]["vllm-gpu"]["name"] == "vLLM GPU"
    assert after_gpu["providers"]["vllm-gpu"]["base_url"] == "http://127.0.0.1:18435/v1"
    assert after_gpu["providers"]["vllm"] == remote
    assert after_gpu["model"]["provider"] == "custom"
    assert after_gpu["model"]["base_url"] == "http://127.0.0.1:18435/v1"
    assert after_gpu["auxiliary"]["compression"]["base_url"] == "http://127.0.0.1:18436/v1"


def _picker_rows(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    from hermes_cli.inventory import build_models_payload, load_picker_context

    return build_models_payload(
        load_picker_context(), explicit_only=True,
        probe_custom_providers=False, probe_current_custom_provider=False,
    )["providers"]


def test_picker_lists_each_managed_device_once(tmp_path, monkeypatch):
    """A managed loopback in ``providers.vllm`` is hidden, and the synthetic
    runtime row does not repeat ``providers.vllm-gpu`` / ``providers.vllm-cpu``.

    Chat stays ``provider: custom``. The same URL is not also a ``custom`` row.
    """
    home = _isolate_home(tmp_path, monkeypatch)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {
            "provider": "custom",
            "base_url": "http://127.0.0.1:18435/v1",
            "default": "qwen3:14b",
        },
        "providers": {
            "vllm": {
                "name": "vLLM CPU",
                "base_url": "http://127.0.0.1:18436/v1",
                "model": "qwen3:4b",
            },
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
            "enabled": True,
            "engine": "vllm",
            "vllm": {"device": "gpu", "served_model_name": "qwen3:14b"},
        },
    }), encoding="utf-8")

    from hermes_cli.config import load_config
    from hermes_cli.web_routers.config_env import _custom_endpoint_response

    rows = _picker_rows(monkeypatch)
    saved = load_config()
    legacy = saved["providers"]["vllm"]
    assert str(legacy.get("base_url") or legacy.get("url") or "") == ""
    assert legacy.get("enabled") is False
    assert saved["providers"]["vllm-gpu"]["base_url"] == "http://127.0.0.1:18435/v1"
    assert saved["providers"]["vllm-cpu"]["base_url"] == "http://127.0.0.1:18436/v1"
    assert saved["model"]["provider"] == "custom"

    names = [row["name"] for row in rows]
    assert names.count("vLLM GPU") == 1
    assert names.count("vLLM CPU") == 1
    assert "Custom endpoint" not in names
    assert "vllm" not in {row["slug"] for row in rows}
    assert "custom" not in {row["slug"] for row in rows}
    gpu = "http://127.0.0.1:18435/v1"
    cpu = "http://127.0.0.1:18436/v1"

    def _url(row: dict) -> str:
        return str(row.get("api_url") or "").rstrip("/")

    assert [row["slug"] for row in rows if _url(row) == gpu] == ["vllm-gpu"]
    assert [row["slug"] for row in rows if _url(row) == cpu] == ["vllm-cpu"]
    by_slug = {row["slug"]: row for row in rows}
    assert by_slug["vllm-gpu"]["is_current"] is True
    assert by_slug["vllm-cpu"]["is_current"] is False

    ids = [endpoint["id"] for endpoint in _custom_endpoint_response(saved)["endpoints"]]
    assert ids.count("vllm-gpu") == 1
    assert ids.count("vllm-cpu") == 1
    assert "vllm" not in ids
    assert "custom" not in ids


def test_picker_keeps_a_remote_vllm_endpoint(tmp_path, monkeypatch):
    """A non-loopback ``providers.vllm`` stays beside the two device endpoints."""
    home = _isolate_home(tmp_path, monkeypatch)
    remote = "http://gpu-box.example:8000/v1"
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {
            "provider": "custom",
            "base_url": "http://127.0.0.1:18435/v1",
            "default": "qwen3:14b",
        },
        "providers": {
            "vllm": {"name": "GPU box", "base_url": remote},
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
            "enabled": True,
            "engine": "vllm",
            "vllm": {"device": "gpu", "served_model_name": "qwen3:14b"},
        },
    }), encoding="utf-8")

    from hermes_cli.config import load_config

    rows = _picker_rows(monkeypatch)
    assert load_config()["providers"]["vllm"]["base_url"] == remote
    remote_rows = [row for row in rows if row["slug"] == "vllm"]
    assert len(remote_rows) == 1
    assert remote_rows[0]["name"] == "GPU box"
    assert remote_rows[0]["api_url"].rstrip("/") == remote
    names = [row["name"] for row in rows]
    assert names.count("vLLM GPU") == 1
    assert names.count("vLLM CPU") == 1
    assert "Custom endpoint" not in names


def _write_device_state(device: str, *, port: int, served: str) -> None:
    import json

    from hermes_cli.vllm_runtime.supervisor import state_path

    path = state_path(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "base_url": f"http://127.0.0.1:{port}/v1",
        "pid": 1,
        "served_model_name": served,
    }), encoding="utf-8")


def test_cpu_use_smol_does_not_write_gpu_endpoint(tmp_path, monkeypatch):
    """CPU Use of SmolLM3-3B must not label the GPU row or GPU chat URL.

    The live bug wrote providers.vllm-gpu.model / model.default = SmolLM3-3B
    and model.base_url = :18435, so the picker said Smol was running on GPU.
    """
    home = _isolate_home(tmp_path, monkeypatch)
    monkeypatch.setattr("hermes_cli.vllm_runtime.bootstrap.get_supervisor", _supervisor)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.repo_quant_method", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.recommend.parser_for_hf_id", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.recommend.serve_len_cap", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.cached_model_config", lambda *_a, **_k: {})
    gpu_url = "http://127.0.0.1:18435/v1"
    cpu_url = "http://127.0.0.1:18436/v1"
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {
            "provider": "custom",
            "base_url": gpu_url,
            "default": "qwen3:14b",
        },
        "providers": {
            "vllm-gpu": {
                "name": "vLLM GPU",
                "base_url": gpu_url,
                "model": "qwen3:14b",
                "models": {"qwen3:14b": {}},
            },
            "vllm-cpu": {
                "name": "vLLM CPU",
                "base_url": cpu_url,
                "model": "qwen3:4b",
            },
        },
        "local_runtime": {
            "enabled": True,
            "engine": "vllm",
            "vllm": {
                "device": "cpu",
                "model": "Qwen/Qwen3-14B-AWQ",
                "served_model_name": "qwen3:14b",
            },
        },
    }), encoding="utf-8")
    _write_device_state("gpu", port=18435, served="qwen3:14b")
    _write_device_state("cpu", port=18436, served="SmolLM3-3B")

    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.bootstrap import activate_vllm_provider
    from hermes_cli.vllm_runtime.inventory import apply_vllm_model
    from hermes_cli.web_routers.config_env import _custom_endpoint_response

    apply_vllm_model("HuggingFaceTB/SmolLM3-3B")
    activate_vllm_provider(load_config())
    after = load_config()
    gpu = after["providers"]["vllm-gpu"]
    cpu = after["providers"]["vllm-cpu"]
    assert gpu["model"] != "SmolLM3-3B"
    assert gpu["model"] == "qwen3:14b"
    assert gpu["base_url"].rstrip("/") == gpu_url
    assert cpu["model"] == "SmolLM3-3B"
    assert cpu["base_url"].rstrip("/") == cpu_url
    assert after["model"]["base_url"].rstrip("/") == cpu_url
    assert after["model"]["default"] == "SmolLM3-3B"
    devices = after["local_runtime"]["vllm"].get("devices") or {}
    assert (devices.get("gpu") or {}).get("model") == "Qwen/Qwen3-14B-AWQ"
    assert (devices.get("gpu") or {}).get("served_model_name") == "qwen3:14b"
    assert (devices.get("cpu") or {}).get("model") == "HuggingFaceTB/SmolLM3-3B"
    raw = yaml.safe_load((home / "config.yaml").read_text())
    leftover = (raw.get("local_runtime") or {}).get("vllm") or {}
    assert "model" not in leftover
    assert "served_model_name" not in leftover
    assert not isinstance(leftover.get("cpu"), dict)

    listed = {row["id"]: row for row in _custom_endpoint_response(after)["endpoints"]}
    assert listed["vllm-gpu"]["model"] != "SmolLM3-3B"
    assert listed["vllm-gpu"]["model"] == "qwen3:14b"
    assert listed["vllm-cpu"]["model"] == "SmolLM3-3B"
    assert listed["vllm-gpu"]["name"] == "vLLM GPU"
    assert listed["vllm-cpu"]["name"] == "vLLM CPU"


def test_picker_does_not_show_cpu_smol_as_gpu_model(tmp_path, monkeypatch):
    """A leftover vllm-gpu.model=SmolLM3-3B must not stay the GPU picker row."""
    home = _isolate_home(tmp_path, monkeypatch)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {
            "provider": "custom",
            "base_url": "http://127.0.0.1:18435/v1",
            "default": "SmolLM3-3B",
        },
        "providers": {
            "vllm-gpu": {
                "name": "vLLM GPU",
                "base_url": "http://127.0.0.1:18435/v1",
                "model": "SmolLM3-3B",
                "models": {"qwen3:14b": {}, "SmolLM3-3B": {}},
            },
            "vllm-cpu": {
                "name": "vLLM CPU",
                "base_url": "http://127.0.0.1:18436/v1",
                "model": "SmolLM3-3B",
            },
        },
        "local_runtime": {
            "enabled": True,
            "engine": "vllm",
            "vllm": {
                "device": "gpu",
                "model": "Qwen/Qwen3-14B-AWQ",
                "served_model_name": "qwen3:14b",
            },
        },
    }), encoding="utf-8")
    _write_device_state("gpu", port=18435, served="qwen3:14b")
    _write_device_state("cpu", port=18436, served="SmolLM3-3B")

    rows = _picker_rows(monkeypatch)
    by_slug = {row["slug"]: row for row in rows}
    assert by_slug["vllm-gpu"]["name"] == "vLLM GPU"
    assert by_slug["vllm-cpu"]["name"] == "vLLM CPU"
    gpu_models = [str(m) for m in (by_slug["vllm-gpu"].get("models") or [])]
    cpu_models = [str(m) for m in (by_slug["vllm-cpu"].get("models") or [])]
    assert gpu_models[0] != "SmolLM3-3B"
    assert "qwen3:14b" in gpu_models
    assert cpu_models[0] == "SmolLM3-3B"
