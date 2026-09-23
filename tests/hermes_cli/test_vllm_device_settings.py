"""Device-keyed ``local_runtime.vllm.devices.<id>`` — isolation + migrate-on-read."""

from __future__ import annotations

import json

import yaml
from fastapi.testclient import TestClient

from hermes_cli.vllm_runtime.recommend import as_vllm_config, recommend_vllm
from hermes_cli.vllm_runtime.settings import (
    migrate_vllm_devices, persist_migrated_vllm, vllm_settings,
)


_GIB = 1 << 30
_SMOL = "HuggingFaceTB/SmolLM3-3B"
_GPU_14B = "Qwen/Qwen3-14B-AWQ"


def _client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr("cli._hermes_home", home)
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.gated_repo_reason", lambda hid: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.hf_repo_access_issue", lambda hid: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)
    from hermes_cli.web_routers import local_models as lm
    if lm._QUICKSTART_LOCK.locked():
        lm._QUICKSTART_LOCK.release()
    from hermes_cli import web_server

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client, home


def _seed_cache(tmp_path, monkeypatch, *repos):
    hub = tmp_path / "hf-hub"
    for repo in repos:
        root = hub / ("models--" + repo.replace("/", "--"))
        snap = root / "snapshots" / "main"
        snap.mkdir(parents=True, exist_ok=True)
        (root / "refs").mkdir(parents=True, exist_ok=True)
        (root / "refs" / "main").write_text("main", encoding="utf-8")
        (snap / "config.json").write_text(
            '{"max_position_embeddings": 40960, "torch_dtype": "bfloat16"}',
            encoding="utf-8",
        )
        (snap / "w.bin").write_bytes(b"y" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))


def _write_cfg(home, payload):
    (home / "config.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def _devices(cfg):
    vllm = ((cfg.get("local_runtime") or {}).get("vllm") or {})
    raw = vllm.get("devices") if isinstance(vllm.get("devices"), dict) else {}
    return raw


def test_cpu_use_smol_does_not_change_devices_gpu(tmp_path, monkeypatch):
    """CPU Use writes devices.cpu only. The GPU checkpoint stays put."""
    client, home = _client(tmp_path, monkeypatch)
    gpu_before = {
        "model": _GPU_14B,
        "served_model_name": "qwen3:14b",
        "max_model_len": 40960,
        "quantization": "awq",
        "kv_cache_dtype": "fp8",
        "tool_call_parser": "hermes",
    }
    _write_cfg(home, {
        "model": {},
        "local_runtime": {
            "enabled": True,
            "engine": "vllm",
            "vllm": {
                "selected": "cpu",
                "device": "cpu",
                "devices": {"gpu": dict(gpu_before)},
            },
        },
    })
    _seed_cache(tmp_path, monkeypatch, _SMOL, _GPU_14B)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: type("S", (), {"base_url": "http://127.0.0.1:18436/v1"})())
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None, device=None: "http://127.0.0.1:18436/v1")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bench.verify_tool_calls",
        lambda *a, **k: {"ok": True})
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_device", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.hardware._ram_stats",
        lambda: (128 * _GIB, 16 * _GIB, 96 * _GIB),
    )

    used = client.post("/api/local-models/vllm/use", json={"model": _SMOL})
    assert used.status_code == 200, used.text
    from hermes_cli.config import load_config

    cfg = load_config()
    devices = _devices(cfg)
    assert (devices.get("cpu") or {}).get("model") == _SMOL
    assert (devices.get("gpu") or {}).get("model") == gpu_before["model"]
    assert (devices.get("gpu") or {}).get("served_model_name") == gpu_before["served_model_name"]
    assert int((devices.get("gpu") or {}).get("max_model_len")) == gpu_before["max_model_len"]
    assert vllm_settings(cfg, device="cpu")["model"] == _SMOL
    assert vllm_settings(cfg, device="gpu")["model"] == _GPU_14B


def test_gpu_start_after_cpu_smol_serves_gpu_pick_not_smol(tmp_path, monkeypatch):
    """GPU start serves devices.gpu (14B/AWQ recommend), never the CPU Smol."""
    client, home = _client(tmp_path, monkeypatch)
    rec = recommend_vllm(total_bytes=24 * _GIB)
    gpu_pick = as_vllm_config(rec)
    assert "awq" in gpu_pick["model"].lower() or gpu_pick["quantization"] == "awq"
    _write_cfg(home, {
        "model": {},
        "local_runtime": {
            "enabled": True,
            "engine": "vllm",
            "vllm": {
                "selected": "cpu",
                "device": "cpu",
                "devices": {
                    "gpu": dict(gpu_pick),
                    "cpu": {
                        "model": _SMOL,
                        "served_model_name": "SmolLM3-3B",
                        "max_model_len": 65536,
                    },
                },
            },
        },
    })
    _seed_cache(tmp_path, monkeypatch, _SMOL, gpu_pick["model"])
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.recommend.recommend_vllm",
        lambda **k: rec)
    started: list[str] = []

    class _Sup:
        base_url = "http://127.0.0.1:18435/v1"

    def _ensure(cfg=None, **_k):
        from hermes_cli.vllm_runtime.supervisor import vllm_settings as settings_of

        hid = settings_of(cfg, device="gpu")["model"]
        started.append(hid)
        assert hid != _SMOL
        assert hid == gpu_pick["model"]
        return _Sup()

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime", _ensure)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None, device=None: "http://127.0.0.1:18435/v1")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bench.verify_tool_calls",
        lambda *a, **k: {"ok": True})
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_device", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)

    switched = client.post("/api/local-models/vllm/device", json={"device": "gpu"})
    assert switched.status_code == 200, switched.text
    from hermes_cli.config import load_config

    after_switch = load_config()
    assert (after_switch["local_runtime"]["vllm"].get("selected")
            or after_switch["local_runtime"]["vllm"].get("device")) == "gpu"
    assert _devices(after_switch)["cpu"]["model"] == _SMOL
    assert _devices(after_switch)["gpu"]["model"] == gpu_pick["model"]

    started_ok = client.post("/api/local-models/server", json={"action": "start"})
    assert started_ok.status_code == 200, started_ok.text
    assert started == [gpu_pick["model"]]
    assert _SMOL not in started


def test_migrate_shared_model_onto_selected_device_only():
    """Old shared vllm.model attaches to selected only — never both devices."""
    smol = _SMOL
    cpu_only = migrate_vllm_devices({
        "device": "cpu",
        "model": smol,
        "served_model_name": "SmolLM3-3B",
        "max_model_len": 65536,
    }, selected="cpu")
    assert cpu_only["cpu"]["model"] == smol
    assert cpu_only["cpu"]["served_model_name"] == "SmolLM3-3B"
    assert (cpu_only.get("gpu") or {}).get("model") != smol

    gpu_live = migrate_vllm_devices({
        "device": "gpu",
        "model": smol,
        "served_model_name": "SmolLM3-3B",
    }, selected="gpu")
    assert gpu_live["gpu"]["model"] == smol
    assert (gpu_live.get("cpu") or {}).get("model") != smol

    split = migrate_vllm_devices({
        "model": _GPU_14B,
        "served_model_name": "qwen3:14b",
        "cpu": {"model": smol, "served_model_name": "SmolLM3-3B"},
    }, selected="gpu")
    assert split["cpu"]["model"] == smol
    assert split["gpu"]["model"] == _GPU_14B

    on_cpu = vllm_settings({
        "local_runtime": {
            "engine": "vllm",
            "vllm": {"device": "cpu", "model": smol, "served_model_name": "SmolLM3-3B"},
        },
    })
    assert on_cpu["model"] == smol
    gpu_view = vllm_settings({
        "local_runtime": {
            "engine": "vllm",
            "vllm": {"device": "cpu", "model": smol, "served_model_name": "SmolLM3-3B"},
        },
    }, device="gpu")
    assert gpu_view["model"] != smol


def test_cpu_activate_refreshes_stale_provider_url(tmp_path, monkeypatch):
    """Starting CPU rewrites providers.vllm-cpu (18436), not leftover 42477."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)
    _write_cfg(home, {
        "model": {"provider": "custom", "base_url": "http://127.0.0.1:18436/v1"},
        "providers": {
            "vllm-cpu": {
                "name": "vLLM CPU",
                "base_url": "http://127.0.0.1:42477/v1",
                "model": "qwen3:4b",
            },
        },
        "local_runtime": {
            "enabled": True,
            "engine": "vllm",
            "vllm": {"selected": "cpu", "device": "cpu"},
        },
    })

    class _Sup:
        base_url = "http://127.0.0.1:18436/v1"

    import hermes_cli.vllm_runtime.bootstrap as boot

    monkeypatch.setattr(boot, "_SUPERVISORS", {"gpu": None, "cpu": _Sup()})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint._state_endpoint",
        lambda device, config=None: None)
    from hermes_cli.config import load_config

    url = boot.activate_vllm_provider(load_config())
    assert url.rstrip("/") == "http://127.0.0.1:18436/v1"
    after = load_config()["providers"]["vllm-cpu"]["base_url"]
    assert after.rstrip("/") == "http://127.0.0.1:18436/v1"
    assert "42477" not in after


def test_migrate_save_persists_devices_and_cpu_url_follows_server_json(
        tmp_path, monkeypatch):
    """migrate+save keeps both devices, drops leftover nest/shared keys,
    and CPU start rewrites providers.vllm-cpu from that device's server.json.
    GPU start must not write the CPU URL.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)

    cpu_live = "http://127.0.0.1:52269/v1"
    gpu_live = "http://127.0.0.1:18435/v1"
    stale_cpu = "http://127.0.0.1:18436/v1"
    _write_cfg(home, {
        "model": {
            "provider": "custom",
            "base_url": gpu_live,
            "default": "qwen3:14b",
        },
        "providers": {
            "vllm": {"name": "vLLM CPU", "base_url": "", "enabled": False},
            "vllm-gpu": {
                "name": "vLLM GPU",
                "base_url": gpu_live,
                "model": "qwen3:14b",
            },
            "vllm-cpu": {
                "name": "vLLM CPU",
                "base_url": stale_cpu,
                "model": "SmolLM3-3B",
            },
        },
        "local_runtime": {
            "enabled": True,
            "engine": "vllm",
            "vllm": {
                "selected": "gpu",
                "device": "gpu",
                "devices": {
                    "gpu": {
                        "model": _GPU_14B,
                        "served_model_name": "qwen3:14b",
                        "max_model_len": 65536,
                    },
                },
                "model": _SMOL,
                "served_model_name": "SmolLM3-3B",
                "max_model_len": 65536,
                "cpu": {
                    "model": _SMOL,
                    "served_model_name": "SmolLM3-3B",
                    "max_model_len": 65536,
                },
            },
        },
    })
    from hermes_cli.vllm_runtime.supervisor import state_path

    for device, port, served in (
        ("gpu", 18435, "qwen3:14b"),
        ("cpu", 52269, "SmolLM3-3B"),
    ):
        path = state_path(device)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "base_url": f"http://127.0.0.1:{port}/v1",
            "pid": 1,
            "served_model_name": served,
        }), encoding="utf-8")

    from hermes_cli.config import load_config, read_raw_config
    import hermes_cli.vllm_runtime.bootstrap as boot

    monkeypatch.setattr(boot, "_SUPERVISORS", {"gpu": None, "cpu": None})
    assert persist_migrated_vllm() is True
    raw = read_raw_config()
    vllm = (raw.get("local_runtime") or {}).get("vllm") or {}
    devices = vllm.get("devices") or {}
    assert (devices.get("gpu") or {}).get("model") == _GPU_14B
    assert (devices.get("cpu") or {}).get("model") == _SMOL
    assert int((devices.get("cpu") or {}).get("max_model_len") or 0) == 65536
    assert "model" not in vllm
    assert "served_model_name" not in vllm
    assert "cpu" not in vllm
    assert persist_migrated_vllm() is False

    boot.activate_vllm_provider(load_config(), device="cpu")
    after_cpu = load_config()
    assert after_cpu["providers"]["vllm-cpu"]["base_url"].rstrip("/") == cpu_live
    assert after_cpu["providers"]["vllm-cpu"]["model"] == "SmolLM3-3B"
    assert after_cpu["providers"]["vllm-gpu"]["base_url"].rstrip("/") == gpu_live
    assert after_cpu["model"]["base_url"].rstrip("/") == gpu_live
    assert after_cpu["model"]["default"] == "qwen3:14b"
    leftover = read_raw_config()["local_runtime"]["vllm"]
    assert "model" not in leftover
    assert "cpu" not in leftover

    boot.activate_vllm_provider(load_config(), device="gpu")
    after_gpu = load_config()
    assert after_gpu["providers"]["vllm-cpu"]["base_url"].rstrip("/") == cpu_live
    assert after_gpu["providers"]["vllm-gpu"]["base_url"].rstrip("/") == gpu_live
    assert "18436" not in after_gpu["providers"]["vllm-cpu"]["base_url"]
    assert after_gpu["model"]["base_url"].rstrip("/") == gpu_live
