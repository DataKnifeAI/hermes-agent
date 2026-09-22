"""CPU vLLM is a second isolated engine — not ``--device cpu`` on the CUDA wheel."""

from __future__ import annotations

import json
from pathlib import Path

from hermes_cli.local_engines import engine_from_config
from hermes_cli.vllm_runtime.device import (
    CPU_LISTEN_PORT, ENGINE_CPU, ENGINE_GPU, GPU_LISTEN_PORT, USER_SERVER_PORTS,
)
from hermes_cli.vllm_runtime.supervisor import serve_argv
from hermes_cli.vllm_runtime.venv import runtimes_root, venv_dir


def test_cpu_and_gpu_venvs_are_different_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)

    gpu = venv_dir("gpu")
    cpu = venv_dir("cpu")
    assert gpu != cpu
    assert gpu == tmp_path / "runtimes" / "vllm" / ".venv"
    assert cpu == tmp_path / "runtimes" / "vllm-cpu" / ".venv"
    assert runtimes_root("gpu") != runtimes_root("cpu")


def test_engine_from_config_is_llamacpp_or_vllm():
    """One vLLM page. Legacy ``vllm-cpu`` folds to engine vllm + device cpu."""
    from hermes_cli.local_engines import vllm_device_from_config
    from hermes_cli.vllm_runtime.device import CPU, PUBLIC_ENGINES

    assert PUBLIC_ENGINES == frozenset({"llamacpp", ENGINE_GPU})
    assert ENGINE_CPU not in PUBLIC_ENGINES
    assert engine_from_config({}) == "llamacpp"
    assert engine_from_config({"local_runtime": {"engine": "vllm"}}) == ENGINE_GPU
    assert engine_from_config({"local_runtime": {"engine": "vllm-cpu"}}) == ENGINE_GPU
    assert engine_from_config({"local_runtime": {"engine": "VLLM_CPU"}}) == ENGINE_GPU
    assert vllm_device_from_config({"local_runtime": {"engine": "vllm-cpu"}}) == CPU
    assert vllm_device_from_config({
        "local_runtime": {"engine": "vllm", "vllm": {"device": "cpu"}},
    }) == CPU


def test_cpu_serve_argv_uses_cpu_port_and_omits_cuda_flags():
    argv = serve_argv("/opt/cpu-venv/bin/vllm", {
        "model": "Qwen/Qwen3-8B-AWQ",
        "port": 0,
        "gpu_memory_utilization": 0.75,
        "kv_cache_dtype": "fp8",
    }, device="cpu")
    assert "--device" not in argv
    assert argv[argv.index("--port") + 1] == str(CPU_LISTEN_PORT)
    assert "--gpu-memory-utilization" not in argv
    assert "--kv-cache-dtype" not in argv
    assert str(GPU_LISTEN_PORT) not in argv
    for banned in USER_SERVER_PORTS:
        assert str(banned) not in argv

    gpu = serve_argv("/opt/gpu-venv/bin/vllm", {
        "model": "Qwen/Qwen3-8B-AWQ",
        "port": 0,
        "gpu_memory_utilization": 0.75,
        "kv_cache_dtype": "fp8",
    }, device="gpu")
    assert gpu[gpu.index("--port") + 1] == str(GPU_LISTEN_PORT)
    assert "--gpu-memory-utilization" in gpu


def test_cpu_start_skips_gpu_occupancy(tmp_path, monkeypatch):
    """Starting CPU vLLM must not require a free GPU."""
    from hermes_cli.web_routers import local_models_engine as engine

    order: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.occupancy.require_gpu_free",
        lambda: order.append("occupancy"))
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: order.append("stop_llama"))
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.configured_unservable_reason",
        lambda settings: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.configured_cache_missing",
        lambda settings: False)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.state_served_model_name",
        lambda device="gpu": "")

    class _Sup:
        base_url = "http://127.0.0.1:18436/v1"

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: order.append("start") or _Sup())
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:18436/v1")

    cfg = {"local_runtime": {"engine": "vllm-cpu", "vllm": {"model": "Qwen/Qwen3-8B-AWQ"}}}
    engine._start_configured_vllm(cfg, cfg["local_runtime"]["vllm"])
    assert "occupancy" not in order
    assert "stop_llama" not in order
    assert "start" in order


def test_gpu_start_still_occupancy_checks(tmp_path, monkeypatch):
    from hermes_cli.web_routers import local_models_engine as engine

    order: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.occupancy.require_gpu_free",
        lambda: order.append("occupancy"))
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.configured_unservable_reason",
        lambda settings: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.configured_cache_missing",
        lambda settings: False)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.state_served_model_name",
        lambda device="gpu": "")

    class _Sup:
        base_url = "http://127.0.0.1:18435/v1"

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: order.append("start") or _Sup())
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:18435/v1")

    cfg = {"local_runtime": {"engine": "vllm", "vllm": {"model": "Qwen/Qwen3-8B-AWQ"}}}
    engine._start_configured_vllm(cfg, cfg["local_runtime"]["vllm"])
    assert order == ["occupancy", "start"]


def test_hardware_payload_includes_cpu_fields(tmp_path, monkeypatch):
    from hermes_cli.local_runtime.hardware import parse_cpuinfo_model, probe_cpu

    assert parse_cpuinfo_model(
        "processor\t: 0\nmodel name\t: AMD Ryzen 9 7950X 16-Core Processor\n"
    ) == "AMD Ryzen 9 7950X 16-Core Processor"
    facts = probe_cpu()
    assert "cpu_name" in facts and "cpu_cores" in facts
    if facts["cpu_cores"] is not None:
        assert facts["cpu_cores"] >= 1


def test_set_engine_accepts_vllm_cpu(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    from hermes_cli import web_server

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    r = client.post("/api/local-models/engine", json={"engine": "vllm-cpu"})
    assert r.status_code == 200
    assert r.json()["engine"] == "vllm"
    assert r.json()["vllm_device"] == "cpu"
    from hermes_cli.config import load_config

    saved = load_config()["local_runtime"]
    assert saved["engine"] == "vllm"
    assert saved["vllm"]["device"] == "cpu"
    switched = client.post("/api/local-models/vllm/device", json={"device": "gpu"})
    assert switched.status_code == 200
    assert switched.json()["vllm_device"] == "gpu"
    assert load_config()["local_runtime"]["vllm"]["device"] == "gpu"
    assert load_config()["local_runtime"]["engine"] == "vllm"


def test_cpu_catalog_keeps_official_qwen_when_gpu_is_24gb(monkeypatch):
    """CPU fit is RAM, not the 24 GB VRAM hide that drops official 32B Qwen."""
    from hermes_cli.vllm_runtime.inventory import catalog_models, hide_catalog_row_by_default
    from hermes_cli.vllm_runtime.recommend import (
        NvidiaProbe, VllmRecommendation, catalog_tiers,
    )

    _GIB = 1 << 30
    pick = next(t for t in catalog_tiers() if t.id == "24gb")
    rec = VllmRecommendation(NvidiaProbe(24 * _GIB, 24 * _GIB, "data"), pick, True, "ok")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory.list_cached_repos", lambda: [])
    monkeypatch.setattr("hermes_cli.vllm_runtime.supervisor.vllm_settings", lambda cfg=None: {})
    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory.running_served_model_name", lambda: "")
    monkeypatch.setattr(
        "hermes_cli.local_runtime.hardware._ram_stats",
        lambda: (128 * _GIB, 32 * _GIB, 96 * _GIB),
    )
    rows = catalog_models({"local_runtime": {"engine": "vllm-cpu"}})
    by_id = {r["id"]: r for r in rows}
    forty = by_id["Qwen/Qwen3-32B-AWQ"]
    assert forty["fit"] != "fits-gpu"
    assert forty["hide_by_default"] is False
    assert hide_catalog_row_by_default(forty) is False


def test_cpu_default_is_bf16_qwen3_4b_not_gpu_awq(tmp_path, monkeypatch):
    """CPU recommend / start / catalog use a small BF16 checkpoint.

    Qwen3-4B-Instruct-2507 is BF16, hermes ``<tool_call>`` XML, native 262144.
    The GPU shipped id stays Qwen3-8B-AWQ. CUDA AWQ/FP8 is not the CPU default.
    """
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.vllm_runtime.inventory import catalog_models
    from hermes_cli.vllm_runtime.recommend import (
        MIN_CONTEXT, as_vllm_config, gpu_shipped_model, recommend_vllm,
        recommend_vllm_cpu,
    )
    from hermes_cli.vllm_runtime.supervisor import serve_argv, vllm_settings
    from hermes_cli.web_routers.local_models_engine import recommend_payload

    cpu = recommend_vllm_cpu()
    overlay = as_vllm_config(cpu)
    gpu_shipped = gpu_shipped_model()
    assert overlay["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert overlay["model"] != gpu_shipped
    assert "awq" not in overlay["model"].lower()
    assert "fp8" not in overlay["model"].lower()
    assert "gguf" not in overlay["model"].lower()
    assert overlay["quantization"] == ""
    assert overlay["kv_cache_dtype"] == ""
    assert overlay["tool_call_parser"] == "hermes"
    assert overlay["max_model_len"] >= MIN_CONTEXT
    assert cpu.feasible is True

    assert DEFAULT_CONFIG["local_runtime"]["vllm"]["model"] == gpu_shipped
    assert recommend_vllm(total_bytes=0).model == gpu_shipped
    gpu_settings = vllm_settings({"local_runtime": {"engine": "vllm"}})
    assert gpu_settings["model"] == gpu_shipped
    assert gpu_settings["quantization"] == "awq"

    fresh = vllm_settings({"local_runtime": {"engine": "vllm-cpu"}})
    assert fresh["model"] == overlay["model"]
    assert fresh["quantization"] == ""
    assert fresh["kv_cache_dtype"] == ""
    assert fresh["tool_call_parser"] == "hermes"
    shipped_slot = vllm_settings({
        "local_runtime": {
            "engine": "vllm-cpu",
            "vllm": {"model": gpu_shipped, "quantization": "awq", "kv_cache_dtype": "fp8"},
        },
    })
    assert shipped_slot["model"] == overlay["model"]
    assert shipped_slot["quantization"] == ""
    explicit = vllm_settings({
        "local_runtime": {
            "engine": "vllm-cpu",
            "vllm": {"model": "org/custom-bf16"},
        },
    })
    assert explicit["model"] == "org/custom-bf16"

    hid = overlay["model"]
    hub = tmp_path / "hf-hub"
    root = hub / ("models--" + hid.replace("/", "--"))
    snap = root / "snapshots" / "main"
    snap.mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("main", encoding="utf-8")
    (snap / "config.json").write_text(json.dumps({
        "max_position_embeddings": 262144,
        "torch_dtype": "bfloat16",
        "rope_scaling": None,
    }), encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    argv = serve_argv("/opt/cpu-venv/bin/vllm", fresh, device="cpu")
    assert hid in argv
    assert "--quantization" not in argv
    assert "--kv-cache-dtype" not in argv
    assert "--gpu-memory-utilization" not in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"
    served_len = int(argv[argv.index("--max-model-len") + 1])
    assert served_len >= MIN_CONTEXT
    assert served_len == MIN_CONTEXT

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"local_runtime": {"engine": "vllm-cpu"}},
    )
    body = recommend_payload()
    assert body["model"] == hid
    assert body["quantization"] == ""
    assert body["kv_cache_dtype"] == ""
    assert body["tool_call_parser"] == "hermes"
    assert body["feasible"] is True
    assert body["max_model_len"] >= MIN_CONTEXT

    _GIB = 1 << 30
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: cpu)
    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory.list_cached_repos", lambda: [])
    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory.running_served_model_name", lambda: "")
    monkeypatch.setattr(
        "hermes_cli.local_runtime.hardware._ram_stats",
        lambda: (128 * _GIB, 16 * _GIB, 96 * _GIB),
    )
    rows = catalog_models({"local_runtime": {"engine": "vllm-cpu"}})
    recommended = [r for r in rows if r.get("recommended")]
    assert [r["id"] for r in recommended] == [hid]
    assert recommended[0]["hide_by_default"] is False
    assert "awq" not in recommended[0]["quantization"]
    assert "fp8" not in recommended[0]["quantization"]


def test_cpu_setup_plans_bf16_not_gpu_awq(monkeypatch):
    """Set up for me on vllm-cpu downloads the CPU default, not Qwen3-8B-AWQ."""
    from hermes_cli.web_routers.local_models_engine import vllm_quickstart_plan

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"local_runtime": {"engine": "vllm-cpu"}},
    )
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.venv_ready", lambda *a, **k: False)
    plan = vllm_quickstart_plan()
    assert plan["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert "awq" not in plan["model"].lower()
    assert "fp8" not in plan["model"].lower()
    assert plan["apply_recommend"] is True

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "local_runtime": {
                "engine": "vllm",
                "vllm": {
                    "device": "cpu",
                    "model": "Qwen/Qwen3-14B-AWQ",
                    "quantization": "awq",
                },
            }
        },
    )
    on_device = vllm_quickstart_plan()
    assert on_device["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert on_device["display_name"] == "Qwen/Qwen3-4B-Instruct-2507"


def test_cpu_device_quickstart_downloads_4b_not_gpu_awq(monkeypatch):
    """Set up for me on device cpu downloads the BF16 default, not 14B AWQ."""
    from hermes_cli.web_routers.local_models_engine import (
        run_vllm_quickstart, vllm_quickstart_plan,
    )

    cfg = {
        "local_runtime": {
            "engine": "vllm",
            "vllm": {
                "device": "cpu",
                "model": "Qwen/Qwen3-14B-AWQ",
                "served_model_name": "qwen3:14b",
                "quantization": "awq",
                "python": "",
            },
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    def _save(key, value):
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    monkeypatch.setattr("cli.save_config_value", _save)
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.venv_ready", lambda *a, **k: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.ensure_both_vllm_venvs",
        lambda *a, **k: {"gpu": Path("/tmp/vllm-gpu"), "cpu": Path("/tmp/vllm-cpu")},
    )
    downloaded: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.ensure_hf_weights",
        lambda hid, job=None: downloaded.append(hid),
    )
    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory.repo_is_cached", lambda hid: False)
    occupied: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.occupancy.require_gpu_free",
        lambda: occupied.append("gpu"),
    )
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_device", lambda device: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.supervisor.disable_auto_start", lambda: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.supervisor.clear_last_error", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.supervisor.read_last_error", lambda *a, **k: "")
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.start_active_engine",
        lambda **k: None,
    )
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.activate_vllm",
        lambda: {"ok": True},
    )

    plan = vllm_quickstart_plan()
    job: dict = {}
    run_vllm_quickstart(job, plan)
    assert plan["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert downloaded == ["Qwen/Qwen3-4B-Instruct-2507"]
    assert occupied == []
    assert cfg["local_runtime"]["vllm"]["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert "AWQ" not in job["detail"]
    assert "14B" not in job["detail"]
    assert "CUDA" not in job["detail"]
    assert "Qwen/Qwen3-4B-Instruct-2507" in job["detail"]


def test_gpu_setup_stays_official_awq(monkeypatch):
    """GPU Set up for me stays on the VRAM-fit Qwen AWQ, not the CPU BF16."""
    from hermes_cli.vllm_runtime.recommend import (
        NvidiaProbe, VllmRecommendation, catalog_tiers,
    )
    from hermes_cli.web_routers.local_models_engine import vllm_quickstart_plan

    tier = next(t for t in catalog_tiers() if t.id == "24gb")
    rec = VllmRecommendation(
        NvidiaProbe(24 * (1 << 30), 24 * (1 << 30), "data"), tier, True, "ok")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.hf_repo_access_issue", lambda hid: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.venv_ready", lambda *a, **k: False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "local_runtime": {
                "engine": "vllm",
                "vllm": {"device": "gpu", "model": "Qwen/Qwen3-4B-Instruct-2507"},
            }
        },
    )
    plan = vllm_quickstart_plan()
    assert plan["model"] == "Qwen/Qwen3-14B-AWQ"
    assert plan["display_name"] == "Qwen/Qwen3-14B-AWQ"
    assert plan["model"] != "Qwen/Qwen3-4B-Instruct-2507"


def test_ensure_both_venvs_calls_gpu_and_cpu(monkeypatch):
    from hermes_cli.vllm_runtime import venv as venv_mod

    seen: list[str] = []

    def _fake(pin="", *, upgrade=False, version=None, device="gpu"):
        seen.append(device)
        return Path(f"/tmp/vllm-{device}")

    monkeypatch.setattr(venv_mod, "ensure_vllm_venv", _fake)
    out = venv_mod.ensure_both_vllm_venvs("")
    assert seen == ["gpu", "cpu"]
    assert set(out) == {"gpu", "cpu"}
