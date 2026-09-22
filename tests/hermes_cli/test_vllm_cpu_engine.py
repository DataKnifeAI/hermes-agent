"""CPU vLLM is a second isolated engine — not ``--device cpu`` on the CUDA wheel."""

from __future__ import annotations

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


def test_engine_from_config_distinguishes_cpu_and_gpu():
    assert engine_from_config({}) == "llamacpp"
    assert engine_from_config({"local_runtime": {"engine": "vllm"}}) == ENGINE_GPU
    assert engine_from_config({"local_runtime": {"engine": "vllm-cpu"}}) == ENGINE_CPU
    assert engine_from_config({"local_runtime": {"engine": "VLLM_CPU"}}) == ENGINE_CPU


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
    assert r.json()["engine"] == "vllm-cpu"
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["engine"] == "vllm-cpu"


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
