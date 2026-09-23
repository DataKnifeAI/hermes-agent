"""CPU vLLM is a second isolated engine — not ``--device cpu`` on the CUDA wheel."""

from __future__ import annotations

import json
import math
from pathlib import Path

import os

import pytest

from hermes_cli.local_engines import engine_from_config
from hermes_cli.vllm_runtime.device import (
    CPU_LISTEN_PORT, ENGINE_CPU, ENGINE_GPU, GPU_LISTEN_PORT, USER_SERVER_PORTS,
)
from hermes_cli.vllm_runtime.supervisor import (
    fit_cpu_memory_utilization, format_memory_utilization, parse_vllm_node_meminfo,
    serve_argv, serve_environ,
)
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
    # CPU still passes the flag: vLLM uses it as a fraction of node RAM.
    cpu_util = float(argv[argv.index("--gpu-memory-utilization") + 1])
    assert 0 < cpu_util <= 0.75
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
    assert gpu[gpu.index("--gpu-memory-utilization") + 1] == "0.75"


def test_parse_vllm_node_meminfo_matches_worker_formula():
    text = "\n".join([
        "Node 0 MemTotal:       65766604 kB",
        "Node 0 MemFree:         2097152 kB",
        "Node 0 MemAvailable:   35861299 kB",
        "Node 0 Active(file):   10485760 kB",
        "Node 0 Inactive(file): 20971520 kB",
        "Node 0 SReclaimable:    5242880 kB",
    ])
    total, available = parse_vllm_node_meminfo(text)
    assert total == 65766604 * 1024
    # MemAvailable is not the worker's number.
    assert available == (2097152 + 10485760 + 20971520 + 5242880) * 1024


def test_cpu_memory_utilization_fits_busy_numa_node():
    """0.92 of node total wanted 57.68 GiB; node 0 had 34.2 of 62.7 free."""
    gib = 1024 ** 3
    total = int(62.7 * gib)
    available = int(34.2 * gib)
    assert math.ceil(total * 0.92) > available
    assert math.ceil(total * 0.75) > available
    util = fit_cpu_memory_utilization(total, available, 0.75)
    assert math.ceil(total * float(format_memory_utilization(util))) <= available
    assert util < 0.75


def test_cpu_memory_utilization_keeps_configured_cap_when_node_is_free():
    gib = 1024 ** 3
    util = fit_cpu_memory_utilization(64 * gib, 60 * gib, 0.75)
    assert util == 0.75
    assert format_memory_utilization(util) == "0.75"


def test_cpu_serve_argv_shrinks_reservation_to_free_node_ram(monkeypatch):
    gib = 1024 ** 3
    total = int(62.7 * gib)
    available = int(34.2 * gib)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.cpu_rank0_node_memory",
        lambda: (total, available),
    )
    argv = serve_argv("/opt/cpu-venv/bin/vllm", {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "port": 18436,
        "gpu_memory_utilization": 0.75,
    }, device="cpu")
    flag = argv[argv.index("--gpu-memory-utilization") + 1]
    assert math.ceil(total * float(flag)) <= available
    assert float(flag) < 0.75


def test_cpu_serve_env_selects_cpu_platform_before_import(tmp_path, monkeypatch):
    """CPU serve must not load platforms/cuda.py (that needs libtorch_cuda.so)."""
    monkeypatch.delenv("VLLM_TARGET_DEVICE", raising=False)
    root = tmp_path / "cpu-venv"
    lib = root / "lib" / "python3.12" / "site-packages" / "intel_openmp" / "libiomp5.so"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"")
    exe = root / "bin" / "vllm"
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8")

    env = serve_environ(exe, device="cpu")
    assert env["VLLM_TARGET_DEVICE"] == "cpu"
    assert env["LD_PRELOAD"].startswith(str(lib))
    assert "libtorch_cuda" not in env.get("LD_PRELOAD", "")

    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
    gpu_env = serve_environ(root / "bin" / "vllm", device="gpu")
    assert gpu_env.get("VLLM_TARGET_DEVICE") != "cpu"
    assert "VLLM_TARGET_DEVICE" not in gpu_env


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

    cfg = {"local_runtime": {"engine": "vllm-cpu", "vllm": {"model": "Qwen/Qwen3-4B-Instruct-2507"}}}
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


def test_cpu_and_gpu_catalogs_list_different_checkpoints(monkeypatch):
    """CPU omits CUDA quants; GPU still lists official AWQ that fits.

    Show (``visible_catalog_models(..., show_unfitting=True)``) reveals a
    too-big compatible row. It does not reveal an AWQ id the CPU wheel
    cannot start. CPU fit for the BF16 default is RAM (``needs-ram``),
    not the GPU ``fits-gpu`` badge.
    """
    from hermes_cli.vllm_runtime.inventory import catalog_models, visible_catalog_models
    from hermes_cli.vllm_runtime.recommend import (
        NvidiaProbe, VllmRecommendation, catalog_tiers,
    )

    _GIB = 1 << 30
    bf16 = "Qwen/Qwen3-4B-Instruct-2507"
    awq = "Qwen/Qwen3-8B-AWQ"
    wide_awq = "Qwen/Qwen3-32B-AWQ"
    fp8 = "Qwen/Qwen3.8-27B-FP8"
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
    cpu_rows = catalog_models({"local_runtime": {"engine": "vllm-cpu"}})
    cpu_ids = {r["id"] for r in cpu_rows}
    assert bf16 in cpu_ids
    assert awq not in cpu_ids
    assert wide_awq not in cpu_ids
    assert fp8 not in cpu_ids
    four = next(r for r in cpu_rows if r["id"] == bf16)
    assert four["fit"] == "needs-ram"
    assert four["fit"] != "fits-gpu"
    assert four["hide_by_default"] is False
    shown = visible_catalog_models(cpu_rows, show_unfitting=True)
    assert awq not in {r["id"] for r in shown}

    gpu_rows = catalog_models({
        "local_runtime": {"engine": "vllm", "vllm": {"device": "gpu"}},
    })
    gpu_by_id = {r["id"]: r for r in gpu_rows}
    assert gpu_by_id[awq]["fit"] == "fits-gpu"
    assert gpu_by_id[awq]["hide_by_default"] is False
    assert bf16 not in gpu_by_id


def test_cpu_in_use_stays_visible_when_own_serve_holds_ram(monkeypatch):
    """Free RAM after our CPU serve starts must not hide the model in use.

    Fit credits that process's RSS back, so the running checkpoint does not
    flip to Too big. The same checkpoint still hides when the server is
    stopped and free RAM is below the fit budget. A row that does not fit
    the restored budget stays hidden. GPU too-big rows are a different probe.
    """
    from hermes_cli.vllm_runtime.inventory import (
        catalog_models, classify_vllm_repo, hide_catalog_row_by_default,
        visible_catalog_models,
    )
    from hermes_cli.vllm_runtime.recommend import NvidiaProbe, VllmRecommendation, catalog_tiers

    hid = "Qwen/Qwen3-4B-Instruct-2507"
    bigger = "Qwen/Qwen3-32B-AWQ"
    _gib = 1 << 30
    tight = None
    roomy = None
    for n in range(1, 96):
        ram = n * _gib
        four = classify_vllm_repo(hid, device="cpu", total_ram=ram)["fit"]
        wide = classify_vllm_repo(bigger, device="cpu", total_ram=ram)["fit"]
        if four == "too-big":
            tight = ram
        elif tight is not None and four == "needs-ram" and wide == "too-big":
            roomy = ram
            break
    assert tight is not None and roomy is not None and tight < roomy

    pick = next(t for t in catalog_tiers() if t.id == "24gb")
    rec = VllmRecommendation(NvidiaProbe(24 * _gib, 24 * _gib, "data"), pick, True, "ok")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory.list_cached_repos", lambda: [])
    monkeypatch.setattr("hermes_cli.vllm_runtime.supervisor.vllm_settings", lambda cfg=None: {})
    held = {"bytes": 0}
    served = {"name": ""}
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.occupancy.managed_device_rss_bytes",
        lambda device="cpu": held["bytes"] if device == "cpu" else 0,
    )
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.running_served_model_name",
        lambda: served["name"],
    )
    total = roomy + (8 * _gib)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.hardware._ram_stats",
        lambda: (total, total - tight, tight),
    )
    cfg = {"local_runtime": {"engine": "vllm-cpu"}}

    stopped = {r["id"]: r for r in catalog_models(cfg)}
    assert stopped[hid]["fit"] == "too-big"
    assert stopped[hid]["hide_by_default"] is True
    assert stopped[hid]["active"] is False
    assert hid not in {r["id"] for r in visible_catalog_models(list(stopped.values()))}

    held["bytes"] = roomy - tight
    served["name"] = hid
    running = {r["id"]: r for r in catalog_models(cfg)}
    assert running[hid]["active"] is True
    assert running[hid]["fit"] == "needs-ram"
    assert running[hid]["hide_by_default"] is False
    assert hide_catalog_row_by_default(running[hid]) is False
    visible = {r["id"] for r in visible_catalog_models(list(running.values()))}
    assert hid in visible
    # CUDA quant is absent, not parked behind Show. The served AWQ stays.
    assert bigger not in running
    assert bigger not in {
        r["id"] for r in visible_catalog_models(list(running.values()), show_unfitting=True)
    }
    served["name"] = "qwen3:8b"
    live_awq = {r["id"]: r for r in catalog_models(cfg)}
    assert live_awq["Qwen/Qwen3-8B-AWQ"]["active"] is True
    assert "Qwen/Qwen3-14B-AWQ" not in live_awq
    assert hid in live_awq
    assert hide_catalog_row_by_default(
        {"active": True, "added_by_you": False, "fit": "too-big", "fits": False}
    ) is False


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
    # engine: vllm + device: cpu is the live Local Models path. An explicit
    # BF16 pick must not be overwritten by the Qwen 4B CPU default.
    smol = "HuggingFaceTB/SmolLM3-3B"
    on_device = vllm_settings({
        "local_runtime": {
            "engine": "vllm",
            "vllm": {
                "device": "cpu",
                "model": smol,
                "served_model_name": "SmolLM3-3B",
                "max_model_len": 65536,
            },
        },
    })
    assert on_device["model"] == smol
    assert on_device["served_model_name"] == "SmolLM3-3B"
    assert on_device["model"] != overlay["model"]
    assert int(on_device["max_model_len"]) == 65536

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
    cpu_util = float(argv[argv.index("--gpu-memory-utilization") + 1])
    assert 0 < cpu_util <= float(fresh.get("gpu_memory_utilization") or 0.75)
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
    cpu = ((cfg["local_runtime"]["vllm"].get("devices") or {}).get("cpu") or {})
    assert cpu.get("model") == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg["local_runtime"]["vllm"]["model"] == "Qwen/Qwen3-14B-AWQ"
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


def _isolate_runtime(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    return home


def _bin_dir(dest: Path) -> Path:
    return dest / ("Scripts" if os.name == "nt" else "bin")


def _touch_venv_bins(dest: Path) -> None:
    bin_dir = _bin_dir(dest)
    bin_dir.mkdir(parents=True, exist_ok=True)
    py_name = "python.exe" if os.name == "nt" else "python"
    (bin_dir / py_name).write_text("keep", encoding="utf-8")
    exe = "vllm.exe" if os.name == "nt" else "vllm"
    (bin_dir / exe).write_text("", encoding="utf-8")
    ninja = "ninja.exe" if os.name == "nt" else "ninja"
    (bin_dir / ninja).write_text("", encoding="utf-8")


def test_cpu_install_uses_cpu_wheel_index_not_pypi_cuda(tmp_path, monkeypatch):
    """A CPU venv that already has the CUDA ``vllm`` wheel must be replaced.

    ``--torch-backend=cpu`` only selects the torch wheel. The vLLM package has
    to come from ``wheels.vllm.ai/<version>/cpu`` as ``vllm==<version>+cpu``.
    """
    _isolate_runtime(tmp_path, monkeypatch)
    from hermes_cli.vllm_runtime import venv as venv_mod

    dest = venv_mod.venv_dir("cpu")
    _touch_venv_bins(dest)
    calls: list[tuple[list[str], dict | None]] = []

    def _fake_stream(cmd, log_path, cwd=None, env=None):
        calls.append((list(cmd), env))

    monkeypatch.setattr(venv_mod, "_stream", _fake_stream)
    monkeypatch.setattr(venv_mod, "_assert_cpu", lambda *a, **k: None)
    monkeypatch.setattr(venv_mod, "_write_manifest", lambda *a, **k: None)
    monkeypatch.setattr(venv_mod, "_venv_python_version", lambda exe: "3.12")
    monkeypatch.setattr(venv_mod, "installed_vllm_version", lambda device="gpu": "0.30.0")
    monkeypatch.setattr(venv_mod, "resolve_venv_python", lambda pin="": "3.12")
    monkeypatch.setattr(venv_mod.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    venv_mod.ensure_vllm_venv("", device="cpu", version="0.30.0")
    pip_cmds = [(c, e) for c, e in calls if "pip" in c]
    assert pip_cmds
    cmd, env = pip_cmds[-1]
    joined = " ".join(cmd)
    assert "vllm==0.30.0+cpu" in cmd
    assert "--extra-index-url" in cmd
    assert cmd[cmd.index("--extra-index-url") + 1] == "https://wheels.vllm.ai/0.30.0/cpu"
    assert "--index-strategy" in cmd
    assert cmd[cmd.index("--index-strategy") + 1] == "first-index"
    assert "--torch-backend=cpu" in cmd
    assert "files.pythonhosted.org" not in joined
    assert env["VLLM_TARGET_DEVICE"] == "cpu"
    assert "vllm==0.30.0 " not in joined + " "
    assert "vllm==0.30.0+cpu" in joined


def test_cpu_pip_fallback_does_not_install_pypi_vllm(tmp_path, monkeypatch):
    _isolate_runtime(tmp_path, monkeypatch)
    from hermes_cli.vllm_runtime import venv as venv_mod

    dest = venv_mod.venv_dir("cpu")
    _touch_venv_bins(dest)
    calls: list[list[str]] = []

    def _fake_stream(cmd, log_path, cwd=None, env=None):
        calls.append(list(cmd))

    monkeypatch.setattr(venv_mod, "_stream", _fake_stream)
    monkeypatch.setattr(venv_mod, "_assert_cpu", lambda *a, **k: None)
    monkeypatch.setattr(venv_mod, "_write_manifest", lambda *a, **k: None)
    monkeypatch.setattr(venv_mod, "_venv_python_version", lambda exe: "3.12")
    monkeypatch.setattr(venv_mod, "installed_vllm_version", lambda device="gpu": "0.30.0")
    monkeypatch.setattr(venv_mod, "resolve_venv_python", lambda pin="": "3.12")
    monkeypatch.setattr(venv_mod.shutil, "which", lambda name: None)

    venv_mod.ensure_vllm_venv("", device="cpu", version="0.30.0")
    vllm_cmds = [c for c in calls if any(part.startswith("vllm") for part in c)]
    assert vllm_cmds
    for cmd in vllm_cmds:
        assert "vllm==0.30.0+cpu" in cmd
        assert "--extra-index-url" in cmd
        assert "https://wheels.vllm.ai/0.30.0/cpu" in cmd


def test_cpu_wheel_already_installed_is_left_alone(tmp_path, monkeypatch):
    _isolate_runtime(tmp_path, monkeypatch)
    from hermes_cli.vllm_runtime import venv as venv_mod

    dest = venv_mod.venv_dir("cpu")
    _touch_venv_bins(dest)
    calls: list[list[str]] = []
    monkeypatch.setattr(venv_mod, "_stream", lambda *a, **k: calls.append(list(a[0])))
    monkeypatch.setattr(venv_mod, "_write_manifest", lambda *a, **k: None)
    monkeypatch.setattr(venv_mod, "_venv_python_version", lambda exe: "3.12")
    monkeypatch.setattr(venv_mod, "installed_vllm_version", lambda device="gpu": "0.30.0+cpu")
    monkeypatch.setattr(venv_mod, "resolve_venv_python", lambda pin="": "3.12")

    venv_mod.ensure_vllm_venv("", device="cpu")
    assert calls == []


def test_python_314_venv_recreated_unless_that_server_is_live(tmp_path, monkeypatch):
    """3.14 venvs are recreated. A live GPU server is not deleted out from under itself."""
    _isolate_runtime(tmp_path, monkeypatch)
    from hermes_cli.vllm_runtime import venv as venv_mod

    gpu = venv_mod.venv_dir("gpu")
    _touch_venv_bins(gpu)
    py_name = "python.exe" if os.name == "nt" else "python"
    marker = _bin_dir(gpu) / py_name
    state = venv_mod.runtimes_root("gpu") / "server.json"
    state.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    monkeypatch.setattr(venv_mod, "_venv_python_version", lambda exe: "3.14")
    monkeypatch.setattr(venv_mod, "resolve_venv_python", lambda pin="": "3.12")
    monkeypatch.setattr(venv_mod, "_write_manifest", lambda *a, **k: None)
    monkeypatch.setattr(
        venv_mod.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    def _boom(*a, **k):
        raise AssertionError("refused to rewrite a live vLLM venv")

    monkeypatch.setattr(venv_mod, "_stream", _boom)

    venv_mod.ensure_vllm_venv("", device="gpu")
    assert marker.read_text(encoding="utf-8") == "keep"

    cpu = venv_mod.venv_dir("cpu")
    _touch_venv_bins(cpu)
    created: list[list[str]] = []

    def _fake_stream(cmd, log_path, cwd=None, env=None):
        created.append(list(cmd))
        _touch_venv_bins(cpu)

    monkeypatch.setattr(venv_mod, "_stream", _fake_stream)
    monkeypatch.setattr(venv_mod, "_assert_cpu", lambda *a, **k: None)
    monkeypatch.setattr(venv_mod, "installed_vllm_version", lambda device="gpu": "")
    venv_mod.ensure_vllm_venv("", device="cpu", version="0.30.0")
    assert any("venv" in cmd for cmd in created)
    pip_cmds = [cmd for cmd in created if "pip" in cmd]
    assert pip_cmds
    assert "vllm==0.30.0+cpu" in pip_cmds[-1]
    assert marker.read_text(encoding="utf-8") == "keep"


def test_assert_cpu_probe_requires_cpu_build(monkeypatch):
    from hermes_cli.vllm_runtime import venv as venv_mod

    seen: dict = {}

    def _run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env")

        class _Proc:
            returncode = 2
            stderr = "cuda-wheel 0.30.0\n"
            stdout = ""

        return _Proc()

    monkeypatch.setattr(venv_mod.subprocess, "run", _run)
    with pytest.raises(RuntimeError, match="CPU"):
        venv_mod._assert_cpu(Path("/opt/cpu-venv/bin/python"), "cpu")
    assert seen["env"]["VLLM_TARGET_DEVICE"] == "cpu"
    script = seen["cmd"][-1]
    assert "+cpu" in script
    assert 'os.environ["VLLM_TARGET_DEVICE"] = "cpu"' in script
    assert "is_cpu" in script


def test_cuda_quants_are_cpu_incompatible_and_gpu_servable():
    """AWQ, compressed-tensors AWQ-4bit, FP8/MXFP4, and FP8 KV fail on the CPU wheel.

    GPU still accepts official AWQ. DSpark is refused on both, same as serve.
    """
    from hermes_cli.vllm_runtime.serve_compat import incompatible_with_device

    awq = "Qwen/Qwen3-8B-AWQ"
    bf16 = "Qwen/Qwen3-4B-Instruct-2507"
    assert incompatible_with_device(awq, "cpu")
    assert incompatible_with_device(awq, "gpu") is None
    assert incompatible_with_device(bf16, "cpu") is None
    assert incompatible_with_device(bf16, "gpu") is None
    smol = "HuggingFaceTB/SmolLM3-3B"
    smol_cfg = {"torch_dtype": "bfloat16", "max_position_embeddings": 65536}
    assert incompatible_with_device(smol, "cpu", config=smol_cfg) is None
    assert incompatible_with_device(smol, "gpu", config=smol_cfg) is None
    packed = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": {"num_bits": 4}}},
        },
    }
    assert incompatible_with_device("org/hermes-awq-4bit", "cpu", config=packed)
    assert incompatible_with_device("org/hermes-awq-4bit", "gpu", config=packed) is None
    fp8 = {"quantization_config": {"quant_method": "fp8"}}
    assert incompatible_with_device("org/weights-fp8", "cpu", config=fp8)
    assert incompatible_with_device("org/weights-fp8", "gpu", config=fp8) is None
    assert incompatible_with_device("openai/gpt-oss-20b", "cpu")
    assert incompatible_with_device("openai/gpt-oss-20b", "gpu") is None
    assert incompatible_with_device(bf16, "cpu", config={"kv_cache_dtype": "fp8"})
    assert incompatible_with_device(bf16, "gpu", config={"kv_cache_dtype": "fp8"}) is None
    draft = "nvidia/NVIDIA-Nemotron-3-Nano-DSpark"
    cpu_draft = incompatible_with_device(draft, "cpu") or ""
    gpu_draft = incompatible_with_device(draft, "gpu") or ""
    assert "draft" in cpu_draft.lower()
    assert cpu_draft == gpu_draft


def test_cpu_search_drops_awq_gpu_search_keeps_it(monkeypatch):
    from hermes_cli.vllm_runtime.inventory import search_hf_models
    from hermes_cli.vllm_runtime.recommend import NvidiaProbe, VllmRecommendation, catalog_tiers

    _gib = 1 << 30
    pick = next(t for t in catalog_tiers() if t.id == "24gb")
    rec = VllmRecommendation(NvidiaProbe(24 * _gib, 24 * _gib, "data"), pick, True, "ok")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.hardware._ram_stats",
        lambda: (128 * _gib, 16 * _gib, 96 * _gib),
    )
    chosen = {"device": "cpu"}
    monkeypatch.setattr(
        "hermes_cli.local_engines.vllm_device_from_config",
        lambda config=None: chosen["device"],
    )

    def _fake(_url, timeout=15):
        return [
            {"id": "Qwen/Qwen3-8B-AWQ", "downloads": 9, "likes": 1,
             "lastModified": "", "gated": False, "tags": ["awq", "instruct"]},
            {"id": "Qwen/Qwen3-4B-Instruct-2507", "downloads": 4, "likes": 1,
             "lastModified": "", "gated": False, "tags": ["instruct"]},
            {"id": "nvidia/NVIDIA-Nemotron-3-Nano-DSpark", "downloads": 1, "likes": 0,
             "lastModified": "", "gated": False, "tags": ["dspark"]},
        ]

    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory._hf_json", _fake)
    cpu_repos = {hit["repo"] for hit in search_hf_models("qwen")}
    assert "Qwen/Qwen3-4B-Instruct-2507" in cpu_repos
    assert "Qwen/Qwen3-8B-AWQ" not in cpu_repos
    assert not any("dspark" in repo.lower() for repo in cpu_repos)
    four = next(hit for hit in search_hf_models("qwen") if hit["repo"].endswith("2507"))
    assert four["fit"] == "needs-ram"

    chosen["device"] = "gpu"
    gpu_repos = {hit["repo"] for hit in search_hf_models("qwen")}
    assert "Qwen/Qwen3-8B-AWQ" in gpu_repos
    assert not any("dspark" in repo.lower() for repo in gpu_repos)


def test_cpu_supervisor_refuses_awq_before_spawn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    from hermes_cli.vllm_runtime.supervisor import VllmSupervisor

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.pick_listen_port",
        lambda preferred=0, **k: 19991,
    )
    fake = tmp_path / "vllm"
    fake.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    spawned: list[str] = []
    sup = VllmSupervisor(
        {"model": "Qwen/Qwen3-8B-AWQ", "port": 19991},
        executable=fake,
        device="cpu",
        log_path=home / "vllm-server.log",
    )
    monkeypatch.setattr(sup, "_spawn", lambda: spawned.append("spawn"))
    with pytest.raises(RuntimeError, match="CPU vLLM wheel"):
        sup.start(timeout_s=1)
    assert spawned == []
