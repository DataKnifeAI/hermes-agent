"""Contracts for the managed vLLM engine (Phase 1).

Relationships, not snapshots: default engine merge, argv bind/tool-parser,
VRAM → feasible, activate loopback-vs-remote, stop-other-engine ordering.
A fake ``vllm`` executable serves ``GET /v1/models`` — no live GPU.
"""

from __future__ import annotations

import json
import socket
import stat
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.vllm_runtime.recommend import MIN_CONTEXT, TIERS, TOOL_PARSERS, recommend_vllm
from hermes_cli.vllm_runtime.supervisor import serve_argv, vllm_settings


_GIB = 1 << 30

_FAKE_VLLM = textwrap.dedent("""\
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    port = 18435
    host = "127.0.0.1"
    args = sys.argv[1:]
    if args and args[0] == "serve":
        args = args[1:]
    if args:
        args = args[1:]  # model id
    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1]); i += 2; continue
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]; i += 2; continue
        i += 1
    bind = "127.0.0.1" if host in ("0.0.0.0", "::") else host

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/v1/models":
                body = b'{"object":"list","data":[{"id":"hermes3:8b"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()
        def log_message(self, *args):
            pass

    HTTPServer((bind, port), H).serve_forever()
""")


def _write_fake_vllm(path: Path) -> Path:
    path.write_text(f"#!{sys.executable}\n{_FAKE_VLLM}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_default_engine_is_llamacpp_and_missing_key_deep_merges():
    from hermes_cli.config import _deep_merge

    section = DEFAULT_CONFIG["local_runtime"]
    assert section["engine"] == "llamacpp"
    vllm = section["vllm"]
    assert vllm["host"] == "127.0.0.1"
    from hermes_cli.vllm_runtime.supervisor import DEFAULT_LISTEN_PORT, LLAMA_CPP_PORT

    # 0 = pick at spawn (llama.cpp's rule). Preferred bind is not llama.cpp's
    # port and not the ports a user's own vLLM/Ollama already occupy.
    assert int(vllm["port"]) == 0
    assert DEFAULT_LISTEN_PORT != LLAMA_CPP_PORT
    assert DEFAULT_LISTEN_PORT not in (8000, 8080)
    assert int(vllm["max_model_len"]) >= MIN_CONTEXT

    merged = _deep_merge(DEFAULT_CONFIG, {"local_runtime": {"enabled": True}})
    assert merged["local_runtime"]["engine"] == "llamacpp"
    assert merged["local_runtime"]["vllm"]["host"] == "127.0.0.1"


def test_serve_argv_drops_leftover_awq_on_non_awq_id():
    """Recommend leftover ``awq`` must not ride along onto Dolphin BF16."""
    argv = serve_argv("/opt/venv/bin/vllm", {
        "model": "dphn/dolphin-2.9.1-llama-3-8b",
        "quantization": "awq",
        "port": 18435,
    })
    assert "--quantization" not in argv
    kept = serve_argv("/opt/venv/bin/vllm", {
        "model": "Qwen/Qwen3-8B-AWQ",
        "quantization": "awq",
        "port": 18435,
    })
    assert kept[kept.index("--quantization") + 1] == "awq"


def test_start_refuses_exl2_without_spawning(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    hub = tmp_path / "hub"
    cached = hub / "models--org--Dolphin-8B-exl2"
    cached.mkdir(parents=True)
    (cached / "w.bin").write_bytes(b"x" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    import hermes_constants
    from hermes_cli.vllm_runtime.inventory import UNSERVABLE_FORMAT_MSG
    from hermes_cli.vllm_runtime.supervisor import VllmSupervisor, read_last_error

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.pick_listen_port", lambda preferred=0: 19997)
    fake = tmp_path / "vllm"
    fake.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    spawned: list[str] = []
    sup = VllmSupervisor(
        {"model": "org/Dolphin-8B-exl2", "port": 19997},
        executable=fake,
        log_path=home / "runtimes" / "vllm" / "vllm-server.log",
    )
    monkeypatch.setattr(sup, "_spawn", lambda: spawned.append("spawn"))
    with pytest.raises(RuntimeError, match="cannot serve"):
        sup.start(timeout_s=1)
    assert spawned == []
    assert read_last_error() == UNSERVABLE_FORMAT_MSG


def test_start_sigkill_writes_last_error_not_generic_failed(tmp_path, monkeypatch):
    """rc=-9 (OOM / port fight) must not collapse to ``vllm serve failed``."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    hub = tmp_path / "hub"
    cached = hub / "models--Qwen--Qwen3-8B-AWQ"
    cached.mkdir(parents=True)
    (cached / "w.bin").write_bytes(b"x" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    import hermes_constants
    from hermes_cli.vllm_runtime.supervisor import VllmSupervisor, read_last_error

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.pick_listen_port", lambda preferred=0: 19995)
    fake = tmp_path / "vllm"
    fake.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    sup = VllmSupervisor(
        {"model": "Qwen/Qwen3-8B-AWQ", "port": 19995, "served_model_name": "qwen3:8b"},
        executable=fake,
        log_path=home / "runtimes" / "vllm" / "vllm-server.log",
    )

    class _Killed:
        returncode = -9

        def poll(self):
            return -9

    def _spawn():
        sup.proc = _Killed()

    monkeypatch.setattr(sup, "_spawn", _spawn)
    with pytest.raises(RuntimeError, match="SIGKILL"):
        sup.start(timeout_s=1)
    assert "SIGKILL" in (read_last_error() or "")
    assert "vllm serve failed" not in (read_last_error() or "")


def test_watch_stops_on_fatal_awq_config_error(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("local_runtime:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    hub = tmp_path / "hub"
    cached = hub / "models--dphn--dolphin-2.9.1-llama-3-8b"
    cached.mkdir(parents=True)
    (cached / "w.bin").write_bytes(b"x" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    import hermes_constants
    from hermes_cli.vllm_runtime.supervisor import (
        LEFTOVER_AWQ_MSG, VllmSupervisor, read_last_error)

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.pick_listen_port", lambda preferred=0: 19996)
    log_path = home / "runtimes" / "vllm" / "vllm-server.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "Value error, Cannot find the config file for awq [type=value_error]\n",
        encoding="utf-8")
    spawned: list[str] = []
    sup = VllmSupervisor(
        {"model": "dphn/dolphin-2.9.1-llama-3-8b", "port": 19996, "quantization": "awq"},
        executable=tmp_path / "vllm",
        log_path=log_path,
    )
    monkeypatch.setattr(sup, "_spawn", lambda: spawned.append("spawn"))

    class _Dead:
        def poll(self):
            return 1

    sup.proc = _Dead()
    sup._watch()
    assert spawned == []
    assert sup._stopping is True
    assert read_last_error() == LEFTOVER_AWQ_MSG


def test_serve_argv_loopback_and_hermes_tool_parser():
    from hermes_cli.vllm_runtime.supervisor import DEFAULT_LISTEN_PORT, LLAMA_CPP_PORT

    settings = vllm_settings(DEFAULT_CONFIG)
    argv = serve_argv("/opt/venv/bin/vllm", settings)
    assert argv[0].endswith("vllm")
    assert argv[1] == "serve"
    assert "--host" in argv
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"
    port = int(argv[argv.index("--port") + 1])
    assert port == (int(settings["port"]) or DEFAULT_LISTEN_PORT)
    assert port != LLAMA_CPP_PORT
    assert port not in (8000, 8080)
    # 1-click never binds all-interfaces unless the user set host.
    settings["host"] = "0.0.0.0"
    remote = serve_argv("/opt/venv/bin/vllm", settings)
    assert remote[remote.index("--host") + 1] == "0.0.0.0"


def test_shipped_default_is_smallest_64k_feasible_id():
    """No-probe / DEFAULT_CONFIG use the smallest 64k-feasible official row."""
    feasible = [t for t in TIERS if t.feasible_at_64k]
    assert feasible
    floor = min(t.min_vram_bytes for t in feasible)
    default_tier = next(t for t in feasible if t.min_vram_bytes == floor)
    shipped = DEFAULT_CONFIG["local_runtime"]["vllm"]["model"]
    assert shipped == default_tier.model
    assert shipped == recommend_vllm(total_bytes=0).model
    assert "/" in shipped
    roomy = recommend_vllm(total_bytes=24 * _GIB)
    assert roomy.feasible is True
    assert roomy.model != shipped
    fits = [t for t in TIERS if t.min_vram_bytes <= 24 * _GIB]
    assert roomy.model == max(fits, key=lambda t: t.min_vram_bytes).model


def test_recommend_vram_relationship_not_snapshot():
    floor = min(t.min_vram_bytes for t in TIERS if t.feasible_at_64k)
    tight = recommend_vllm(total_bytes=8 * _GIB)
    assert tight.tier is not None
    assert tight.tier.min_vram_bytes <= 8 * _GIB
    assert tight.feasible is False
    assert tight.max_model_len >= MIN_CONTEXT

    roomy = recommend_vllm(total_bytes=24 * _GIB)
    assert roomy.tier is not None
    assert roomy.tier.min_vram_bytes <= 24 * _GIB
    assert roomy.feasible is True
    assert roomy.feasible == (24 * _GIB >= floor)
    assert "/" in roomy.model  # Hugging Face id

    none = recommend_vllm(total_bytes=0)
    assert none.feasible is False
    assert none.tier is None


def test_recommend_catalog_vram_and_parser_relationship():
    """Every catalog row has min-VRAM + tool-parser; recommend never picks a
    row whose min exceeds probed total; feasible GPUs get distinct HF ids;
    8–12 GB stay infeasible without shrinking below 64k. No GGUF."""
    for tier in TIERS:
        assert tier.min_vram_bytes > 0
        assert tier.tool_call_parser in TOOL_PARSERS
        assert tier.max_model_len >= MIN_CONTEXT
        assert "/" in tier.model
        assert "gguf" not in tier.model.lower()
        assert tier.served_model_name

    feasible = [t for t in TIERS if t.feasible_at_64k]
    assert feasible
    assert len({t.model for t in feasible}) == len(feasible)

    for gib in (8, 12):
        rec = recommend_vllm(total_bytes=gib * _GIB)
        assert rec.feasible is False
        assert rec.max_model_len >= MIN_CONTEXT

    floor = min(t.min_vram_bytes for t in feasible)
    picks: list[str] = []
    for tier in feasible:
        rec = recommend_vllm(total_bytes=tier.min_vram_bytes)
        assert rec.max_model_len >= MIN_CONTEXT
        assert rec.tier is not None
        assert rec.tier.min_vram_bytes <= tier.min_vram_bytes
        assert rec.feasible is (tier.min_vram_bytes >= floor)
        assert rec.tool_call_parser in TOOL_PARSERS
        assert rec.model == tier.model
        assert "gguf" not in rec.model.lower()
        picks.append(rec.model)
    assert len(set(picks)) == len(picks)


def test_official_catalog_is_short_feasible_list():
    """llama.cpp-like: official rows are 64k-feasible and unique; 8–12 GB
    stay infeasible without a sixth official id; the largest row is not
    the 24 GB pick (Qwen3.8 analog only on 80 GB class)."""
    from hermes_cli.vllm_runtime.recommend import catalog_tiers

    official = catalog_tiers()
    assert official
    assert all(t.feasible_at_64k for t in official)
    assert len({t.model for t in official}) == len(official)
    infeasible = [t for t in TIERS if not t.feasible_at_64k]
    assert infeasible
    official_ids = {t.model for t in official}
    assert {t.model for t in infeasible} <= official_ids

    for gib in (8, 12):
        rec = recommend_vllm(total_bytes=gib * _GIB)
        assert rec.feasible is False
        assert rec.tier is None or not rec.tier.catalog

    largest = max(official, key=lambda t: t.min_vram_bytes)
    mid = recommend_vllm(total_bytes=24 * _GIB)
    assert mid.feasible is True
    assert mid.model != largest.model
    assert recommend_vllm(total_bytes=largest.min_vram_bytes).model == largest.model


def test_catalog_models_lists_official_short_list_not_floor_marker(monkeypatch):
    from hermes_cli.vllm_runtime.inventory import catalog_models
    from hermes_cli.vllm_runtime.recommend import (
        NvidiaProbe, VllmRecommendation, catalog_tiers,
    )

    official = catalog_tiers()
    pick = next(t for t in official if t.min_vram_bytes <= 24 * _GIB)
    rec = VllmRecommendation(
        NvidiaProbe(24 * _GIB, 24 * _GIB, "data"), pick, True, "ok")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.list_cached_repos", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.vllm_settings", lambda cfg=None: {})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.running_served_model_name", lambda: "")
    rows = catalog_models({})
    official_ids = {t.model for t in official}
    listed = {r["id"] for r in rows if not r.get("added_by_you")}
    assert listed == official_ids
    assert sum(1 for r in rows if r.get("recommended")) == 1


def test_recommend_libcuda_when_smi_missing(monkeypatch):
    """Driver/library mismatch: nvidia-smi dead, ctypes libcuda still sees the card."""
    import hermes_cli.local_runtime.hardware as hardware
    from hermes_cli.vllm_runtime.recommend import probe_nvidia_vram, recommend_vllm as rec

    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hardware, "_cuda_driver_pool", lambda: (24 * _GIB, False))
    probe = probe_nvidia_vram()
    assert probe.source == "libcuda"
    assert probe.total_bytes == 24 * _GIB
    result = rec(probe=probe)
    assert result.feasible is True


def test_switch_to_vllm_stops_llama_first(monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.shutdown_local_runtime",
        lambda: order.append("llama_stop"))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda config=None, force=False, **kw: order.append("vllm_start") or "sup")
    from hermes_cli.local_engines import ensure_managed_engine

    ensure_managed_engine({"local_runtime": {"enabled": True, "engine": "vllm"}}, force=True)
    assert order == ["llama_stop", "vllm_start"]


def test_switch_to_llamacpp_stops_vllm_first(monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.shutdown_vllm_runtime",
        lambda: order.append("vllm_stop"))
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.ensure_local_runtime",
        lambda config=None, force=False: order.append("llama_start") or "sup")
    from hermes_cli.local_engines import ensure_managed_engine

    ensure_managed_engine({"local_runtime": {"enabled": True, "engine": "llamacpp"}}, force=True)
    assert order == ["vllm_stop", "llama_start"]


def test_activate_writes_loopback_v1_and_preserves_remote(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)

    from cli import save_config_value
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.bootstrap import activate_vllm_provider

    (home / "config.yaml").write_text(yaml.dump({"model": {}}), encoding="utf-8")
    url = activate_vllm_provider(load_config())
    cfg = load_config()
    assert cfg["model"]["provider"] == "vllm"
    assert cfg["model"]["default"]
    assert (cfg.get("providers") or {}).get("vllm", {}).get("name") == "Local"
    assert url.endswith("/v1")
    host = url.split("://", 1)[-1].split(":")[0]
    assert host in ("127.0.0.1", "localhost")
    assert cfg["model"]["base_url"] == url

    save_config_value("model.base_url", "http://gpu-box.example:8000/v1")
    kept = activate_vllm_provider(load_config())
    assert kept == "http://gpu-box.example:8000/v1"
    assert load_config()["model"]["base_url"] == kept


def test_ensure_vllm_runtime_fake_server(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)

    fake = _write_fake_vllm(tmp_path / "fake-vllm")
    hub = tmp_path / "hub"
    cached = hub / "models--solidrust--Hermes-3-Llama-3.1-8B-AWQ"
    cached.mkdir(parents=True)
    (cached / "w.bin").write_bytes(b"x")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    import hermes_cli.vllm_runtime.bootstrap as boot

    monkeypatch.setattr(boot, "_SUPERVISOR", None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)
    cfg = {
        "local_runtime": {
            "enabled": True,
            "engine": "vllm",
            "vllm": {"host": "127.0.0.1",
                     "model": "solidrust/Hermes-3-Llama-3.1-8B-AWQ",
                     "served_model_name": "hermes3:8b"},
        },
    }
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfg["local_runtime"]["vllm"]["port"] = port

    try:
        sup = boot.ensure_vllm_runtime(cfg, force=True, executable=fake, timeout_s=15)
        assert sup is not None
        assert sup.base_url.endswith("/v1")
        assert str(port) in sup.base_url
        state = json.loads((home / "runtimes" / "vllm" / "server.json").read_text())
        assert state["pid"] == sup.proc.pid
        assert state["base_url"].endswith("/v1")
        monkeypatch.setattr("cli._hermes_home", home)
        (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
        from hermes_cli.config import load_config

        url = boot.activate_vllm_provider(cfg)
        assert str(sup.port) in url
        assert str(sup.port) in load_config()["model"]["base_url"]
    finally:
        boot.shutdown_vllm_runtime()
        hermes_constants._default_hermes_root_memo = None


def test_watch_stops_when_configured_cache_is_gone(tmp_path, monkeypatch):
    """Deleted weights must halt `_watch`, not backoff-restart and re-pull."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    import hermes_constants
    from hermes_cli.vllm_runtime.supervisor import (
        MODEL_REMOVED_MSG, VllmSupervisor, read_last_error)

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.pick_listen_port", lambda preferred=0: 19999)
    fake = tmp_path / "vllm"
    fake.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    spawned: list[str] = []
    sup = VllmSupervisor(
        {"model": "acme/gone-awq", "port": 19999},
        executable=fake,
        log_path=home / "runtimes" / "vllm" / "vllm-server.log",
    )
    monkeypatch.setattr(sup, "_spawn", lambda: spawned.append("spawn"))

    class _Dead:
        def poll(self):
            return 1

    sup.proc = _Dead()
    sup._watch()
    assert spawned == []
    assert sup._stopping is True
    assert read_last_error() == MODEL_REMOVED_MSG


def test_start_refuses_when_configured_cache_is_gone(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    import hermes_constants
    from hermes_cli.vllm_runtime.supervisor import (
        MODEL_REMOVED_MSG, VllmSupervisor, read_last_error)

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.pick_listen_port", lambda preferred=0: 19998)
    fake = tmp_path / "vllm"
    fake.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    spawned: list[str] = []
    sup = VllmSupervisor(
        {"model": "acme/gone-awq", "port": 19998},
        executable=fake,
        log_path=home / "runtimes" / "vllm" / "vllm-server.log",
    )
    monkeypatch.setattr(sup, "_spawn", lambda: spawned.append("spawn"))
    with pytest.raises(RuntimeError, match="removed"):
        sup.start(timeout_s=1)
    assert spawned == []
    assert read_last_error() == MODEL_REMOVED_MSG


def test_inventory_lists_and_deletes_only_hf_cache(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    kept = hub / "models--org--kept"
    doomed = hub / "models--org--doomed"
    kept.mkdir(parents=True)
    doomed.mkdir(parents=True)
    (kept / "weights.bin").write_bytes(b"aa")
    (doomed / "weights.bin").write_bytes(b"bb")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    from hermes_cli.vllm_runtime.inventory import delete_cached_repo, list_cached_repos

    ids = {r["id"] for r in list_cached_repos()}
    assert ids == {"org/kept", "org/doomed"}
    delete_cached_repo("org/doomed")
    assert not doomed.exists()
    assert kept.is_dir()


def test_installed_version_uses_isolated_python(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    from hermes_cli.vllm_runtime import venv as venv_mod

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    fake_py = tmp_path / "venv-python"
    fake_py.write_text("", encoding="utf-8")
    seen: list[list[str]] = []

    def _out(cmd, text=True, timeout=30):
        seen.append(list(cmd))
        return "0.10.0\n"

    monkeypatch.setattr(venv_mod, "venv_python", lambda: fake_py)
    monkeypatch.setattr(venv_mod, "_assert_isolated", lambda py: None)
    monkeypatch.setattr(venv_mod.subprocess, "check_output", _out)
    assert venv_mod.installed_vllm_version() == "0.10.0"
    assert seen and seen[0][0] == str(fake_py)
    assert str(Path(sys.prefix).resolve()) not in " ".join(seen[0])


def test_vllm_runtimes_are_machine_scoped(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    from hermes_cli.vllm_runtime.venv import runtimes_root

    assert runtimes_root() == root / "runtimes" / "vllm"
    assert "profiles" not in runtimes_root().parts


def test_venv_python_defaults_to_hermes_interpreter_not_a_pin():
    from hermes_cli.vllm_runtime.venv import resolve_venv_python

    with pytest.raises(RuntimeError, match="need CPython"):
        resolve_venv_python("notapython")
    resolved = resolve_venv_python("")
    assert resolved
    path = Path(resolved)
    if path.is_file():
        import json
        import subprocess
        major, minor = json.loads(subprocess.check_output(
            [str(path), "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"],
            text=True,
        ))
        assert (major, minor) >= (3, 12)
    else:
        assert resolved[0].isdigit()


def test_ensure_venv_never_installs_into_hermes_prefix(tmp_path, monkeypatch):
    """Wheels go under runtimes/vllm/.venv, never sys.prefix (the Hermes venv)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    from hermes_cli.vllm_runtime import venv as venv_mod

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(venv_mod, "resolve_venv_python", lambda pin="": str(Path(sys.executable)))
    hermes_prefix = Path(sys.prefix).resolve()
    calls: list[list[str]] = []

    def _fake_stream(cmd, log_path):
        calls.append(list(cmd))
        dest = venv_mod.venv_dir()
        bin_dir = dest / ("Scripts" if sys.platform == "win32" else "bin")
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / ("python.exe" if sys.platform == "win32" else "python")).write_text("", encoding="utf-8")
        exe = "vllm.exe" if sys.platform == "win32" else "vllm"
        (bin_dir / exe).write_text("", encoding="utf-8")

    monkeypatch.setattr(venv_mod, "_stream", _fake_stream)
    monkeypatch.setattr(venv_mod, "_assert_cuda", lambda py: None)
    monkeypatch.setattr(venv_mod, "_write_manifest", lambda py: None)
    monkeypatch.setattr(venv_mod, "shutil", venv_mod.shutil)
    monkeypatch.setattr(venv_mod.shutil, "which", lambda name: None)

    exe = venv_mod.ensure_vllm_venv("")
    assert hermes_prefix not in exe.resolve().parents
    assert "runtimes" in exe.parts and "vllm" in exe.parts
    assert calls, "expected a venv create / pip command"
    for cmd in calls:
        joined = " ".join(cmd)
        assert str(hermes_prefix) not in joined or "-m venv" in joined
        # pip/uv install target is the isolated dest, not Hermes site-packages.
        if "pip" in cmd or (len(cmd) > 1 and cmd[1] == "pip"):
            assert str(venv_mod.venv_dir()) in joined or str(venv_mod.venv_python()) in joined


def test_uv_pip_install_ignores_project_exclude_newer(tmp_path, monkeypatch):
    """Desktop serve cwd is the Hermes checkout; uv must not read its
    ``exclude-newer = 14 days`` or Update silently keeps the old wheel."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    from hermes_cli.vllm_runtime import venv as venv_mod

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(venv_mod, "resolve_venv_python", lambda pin="": str(Path(sys.executable)))
    calls: list[list[str]] = []

    def _fake_stream(cmd, log_path, cwd=None, env=None):
        calls.append(list(cmd))
        dest = venv_mod.venv_dir()
        bin_dir = dest / ("Scripts" if sys.platform == "win32" else "bin")
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / ("python.exe" if sys.platform == "win32" else "python")).write_text("", encoding="utf-8")
        exe = "vllm.exe" if sys.platform == "win32" else "vllm"
        (bin_dir / exe).write_text("", encoding="utf-8")

    monkeypatch.setattr(venv_mod, "_stream", _fake_stream)
    monkeypatch.setattr(venv_mod, "_assert_cuda", lambda py: None)
    monkeypatch.setattr(venv_mod, "_write_manifest", lambda py: None)
    monkeypatch.setattr(venv_mod.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    venv_mod.ensure_vllm_venv("", upgrade=True, version="0.28.0")
    pip_cmds = [c for c in calls if "pip" in c]
    assert pip_cmds
    for cmd in pip_cmds:
        assert "--no-config" in cmd
        assert "vllm==0.28.0" in cmd


def test_apply_vllm_update_refuses_silent_no_op(tmp_path, monkeypatch):
    """A resolver that leaves the old tag behind must fail the job."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    from hermes_cli.web_routers import local_models_engine as engine
    from hermes_cli.vllm_runtime import venv as venv_mod

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.vllm_settings",
        lambda cfg: {"python": ""})
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(venv_mod, "latest_vllm_pypi_version", lambda: "0.28.0")
    monkeypatch.setattr(venv_mod, "ensure_vllm_venv", lambda *a, **k: Path("/tmp/vllm"))
    monkeypatch.setattr(venv_mod, "installed_vllm_version", lambda: "0.27.1")
    monkeypatch.setattr(venv_mod, "install_log_path", lambda: home / "install.log")
    with pytest.raises(RuntimeError, match="still 0.27.1"):
        engine.apply_vllm_update()


def test_apply_vllm_update_records_matching_pypi_tag(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    from hermes_cli.web_routers import local_models_engine as engine
    from hermes_cli.vllm_runtime import venv as venv_mod

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.vllm_settings",
        lambda cfg: {"python": ""})
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(venv_mod, "latest_vllm_pypi_version", lambda: "0.28.0")
    monkeypatch.setattr(venv_mod, "ensure_vllm_venv", lambda *a, **k: Path("/tmp/vllm"))
    monkeypatch.setattr(venv_mod, "installed_vllm_version", lambda: "0.28.0")
    engine.apply_vllm_update()
    remembered = venv_mod.read_version_check()
    assert remembered["installed"] == "0.28.0"
    assert remembered["update_available"] is False


def test_pick_listen_port_never_shares_llamacpp_and_falls_back_when_busy():
    from hermes_cli.vllm_runtime.supervisor import LLAMA_CPP_PORT, pick_listen_port

    picked = pick_listen_port(0)
    assert picked != LLAMA_CPP_PORT
    assert picked not in (8000, 8080)
    # Asking for llama.cpp's port still lands on the vLLM default or ephemeral.
    redirected = pick_listen_port(LLAMA_CPP_PORT)
    assert redirected != LLAMA_CPP_PORT

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        taken = busy.getsockname()[1]
        fallback = pick_listen_port(taken)
    assert fallback != taken
    assert fallback != LLAMA_CPP_PORT


def test_gpu_process_name_is_foreign_llm_not_desktop():
    from hermes_cli.vllm_runtime.occupancy import gpu_process_is_foreign_llm

    assert gpu_process_is_foreign_llm("VLLM::EngineCore", 42, set())
    assert not gpu_process_is_foreign_llm("VLLM::EngineCore", 42, {42})
    assert gpu_process_is_foreign_llm("llama-server", 7, set())
    assert not gpu_process_is_foreign_llm("kwin_wayland", 1, set())
    assert not gpu_process_is_foreign_llm("brave", 2, set())


def test_managed_serve_child_enginecore_is_not_occupancy():
    """server.json records the serve parent; EngineCore is the GPU child."""
    from hermes_cli.vllm_runtime.occupancy import (
        discover_occupying_llms, expand_managed_pids, occupancy_stop_message)

    parent, child = 100, 101
    ours = expand_managed_pids({parent}, children_of=lambda pid: {child} if pid == parent else set())
    assert child in ours
    hits = discover_occupying_llms(
        ports=(),
        gpu_rows=[(child, "VLLM::EngineCore")],
        our_pids=ours,
        our_ports={18435},
    )
    assert hits == []
    assert occupancy_stop_message(hits) is None
    foreign = discover_occupying_llms(
        ports=(),
        gpu_rows=[(child, "VLLM::EngineCore")],
        our_pids={parent},
        our_ports={18435},
    )
    assert foreign and foreign[0].pid == child


def test_occupancy_http_stub_mentions_stop(monkeypatch):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    from hermes_cli.vllm_runtime.occupancy import (
        OccupyingLlmError, discover_occupying_llms, occupancy_stop_message,
        require_gpu_free)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/v1/models":
                body = b'{"object":"list","data":[{"id":"x","owned_by":"vllm"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        port = httpd.server_address[1]
        empty = discover_occupying_llms(
            ports=(), gpu_rows=[], our_pids=set(), our_ports=set())
        assert empty == []
        assert occupancy_stop_message(empty) is None

        hits = discover_occupying_llms(
            ports=(port,), gpu_rows=[], our_pids=set(), our_ports=set())
        assert hits
        msg = occupancy_stop_message(hits)
        assert msg is not None
        assert "Stop it so managed vLLM" in msg
        assert str(port) in msg

        monkeypatch.setattr(
            "hermes_cli.vllm_runtime.occupancy.discover_occupying_llms",
            lambda **kw: hits)
        with pytest.raises(OccupyingLlmError, match="Stop it so managed vLLM"):
            require_gpu_free()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_start_managed_vllm_refuses_foreign_llm_after_stopping_llama(monkeypatch):
    order: list[str] = []
    from hermes_cli.vllm_runtime.occupancy import OccupyingLlmError

    monkeypatch.setattr("cli.save_config_value", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.shutdown_local_runtime",
        lambda: order.append("llama_stop"))

    def _occupied():
        order.append("occupancy")
        raise OccupyingLlmError(
            "Another LLM is already running (Ollama on http://127.0.0.1:11434). "
            "Stop it so managed vLLM can use the GPU.")

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.occupancy.require_gpu_free", _occupied)
    monkeypatch.setattr(
        "hermes_cli.local_engines.ensure_managed_engine",
        lambda *a, **k: order.append("start") or "sup")
    from hermes_cli.vllm_runtime.bootstrap import start_managed_vllm

    with pytest.raises(OccupyingLlmError, match="Stop it so managed vLLM"):
        start_managed_vllm({"local_runtime": {}}, apply_recommend=False)
    assert order == ["llama_stop", "occupancy"]


def _alive_supervisor(tmp_path, monkeypatch, port=19980):
    from hermes_cli.vllm_runtime.supervisor import VllmSupervisor

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.pick_listen_port", lambda preferred=0: port)
    sup = VllmSupervisor(
        {"model": "acme/wait-awq", "port": port},
        executable=tmp_path / "vllm",
        log_path=tmp_path / "vllm-server.log",
    )

    class _Alive:
        def poll(self):
            return None

    sup.proc = _Alive()
    return sup


def test_probe_served_model_name_empty_until_200(monkeypatch):
    import urllib.error
    from hermes_cli.vllm_runtime.supervisor import probe_served_model_name

    def refuse(_url, timeout=1.5):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    assert probe_served_model_name("http://127.0.0.1:18435/v1") == ""


def test_probe_served_model_name_reads_models_payload(monkeypatch):
    from hermes_cli.vllm_runtime.supervisor import probe_served_model_name

    class _Ok:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"data":[{"id":"qwen3:8b"}]}'

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Ok())
    assert probe_served_model_name("http://127.0.0.1:18435/v1") == "qwen3:8b"


def test_running_served_name_ignores_spawn_state_until_200(monkeypatch):
    from hermes_cli.vllm_runtime.inventory import running_served_model_name

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: {"base_url": "http://127.0.0.1:18435/v1", "pid": 9})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.state_served_model_name",
        lambda: "Hermes-3-Llama-3.1-8B")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.probe_served_model_name",
        lambda *a, **k: "")
    assert running_served_model_name() == ""


def test_wait_ready_keeps_polling_while_proc_alive_until_200(tmp_path, monkeypatch):
    """Connection refused / 503 is not failure while the pid is still starting."""
    import urllib.error
    from io import BytesIO

    sup = _alive_supervisor(tmp_path, monkeypatch)
    hits = {"n": 0}

    class _Ok:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"data":[]}'

    def fake_urlopen(url, timeout=3):
        hits["n"] += 1
        if hits["n"] == 1:
            raise urllib.error.URLError("connection refused")
        if hits["n"] == 2:
            raise urllib.error.HTTPError(url, 503, "warming", hdrs=None, fp=BytesIO())
        return _Ok()

    now = [0.0]
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    monkeypatch.setattr("time.sleep", lambda s: now.__setitem__(0, now[0] + s))

    sup._wait_ready(5)
    assert hits["n"] == 3


def test_wait_ready_does_not_fail_while_alive_until_deadline(tmp_path, monkeypatch):
    import urllib.error

    sup = _alive_supervisor(tmp_path, monkeypatch, port=19981)

    def fake_urlopen(url, timeout=3):
        raise urllib.error.URLError("connection refused")

    now = [0.0]
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    monkeypatch.setattr("time.sleep", lambda s: now.__setitem__(0, now[0] + 0.5))

    with pytest.raises(TimeoutError, match="not ready"):
        sup._wait_ready(1)
    assert sup.proc.poll() is None


def test_wait_ready_fails_immediately_when_proc_exits(tmp_path, monkeypatch):
    from hermes_cli.vllm_runtime.supervisor import VllmSupervisor

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.pick_listen_port", lambda preferred=0: 19982)
    sup = VllmSupervisor(
        {"model": "acme/dead-awq", "port": 19982},
        executable=tmp_path / "vllm",
        log_path=tmp_path / "vllm-server.log",
    )

    class _Dead:
        returncode = 1

        def poll(self):
            return 1

    sup.proc = _Dead()
    opened = {"n": 0}

    def fake_urlopen(url, timeout=3):
        opened["n"] += 1
        raise AssertionError("must not probe /v1/models after the pid died")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="exited rc=1"):
        sup._wait_ready(30)
    assert opened["n"] == 0

