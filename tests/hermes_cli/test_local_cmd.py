"""Phase 2 ``hermes local`` CLI: occupancy, activate, recommend, stop-only-ours."""

from __future__ import annotations

import argparse
import subprocess
import sys
from types import SimpleNamespace

import yaml

from hermes_cli.local_cmd import cmd_local
from hermes_cli.subcommands.local import build_local_parser
from hermes_cli.vllm_runtime.occupancy import OccupyingLlmError
from hermes_cli.vllm_runtime.recommend import recommend_vllm


_GIB = 1 << 30


def _home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    (home / "config.yaml").write_text(
        yaml.dump({"local_runtime": {"enabled": True, "engine": "vllm"}, "model": {}}),
        encoding="utf-8")
    return home


def test_local_parser_wires_install_start_stop_use_and_ls_alias():
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    seen: list = []
    build_local_parser(sub, cmd_local=lambda args: seen.append(args) or 0)

    for argv, command in (
        (["local", "install"], "install"),
        (["local", "start"], "start"),
        (["local", "stop"], "stop"),
        (["local", "use"], "use"),
        (["local", "ls"], "ls"),
        (["local", "list"], "list"),
        (["local", "engine", "vllm"], "engine"),
        (["local", "recommend", "--apply"], "recommend"),
    ):
        ns = parser.parse_args(argv)
        assert ns.command == "local"
        assert ns.local_command == command
        assert ns.func is not None
    ns = parser.parse_args(["local", "engine", "vllm"])
    assert ns.engine_name == "vllm"
    ns = parser.parse_args(["local", "recommend", "--apply"])
    assert ns.apply is True


def test_local_start_prints_occupancy_message(tmp_path, monkeypatch, capsys):
    """Start must surface OccupyingLlmError, not swallow it into a generic failure."""
    _home(tmp_path, monkeypatch)
    monkeypatch.setattr("cli.save_config_value", lambda *a, **k: True)
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)

    def _occupied():
        raise OccupyingLlmError(
            "Another LLM is already running (Ollama on http://127.0.0.1:11434). "
            "Stop it so managed vLLM can use the GPU.")

    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", _occupied)
    started = []
    monkeypatch.setattr(
        "hermes_cli.local_engines.ensure_managed_engine",
        lambda *a, **k: started.append("start") or "sup")

    rc = cmd_local(SimpleNamespace(local_command="start"))
    assert rc == 1
    assert started == []
    err = capsys.readouterr().err
    assert "Another LLM is already running" in err
    assert "Stop it so managed vLLM" in err


def test_local_use_and_recommend_apply_write_config_not_env(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    rec = recommend_vllm(total_bytes=24 * _GIB)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)

    rc = cmd_local(SimpleNamespace(local_command="recommend", apply=True))
    assert rc == 0
    from hermes_cli.config import load_config

    cfg = load_config()
    vllm = cfg["local_runtime"]["vllm"]
    assert "/" in vllm["model"]
    assert int(vllm["max_model_len"]) >= rec.max_model_len
    assert not (home / ".env").exists()

    url = cmd_local(SimpleNamespace(local_command="use"))
    assert url == 0
    cfg = load_config()
    assert cfg["model"]["provider"] == "vllm"
    assert str(cfg["model"]["base_url"]).endswith("/v1")
    host = str(cfg["model"]["base_url"]).split("://", 1)[-1].split(":")[0]
    assert host in ("127.0.0.1", "localhost")

    from cli import save_config_value

    save_config_value("model.base_url", "http://gpu-box.example:8000/v1")
    assert cmd_local(SimpleNamespace(local_command="use")) == 0
    assert load_config()["model"]["base_url"] == "http://gpu-box.example:8000/v1"
    assert not (home / ".env").exists()


def test_local_stop_kills_state_pid_only(tmp_path, monkeypatch):
    """Stop frees the managed pid from server.json; an unrelated process stays up."""
    _home(tmp_path, monkeypatch)
    import hermes_cli.vllm_runtime.bootstrap as boot
    from hermes_cli.vllm_runtime.supervisor import state_path

    monkeypatch.setattr(boot, "_SUPERVISOR", None)
    ours = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"])
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        path = state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"pid": %d, "port": 18435, "base_url": "http://127.0.0.1:18435/v1"}' % ours.pid,
            encoding="utf-8")
        rc = cmd_local(SimpleNamespace(local_command="stop"))
        assert rc == 0
        ours.wait(timeout=20)
        assert ours.poll() is not None
        assert other.poll() is None
        assert not path.exists()
    finally:
        for proc in (ours, other):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)


def test_doctor_local_row_reports_engine(tmp_path, monkeypatch, capsys):
    _home(tmp_path, monkeypatch)
    from hermes_cli.doctor_local import _check_managed_local_engine

    finding = _check_managed_local_engine(False)
    out = capsys.readouterr().out
    assert "engine: vllm" in out
    assert finding.issues  # venv missing while engine is vllm is the install contract
    assert any("hermes local install" in issue for issue in finding.issues)
