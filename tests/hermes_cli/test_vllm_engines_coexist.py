"""GPU vLLM and CPU vLLM are independent managed servers.

llama.cpp and GPU vLLM still share the GPU. Starting one does not SIGKILL
the other vLLM. Occupancy treats the sibling serve (and its EngineCore
children) as ours. Turn off stops only the selected engine.
"""

from __future__ import annotations

import json


def _reset_supervisors():
    import hermes_cli.vllm_runtime.bootstrap as boot

    boot._SUPERVISORS["gpu"] = None
    boot._SUPERVISORS["cpu"] = None


class _Sup:
    def __init__(self, name: str, stopped: list[str]):
        self.name = name
        self.device = name
        self.settings = {"served_model_name": name}
        self._stopped = stopped

    def stop(self) -> None:
        self._stopped.append(self.name)


def test_starting_gpu_does_not_stop_live_cpu(monkeypatch):
    import hermes_cli.vllm_runtime.bootstrap as boot
    from hermes_cli.local_engines import ensure_managed_engine

    stopped: list[str] = []
    _reset_supervisors()
    boot._SUPERVISORS["cpu"] = _Sup("cpu", stopped)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.shutdown_local_runtime",
        lambda: stopped.append("llama"))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: "gpu-sup")
    try:
        ensure_managed_engine(
            {"local_runtime": {"enabled": True, "engine": "vllm"}}, force=True)
        assert "cpu" not in stopped
        assert boot.get_supervisor("cpu") is not None
        assert "llama" in stopped
    finally:
        _reset_supervisors()


def test_starting_cpu_device_does_not_stop_gpu_or_llama(monkeypatch):
    """``local_runtime.vllm.device: cpu`` is the same server as legacy vllm-cpu."""
    import hermes_cli.vllm_runtime.bootstrap as boot
    from hermes_cli.local_engines import ensure_managed_engine

    stopped: list[str] = []
    _reset_supervisors()
    boot._SUPERVISORS["gpu"] = _Sup("gpu", stopped)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.shutdown_local_runtime",
        lambda: stopped.append("llama"))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: "cpu-sup")
    try:
        ensure_managed_engine({
            "local_runtime": {
                "enabled": True,
                "engine": "vllm",
                "vllm": {"device": "cpu"},
            },
        }, force=True)
        assert stopped == []
        assert boot.get_supervisor("gpu") is not None
    finally:
        _reset_supervisors()


def test_starting_cpu_does_not_stop_gpu_or_llama(monkeypatch):
    import hermes_cli.vllm_runtime.bootstrap as boot
    from hermes_cli.local_engines import ensure_managed_engine

    stopped: list[str] = []
    _reset_supervisors()
    boot._SUPERVISORS["gpu"] = _Sup("gpu", stopped)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.shutdown_local_runtime",
        lambda: stopped.append("llama"))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: "cpu-sup")
    try:
        ensure_managed_engine(
            {"local_runtime": {"enabled": True, "engine": "vllm-cpu"}}, force=True)
        assert stopped == []
        assert boot.get_supervisor("gpu") is not None
    finally:
        _reset_supervisors()


def test_starting_llamacpp_stops_gpu_vllm_and_leaves_cpu(monkeypatch):
    import hermes_cli.vllm_runtime.bootstrap as boot
    from hermes_cli.local_engines import ensure_managed_engine

    stopped: list[str] = []
    _reset_supervisors()
    boot._SUPERVISORS["gpu"] = _Sup("gpu", stopped)
    boot._SUPERVISORS["cpu"] = _Sup("cpu", stopped)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.ensure_local_runtime",
        lambda *a, **k: "llama")
    try:
        ensure_managed_engine(
            {"local_runtime": {"enabled": True, "engine": "llamacpp"}}, force=True)
        assert stopped == ["gpu"]
        assert boot.get_supervisor("cpu") is not None
        assert boot.get_supervisor("gpu") is None
    finally:
        _reset_supervisors()


def test_turn_off_one_vllm_leaves_the_other(monkeypatch):
    import hermes_cli.vllm_runtime.bootstrap as boot
    from hermes_cli.local_engines import stop_configured_engine

    stopped: list[str] = []
    _reset_supervisors()
    boot._SUPERVISORS["gpu"] = _Sup("gpu", stopped)
    boot._SUPERVISORS["cpu"] = _Sup("cpu", stopped)
    try:
        name = stop_configured_engine({"local_runtime": {"engine": "vllm"}})
        assert name == "vllm"
        assert stopped == ["gpu"]
        assert boot.get_supervisor("cpu") is not None
        stopped.clear()
        name = stop_configured_engine({"local_runtime": {"engine": "vllm-cpu"}})
        assert name == "vllm"
        assert stopped == ["cpu"]
    finally:
        _reset_supervisors()


def test_occupancy_ignores_sibling_managed_serve(tmp_path, monkeypatch):
    """GPU scan must not treat the CPU serve (or its EngineCore) as foreign, and the reverse."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)
    from hermes_cli.vllm_runtime.occupancy import discover_occupying_llms
    from hermes_cli.vllm_runtime.supervisor import state_path

    gpu_parent, gpu_child = 4100, 4101
    cpu_parent, cpu_child = 4200, 4201
    ours = {gpu_parent, gpu_child, cpu_parent, cpu_child}
    try:
        for device, pid, port in (
            ("gpu", gpu_parent, 18435),
            ("cpu", cpu_parent, 18436),
        ):
            path = state_path(device)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "pid": pid,
                "port": port,
                "base_url": f"http://127.0.0.1:{port}/v1",
            }), encoding="utf-8")

        monkeypatch.setattr(
            "hermes_cli.vllm_runtime.occupancy._pid_alive",
            lambda pid: pid in ours)

        def _children(pid: int) -> set[int]:
            if pid == gpu_parent:
                return {gpu_child}
            if pid == cpu_parent:
                return {cpu_child}
            return set()

        monkeypatch.setattr(
            "hermes_cli.vllm_runtime.occupancy.descendant_pids", _children)

        def _probe(port: int):
            if port in (18435, 18436):
                raise AssertionError(f"managed vLLM port {port} was scanned as a foreign LLM")
            return None

        monkeypatch.setattr(
            "hermes_cli.vllm_runtime.occupancy.probe_foreign_http", _probe)

        hits = discover_occupying_llms(
            ports=(18435, 18436),
            gpu_rows=[
                (gpu_child, "VLLM::EngineCore"),
                (cpu_child, "VLLM::EngineCore"),
                (999, "VLLM::EngineCore"),
            ],
        )
        assert [h for h in hits if h.pid in ours or h.port in (18435, 18436)] == []
        assert any(h.pid == 999 for h in hits)
    finally:
        hermes_constants._default_hermes_root_memo = None


def test_cpu_compression_base_url_is_written_unless_user_set(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    from cli import save_config_value
    from hermes_cli.config import load_config
    from hermes_cli.local_engines import maybe_bind_cpu_compression

    url = "http://127.0.0.1:18436/v1"
    assert maybe_bind_cpu_compression(url, "qwen3:4b") is True
    comp = load_config()["auxiliary"]["compression"]
    assert comp["base_url"] == url
    assert comp["model"] == "qwen3:4b"
    assert comp["provider"] == "custom"
    assert ":8000" not in comp["base_url"]
    assert ":8080" not in comp["base_url"]

    save_config_value("auxiliary.compression.base_url", "http://aux.example/v1")
    save_config_value("auxiliary.compression.model", "other-model")
    assert maybe_bind_cpu_compression(url, "qwen3:4b") is False
    kept = load_config()["auxiliary"]["compression"]
    assert kept["base_url"] == "http://aux.example/v1"
    assert kept["model"] == "other-model"


def test_cpu_compression_skips_when_only_model_is_set(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)
    (home / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    model: my-summarizer\n",
        encoding="utf-8",
    )
    from hermes_cli.config import load_config
    from hermes_cli.local_engines import maybe_bind_cpu_compression

    assert maybe_bind_cpu_compression("http://127.0.0.1:18436/v1", "qwen3:4b") is False
    comp = load_config()["auxiliary"]["compression"]
    assert comp["model"] == "my-summarizer"
    assert not str(comp.get("base_url") or "").strip()
