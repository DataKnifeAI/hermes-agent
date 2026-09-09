"""Engine-aware Local Models HTTP contract (managed vLLM Phase 3).

Behaviour, not snapshots: status.engine, llama-route guards, set-engine
stop-other order, start/stop dispatch, occupancy copy. Temp HERMES_HOME
only — no live GPU, no GGUF I/O when engine is vllm.
"""

from __future__ import annotations

import yaml
from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    from hermes_cli import web_server

    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client, home


def _write_engine(home, engine: str, extra=None):
    section = {"enabled": True, "engine": engine}
    if extra:
        section.update(extra)
    (home / "config.yaml").write_text(
        yaml.dump({"local_runtime": section, "model": {}}), encoding="utf-8")


def test_status_default_engine_is_llamacpp(tmp_path, monkeypatch):
    client, _home = _client(tmp_path, monkeypatch)
    data = client.get("/api/local-models/status").json()
    assert data["engine"] == "llamacpp"
    assert isinstance(data["server_running"], bool)


def test_status_reports_vllm_not_llama_gguf(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: {"base_url": "http://127.0.0.1:18435/v1", "pid": 1})
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.occupancy_payload",
        lambda: {"occupancy": [], "occupancy_message": None})

    from hermes_cli.local_runtime.bootstrap import models_dir

    models_dir().mkdir(parents=True, exist_ok=True)
    (models_dir() / "Should-Not-Appear.gguf").write_bytes(b"GGUF" + b"\x00" * 64)

    data = client.get("/api/local-models/status").json()
    assert data["engine"] == "vllm"
    assert data["server_running"] is True
    assert data["server_base_url"].endswith("/v1")
    assert data["venv_ready"] is True
    assert "Should-Not-Appear" not in str(data)
    assert all("/" in (m.get("id") or "") for m in data.get("models") or [])


def test_llama_routes_refuse_when_engine_vllm_without_gguf_io(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    from hermes_cli.local_runtime.bootstrap import models_dir

    models_dir().mkdir(parents=True, exist_ok=True)
    before = {p.name for p in models_dir().iterdir()}

    catalog = client.get("/api/local-models/catalog")
    assert catalog.status_code == 400
    assert "llama.cpp" in catalog.json()["detail"]

    for path, body in (
        ("/api/local-models/quickstart", {}),
        ("/api/local-models/sideload", {"path": "/tmp/x.gguf"}),
        ("/api/local-models/eject", {"model_id": "x"}),
        ("/api/local-models/download", {"model_id": "x"}),
        ("/api/local-models/runtime/install", {}),
    ):
        r = client.post(path, json=body)
        assert r.status_code == 400, (path, r.status_code, r.text)
        assert "llama.cpp" in r.json()["detail"]

    after = {p.name for p in models_dir().iterdir()}
    assert after == before


def test_set_engine_persists_without_stopping_the_other(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    order: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.local_engines.stop_llama_engine",
        lambda: order.append("stop_llama"))
    monkeypatch.setattr(
        "hermes_cli.local_engines.stop_vllm_engine",
        lambda: order.append("stop_vllm"))

    r = client.post("/api/local-models/engine", json={"engine": "vllm"})
    assert r.status_code == 200
    assert r.json()["engine"] == "vllm"
    assert order == []
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["engine"] == "vllm"

    r = client.post("/api/local-models/engine", json={"engine": "llamacpp"})
    assert r.status_code == 200
    assert order == []
    assert load_config()["local_runtime"]["engine"] == "llamacpp"


def test_server_start_vllm_stops_llama_then_starts(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    order: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.local_engines.stop_llama_engine",
        lambda: order.append("stop_llama"))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.occupancy.require_gpu_free",
        lambda: order.append("occupancy"))

    class _Sup:
        base_url = "http://127.0.0.1:9/v1"

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: order.append("start_vllm") or _Sup())
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:9/v1")

    r = client.post("/api/local-models/server", json={"action": "start"})
    assert r.status_code == 200, r.text
    assert order == ["stop_llama", "occupancy", "start_vllm"]


def test_server_start_occupancy_surfaces(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    from hermes_cli.vllm_runtime.occupancy import OccupyingLlmError

    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)

    def _occupied():
        raise OccupyingLlmError(
            "Another LLM is already running (Ollama on http://127.0.0.1:11434). "
            "Stop it so managed vLLM can use the GPU.")

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.occupancy.require_gpu_free", _occupied)
    started = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: started.append("start") or None)

    r = client.post("/api/local-models/server", json={"action": "start"})
    assert r.status_code == 409
    assert "Another LLM is already running" in r.json()["detail"]
    assert "Stop it so managed vLLM" in r.json()["detail"]
    assert started == []


def test_server_stop_dispatches_to_configured_engine(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    stopped: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.local_engines.stop_configured_engine",
        lambda cfg=None: stopped.append("vllm") or "vllm")

    r = client.post("/api/local-models/server", json={"action": "stop"})
    assert r.status_code == 200
    assert stopped == ["vllm"]


def test_vllm_recommend_and_use_routes(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    rec = client.get("/api/local-models/vllm/recommend")
    assert rec.status_code == 200
    body = rec.json()
    assert "feasible" in body and "model" in body
    assert "/" in body["model"] or body["model"]

    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)

    class _Sup:
        base_url = "http://127.0.0.1:9/v1"

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: _Sup())
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:9/v1")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bench.verify_tool_calls",
        lambda *a, **k: {"ok": True, "tool_calls": True})

    used = client.post("/api/local-models/vllm/use")
    assert used.status_code == 200
    assert used.json()["base_url"].endswith("/v1")


def test_vllm_list_set_delete_and_search_contracts(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    hub = tmp_path / "hf-hub"
    cached = hub / "models--solidrust--Hermes-3-Llama-3.1-8B-AWQ"
    (cached / "snapshots" / "abc").mkdir(parents=True)
    (cached / "snapshots" / "abc" / "model.safetensors").write_bytes(b"x" * 64)
    extra = hub / "models--acme--sideload-awq"
    extra.mkdir(parents=True)
    (extra / "weights.bin").write_bytes(b"y" * 32)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))

    listed = client.get("/api/local-models/vllm/models")
    assert listed.status_code == 200
    ids = {m["id"] for m in listed.json()["models"]}
    assert "solidrust/Hermes-3-Llama-3.1-8B-AWQ" in ids
    assert "acme/sideload-awq" in ids
    assert all("/" in mid for mid in ids)

    set_r = client.post(
        "/api/local-models/vllm/set", json={"model": "acme/sideload-awq"})
    assert set_r.status_code == 200
    assert set_r.json()["model"] == "acme/sideload-awq"
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["vllm"]["model"] == "acme/sideload-awq"

    gone = client.delete("/api/local-models/vllm/models/acme/sideload-awq")
    assert gone.status_code == 200
    assert not extra.exists()
    assert cached.is_dir()
    missing = client.delete("/api/local-models/vllm/models/no/such-model")
    assert missing.status_code == 404

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda url: [
            {"id": "NousResearch/Hermes-3-Llama-3.1-8B", "downloads": 9,
             "likes": 2, "lastModified": "2026-01-01", "gated": False, "tags": []},
            {"id": "someone/Qwen-GGUF", "downloads": 99, "likes": 1,
             "lastModified": "", "gated": False, "tags": ["gguf"]},
        ])
    search = client.get("/api/local-models/vllm/search?q=hermes")
    assert search.status_code == 200
    repos = [h["repo"] for h in search.json()["hits"]]
    assert "NousResearch/Hermes-3-Llama-3.1-8B" in repos
    assert all("gguf" not in r.lower() for r in repos)


def test_switch_engine_then_status_still_shows_running_vllm(tmp_path, monkeypatch):
    """Dropdown persist is a view: vLLM stays up and status still reports it."""
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm", extra={"vllm": {
        "model": "solidrust/Hermes-3-Llama-3.1-8B-AWQ",
        "served_model_name": "hermes3:8b",
    }})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: {"base_url": "http://127.0.0.1:18435/v1", "pid": 7})
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.occupancy_payload",
        lambda: {"occupancy": [], "occupancy_message": None})
    stopped: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.local_engines.stop_vllm_engine",
        lambda: stopped.append("vllm"))

    assert client.post("/api/local-models/engine", json={"engine": "llamacpp"}).status_code == 200
    assert stopped == []
    assert client.post("/api/local-models/engine", json={"engine": "vllm"}).status_code == 200
    data = client.get("/api/local-models/status").json()
    assert data["engine"] == "vllm"
    assert data["server_running"] is True
    assert data["served_model_name"] == "hermes3:8b"
    assert data["server_base_url"].endswith("/v1")
    assert stopped == []
