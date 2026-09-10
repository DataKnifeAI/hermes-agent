"""Engine-aware Local Models HTTP contract (managed vLLM Phase 3).

Behaviour, not snapshots: status.engine, llama-route guards, set-engine
stop-other order, start/stop dispatch, occupancy copy. Temp HERMES_HOME
only — no live GPU, no GGUF I/O when engine is vllm.
"""

from __future__ import annotations

import urllib.error
from io import BytesIO

import yaml
from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Default vLLM model is Qwen/Qwen3-8B-AWQ. Isolate the hub so Start/Use
    # cannot see a hollow leftover in the developer's ~/.cache/huggingface.
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr("cli._hermes_home", home)
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    # Use/quickstart must not dial huggingface.co from the suite.
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.gated_repo_reason", lambda hid: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.hf_repo_access_issue", lambda hid: None)
    # Occupancy probes the host GPU/ports. These tests must not see a leftover
    # Nemotron on the developer's 18435.
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)
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


def _seed_default_vllm_cache(tmp_path, monkeypatch, repo="Qwen/Qwen3-8B-AWQ"):
    """Tiny weight file so Start sees the DEFAULT_CONFIG model as downloaded."""
    hub = tmp_path / "hf-hub"
    dest = hub / ("models--" + repo.replace("/", "--"))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "model.safetensors").write_bytes(b"x" * 64)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    return hub


def test_status_default_engine_is_llamacpp(tmp_path, monkeypatch):
    client, _home = _client(tmp_path, monkeypatch)
    data = client.get("/api/local-models/status").json()
    assert data["engine"] == "llamacpp"
    assert isinstance(data["server_running"], bool)
    assert data.get("vllm_version") in (None, "")


def test_status_vllm_version_from_isolated_venv(tmp_path, monkeypatch):
    """Status carries the isolated-venv package version — not Hermes on PATH."""
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    reported = "9.9.9"
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.installed_vllm_version", lambda: reported)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.running_served_model_name",
        lambda: None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.occupancy_payload",
        lambda: {"occupancy": [], "occupancy_message": None})

    data = client.get("/api/local-models/status").json()
    assert data["engine"] == "vllm"
    assert data["vllm_version"] == reported
    assert data["vllm_version"] == data["tag"]


def test_status_vllm_version_null_when_venv_missing(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.venv_ready", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.installed_vllm_version", lambda: "")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.running_served_model_name",
        lambda: None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.occupancy_payload",
        lambda: {"occupancy": [], "occupancy_message": None})

    data = client.get("/api/local-models/status").json()
    assert data["engine"] == "vllm"
    assert data.get("vllm_version") in (None, "")


def test_hardware_vllm_version_from_isolated_venv(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    reported = "9.9.9"
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.installed_vllm_version", lambda: reported)

    data = client.get("/api/local-models/hardware").json()
    assert data["engine"] == "vllm"
    assert data["vllm_version"] == reported


def test_hardware_vllm_version_null_when_not_installed(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.installed_vllm_version", lambda: "")

    data = client.get("/api/local-models/hardware").json()
    assert data["engine"] == "vllm"
    assert data.get("vllm_version") in (None, "")


def test_status_reports_vllm_not_llama_gguf(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: {"base_url": "http://127.0.0.1:18435/v1", "pid": 1})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.running_served_model_name",
        lambda: "qwen3:8b")
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

    search = client.get("/api/local-models/search", params={"q": "qwen"})
    assert search.status_code == 400
    assert "llama.cpp" in search.json()["detail"]

    for path, body in (
        ("/api/local-models/sideload", {"path": "/tmp/x.gguf"}),
        ("/api/local-models/eject", {"model_id": "x"}),
        ("/api/local-models/download", {"model_id": "x"}),
        ("/api/local-models/activate", {"model_id": "x"}),
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
    _seed_default_vllm_cache(tmp_path, monkeypatch)
    order: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.local_engines.stop_llama_engine",
        lambda: order.append("stop_llama"))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.occupancy.require_gpu_free",
        lambda: order.append("occupancy"))

    class _Sup:
        base_url = "http://127.0.0.1:9/v1"

    captured = {}
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: captured.update(k) or order.append("start_vllm") or _Sup())
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:9/v1")

    r = client.post("/api/local-models/server", json={"action": "start"})
    assert r.status_code == 200, r.text
    assert order == ["stop_llama", "occupancy", "start_vllm"]
    from hermes_cli.web_routers.local_models_engine import VLLM_START_TIMEOUT_S

    assert captured.get("timeout_s") == VLLM_START_TIMEOUT_S


def test_server_start_occupancy_surfaces(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    _seed_default_vllm_cache(tmp_path, monkeypatch)
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
    _seed_default_vllm_cache(tmp_path, monkeypatch)
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
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))

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


def test_delete_official_model_keeps_catalog_row(tmp_path, monkeypatch):
    """Official ids stay listed after delete so Download can run again."""
    client, home = _client(tmp_path, monkeypatch)
    hid = "Qwen/Qwen3-8B-AWQ"
    _write_engine(home, "vllm", extra={"enabled": True, "vllm": {
        "model": hid, "served_model_name": "qwen3:8b",
    }})
    hub = tmp_path / "hf-hub"
    dest = hub / "models--Qwen--Qwen3-8B-AWQ"
    dest.mkdir(parents=True)
    (dest / "model.safetensors").write_bytes(b"x" * 64)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    from hermes_cli.vllm_runtime.recommend import NvidiaProbe, TIERS, VllmRecommendation

    tier = next(t for t in TIERS if t.model == hid)
    rec = VllmRecommendation(
        NvidiaProbe(24 * (1 << 30), 24 * (1 << 30), "data"), tier, True, "ok")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)

    before = client.get("/api/local-models/vllm/models")
    assert before.status_code == 200
    row = next(m for m in before.json()["models"] if m["id"] == hid)
    assert row["cached"] is True

    gone = client.delete(f"/api/local-models/vllm/models/{hid}")
    assert gone.status_code == 200, gone.text
    assert not dest.exists()

    after = client.get("/api/local-models/vllm/models")
    assert after.status_code == 200
    ids = [m["id"] for m in after.json()["models"]]
    assert hid in ids
    row = next(m for m in after.json()["models"] if m["id"] == hid)
    assert row["cached"] is False
    assert row["added_by_you"] is False


def test_delete_configured_model_does_not_enqueue_download(tmp_path, monkeypatch):
    """Delete drops HF cache + configured id. It must not start a download or Set up."""
    client, home = _client(tmp_path, monkeypatch)
    hid = "acme/doomed-awq"
    _write_engine(home, "vllm", extra={"enabled": True, "vllm": {
        "model": hid, "served_model_name": "doomed",
    }})
    hub = tmp_path / "hf-hub"
    dest = hub / "models--acme--doomed-awq"
    dest.mkdir(parents=True)
    (dest / "w.bin").write_bytes(b"x" * 64)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    started: list[str] = []
    jobs: list[object] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: pulled.append(repo))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.start_active_engine",
        lambda: started.append("start"))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.apply_recommend_and_install",
        lambda *a, **k: started.append("recommend"))
    monkeypatch.setattr(
        "hermes_cli.local_engines.stop_vllm_engine",
        lambda: started.append("stop"))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._spawn_job",
        lambda *a, **k: jobs.append(a))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._job",
        lambda *a, **k: jobs.append(("job", a)))

    gone = client.delete(f"/api/local-models/vllm/models/{hid}")
    assert gone.status_code == 200, gone.text
    assert not dest.exists()
    assert pulled == []
    assert jobs == []
    assert started == ["stop"]
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.supervisor import read_last_error

    cfg = load_config()
    from hermes_cli.vllm_runtime.recommend import catalog_tiers

    official = {t.model for t in catalog_tiers()}
    assert cfg["local_runtime"]["vllm"]["model"] in official
    assert cfg["local_runtime"]["vllm"]["model"] != hid
    assert cfg["local_runtime"]["enabled"] is False
    assert not (read_last_error() or "")

    from hermes_cli.vllm_runtime.recommend import NvidiaProbe, TIERS, VllmRecommendation

    rec = VllmRecommendation(
        NvidiaProbe(24 * (1 << 30), 24 * (1 << 30), "data"),
        next(t for t in TIERS if t.id == "24gb"), True, "ok")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda url: [
            {"id": "NousResearch/Hermes-3-Llama-3.1-8B", "downloads": 9,
             "likes": 2, "lastModified": "2026-01-01", "gated": False, "tags": [],
             "safetensors": {"parameters": {"BF16": 8_030_261_248},
                             "total": 8_030_261_248},
             "usedStorage": 15 * (1 << 30)},
            {"id": "someone/Qwen-GGUF", "downloads": 99, "likes": 1,
             "lastModified": "", "gated": False, "tags": ["gguf"]},
        ])
    search = client.get("/api/local-models/vllm/search?q=hermes")
    assert search.status_code == 200
    repos = [h["repo"] for h in search.json()["hits"]]
    assert "NousResearch/Hermes-3-Llama-3.1-8B" in repos
    assert all("gguf" not in r.lower() for r in repos)
    hermes = next(h for h in search.json()["hits"] if "Hermes-3" in h["repo"])
    # 8B BF16 on a 24 GB card is too-big, never a lying Fits badge.
    # 15 GiB shards + 64k KV used to price under 24 GiB (percent-of-disk).
    assert hermes["fit"] == "too-big"
    assert "64k KV" in (hermes.get("fit_detail") or "")
    # Tools come from HF tags (function-calling), not the word Hermes in the id.
    assert "tools" not in hermes["capabilities"]


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
        "hermes_cli.vllm_runtime.inventory.running_served_model_name",
        lambda: "hermes3:8b")
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


def test_vllm_status_served_name_is_running_server_not_config(tmp_path, monkeypatch):
    """In use identity is the live serve — not config written before /v1/models."""
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm", extra={"vllm": {
        "model": "nvidia/Nemotron-3-Nano-30B-A3B-BF16",
        "served_model_name": "nemotron",
    }})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.occupancy_payload",
        lambda: {"occupancy": [], "occupancy_message": None})

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: {"base_url": "http://127.0.0.1:18435/v1", "pid": 7})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.running_served_model_name",
        lambda: "qwen3:8b")
    live = client.get("/api/local-models/status").json()
    assert live["server_running"] is True
    assert live["served_model_name"] == "qwen3:8b"
    assert live["active_model_id"] == "qwen3:8b"
    assert live["model"] == "nvidia/Nemotron-3-Nano-30B-A3B-BF16"

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.running_served_model_name",
        lambda: "")
    down = client.get("/api/local-models/status").json()
    assert down["server_running"] is False
    assert down["served_model_name"] is None
    assert down["active_model_id"] is None
    assert down["model"] == "nvidia/Nemotron-3-Nano-30B-A3B-BF16"


def test_vllm_status_active_empty_until_models_200(tmp_path, monkeypatch):
    """Spawn-time supervisor state is not In use — GET /v1/models must be 200."""
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm", extra={"vllm": {
        "model": "NousResearch/Hermes-3-Llama-3.1-8B",
        "served_model_name": "Hermes-3-Llama-3.1-8B",
    }})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.occupancy_payload",
        lambda: {"occupancy": [], "occupancy_message": None})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: {"base_url": "http://127.0.0.1:18435/v1", "pid": 7})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.running_served_model_name",
        lambda: "")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.supervisor.state_served_model_name",
        lambda: "Hermes-3-Llama-3.1-8B")

    data = client.get("/api/local-models/status").json()
    assert data["server_running"] is False
    assert data["active_model_id"] is None
    assert data["served_model_name"] is None
    assert data["model"] == "NousResearch/Hermes-3-Llama-3.1-8B"


def _wait_job(client, job_id: str, timeout_s: float = 3.0) -> dict:
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = client.get(f"/api/local-models/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still running")


def _fake_hub_download(hub, repo, job=None):
    dest = hub / ("models--" + repo.replace("/", "--"))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "weights.bin").write_bytes(b"w" * 32)
    if job is not None:
        job["phase"] = "downloading"
        job["done_bytes"] = 32
        job["total_bytes"] = 32


def test_vllm_use_refuses_uncached_without_downloading(tmp_path, monkeypatch):
    """Use is llama-shaped: cached only. Download is a separate POST."""
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    hub = tmp_path / "hf-hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    started: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: pulled.append(repo) or _fake_hub_download(hub, repo, job))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: started.append("start"))

    used = client.post("/api/local-models/vllm/use", json={"model": "acme/fresh-awq"})
    assert used.status_code == 409, used.text
    assert "Download" in used.json()["detail"]
    assert pulled == []
    assert started == []
    from hermes_cli.config import load_config

    assert (load_config().get("local_runtime") or {}).get("vllm", {}).get("model") != "acme/fresh-awq"


def test_vllm_use_refuses_too_big_cached(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    from hermes_cli.vllm_runtime.recommend import NvidiaProbe, TIERS, VllmRecommendation

    tier = next(t for t in TIERS if t.id == "24gb")
    rec = VllmRecommendation(
        NvidiaProbe(24 * (1 << 30), 24 * (1 << 30), "data"), tier, True, "ok")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    hid = "NousResearch/Hermes-3-Llama-3.1-8B"
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.list_cached_repos",
        lambda: [{"id": hid, "size_bytes": 16 * (1 << 30),
                  "size_label": "16.0 GB", "cached": True}],
    )
    started: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: started.append("start"))

    used = client.post("/api/local-models/vllm/use", json={"model": hid})
    assert used.status_code == 400, used.text
    detail = used.json()["detail"].lower()
    assert "too big" in detail or "gpu" in detail
    assert started == []
    from hermes_cli.config import load_config

    assert (load_config().get("local_runtime") or {}).get("vllm", {}).get("model") != hid


def test_vllm_use_cached_sets_then_starts_without_download(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    hub = tmp_path / "hf-hub"
    cached = hub / "models--acme--sideload-awq"
    cached.mkdir(parents=True)
    (cached / "weights.bin").write_bytes(b"y" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    started: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: pulled.append(repo))
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)

    class _Sup:
        base_url = "http://127.0.0.1:9/v1"

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: started.append("start") or _Sup())
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:9/v1")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bench.verify_tool_calls",
        lambda *a, **k: {"ok": True, "tool_calls": True})

    used = client.post("/api/local-models/vllm/use", json={"model": "acme/sideload-awq"})
    assert used.status_code == 200, used.text
    body = used.json()
    assert body["needs_download"] is False
    assert body["already_downloaded"] is True
    assert body["base_url"].endswith("/v1")
    assert pulled == []
    assert started == ["start"]
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["vllm"]["model"] == "acme/sideload-awq"


def test_vllm_download_job_and_cached_noop(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    hub = tmp_path / "hf-hub"
    cached = hub / "models--acme--sideload-awq"
    cached.mkdir(parents=True)
    (cached / "weights.bin").write_bytes(b"y" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: pulled.append(repo) or _fake_hub_download(hub, repo, job))

    skip = client.post("/api/local-models/vllm/download", json={"model": "acme/sideload-awq"})
    assert skip.status_code == 200
    assert skip.json()["already_downloaded"] is True
    assert skip.json()["job_id"] is None
    assert pulled == []

    started: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: started.append("start"))

    fresh = client.post("/api/local-models/vllm/download", json={"model": "acme/fresh-awq"})
    assert fresh.status_code == 200
    assert fresh.json()["already_downloaded"] is False
    job = _wait_job(client, fresh.json()["job_id"])
    assert job["status"] == "done", job
    assert pulled == ["acme/fresh-awq"]
    assert started == []
    from hermes_cli.config import load_config

    # Prefetch only — does not become the default (llama download neither).
    assert (load_config().get("local_runtime") or {}).get("vllm", {}).get("model") != "acme/fresh-awq"


def test_vllm_install_downloads_recommended_weights(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    hub = tmp_path / "hf-hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.ensure_vllm_venv", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: pulled.append(repo))

    r = client.post("/api/local-models/vllm/install")
    assert r.status_code == 200
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert pulled and all("/" in mid for mid in pulled)


def test_vllm_search_fit_tags_never_lie(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    from hermes_cli.vllm_runtime.recommend import NvidiaProbe, TIERS, VllmRecommendation

    tier = next(t for t in TIERS if t.id == "24gb")
    rec = VllmRecommendation(NvidiaProbe(24 * (1 << 30), 24 * (1 << 30), "data"), tier, True, "ok")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    seen_urls: list[str] = []

    def _fake_hf(url):
        seen_urls.append(url)
        return [
            {"id": "Qwen/Qwen3-8B-AWQ", "downloads": 9, "likes": 2,
             "lastModified": "", "gated": False, "tags": ["awq", "instruct"]},
            {"id": "Qwen/Qwen3-14B-AWQ", "downloads": 5, "likes": 1,
             "lastModified": "", "gated": False, "tags": ["awq"]},
            {"id": "someone/mystery-weights", "downloads": 1, "likes": 0,
             "lastModified": "", "gated": False, "tags": []},
            {"id": "Qwen/Qwen3-32B-AWQ", "downloads": 3, "likes": 0,
             "lastModified": "", "gated": False, "tags": ["awq"]},
            {"id": "org/finetune-awq", "downloads": 2, "likes": 0,
             "lastModified": "", "gated": False, "tags": ["awq", "4-bit"],
             "safetensors": {"total": 8_000_000_000,
                             "parameters": {"I32": 7_000_000_000, "BF16": 1_000_000_000}}},
        ]

    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory._hf_json", _fake_hf)

    search = client.get("/api/local-models/vllm/search?q=qwen")
    assert search.status_code == 200
    assert seen_urls and "expand=safetensors" in seen_urls[0]
    assert "expand=cardData" in seen_urls[0]
    assert "expand=createdAt" in seen_urls[0]
    by_repo = {h["repo"]: h for h in search.json()["hits"]}
    assert by_repo["Qwen/Qwen3-8B-AWQ"]["fit"] == "fits-gpu"
    assert "awq" in by_repo["Qwen/Qwen3-8B-AWQ"]["capabilities"]
    assert "instruct" in by_repo["Qwen/Qwen3-8B-AWQ"]["capabilities"]
    # 24 GB card recommends the 24gb tier (14B), not the 16gb 8B row.
    assert by_repo["Qwen/Qwen3-8B-AWQ"]["recommended"] is False
    assert by_repo["Qwen/Qwen3-14B-AWQ"]["recommended"] is True
    assert by_repo["someone/mystery-weights"]["fit"] == "unknown"
    assert by_repo["Qwen/Qwen3-32B-AWQ"]["fit"] == "too-big"
    # No 8B in the id — safetensors.total + AWQ tag still prices a 24 GB card.
    assert by_repo["org/finetune-awq"]["fit"] == "fits-gpu"


def _feasible_rec(monkeypatch, *, feasible=True):
    from hermes_cli.vllm_runtime.recommend import NvidiaProbe, TIERS, VllmRecommendation

    tier = next(t for t in TIERS if t.id == "24gb")
    rec = VllmRecommendation(
        NvidiaProbe(24 * (1 << 30), 24 * (1 << 30), "data"),
        tier, feasible, "ok" if feasible else "vram_below_64k_floor")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    return rec


def test_vllm_quickstart_installs_downloads_starts_activates(tmp_path, monkeypatch):
    """Set up for me: recommend → venv → HF weights → serve → provider. No live HF."""
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    rec = _feasible_rec(monkeypatch)
    hub = tmp_path / "hf-hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    calls: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.venv_ready", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.ensure_vllm_venv",
        lambda *a, **k: calls.append("install"))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: calls.append("download") or _fake_hub_download(hub, repo, job))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.start_active_engine",
        lambda: calls.append("start"))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.activate_vllm",
        lambda: calls.append("activate") or {"ok": True, "base_url": "http://127.0.0.1:9/v1"})

    r = client.post("/api/local-models/quickstart", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["needs_runtime"] is True
    assert body["needs_download"] is True
    assert body["model_id"] == rec.model
    job = _wait_job(client, body["job_id"])
    assert job["status"] == "done", job
    assert job["kind"] == "quickstart"
    assert calls[0] == "install"
    assert calls.index("install") < calls.index("download") < calls.index("start")
    assert calls[-1] == "activate"


def test_vllm_quickstart_skips_satisfied_legs(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    rec = _feasible_rec(monkeypatch)
    _write_engine(home, "vllm", extra={"vllm": {"model": rec.model}})
    hub = tmp_path / "hf-hub"
    cached = hub / ("models--" + rec.model.replace("/", "--"))
    cached.mkdir(parents=True)
    (cached / "weights.bin").write_bytes(b"w" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    calls: list[str] = []
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.ensure_vllm_venv",
        lambda *a, **k: calls.append("install"))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: calls.append("download"))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.start_active_engine",
        lambda: calls.append("start"))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.activate_vllm",
        lambda: calls.append("activate") or {"ok": True, "base_url": "http://127.0.0.1:9/v1"})

    r = client.post("/api/local-models/quickstart", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["needs_runtime"] is False
    assert body["needs_download"] is False
    job = _wait_job(client, body["job_id"])
    assert job["status"] == "done", job
    assert "download" not in calls
    assert calls[-2:] == ["start", "activate"] or calls[-1] == "activate"


def test_vllm_quickstart_no_probe_uses_16gb_default(tmp_path, monkeypatch):
    """Missing VRAM probe still plans the shipped 16 GB id — not a 409."""
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    from hermes_cli.vllm_runtime.recommend import NvidiaProbe, VllmRecommendation, recommend_vllm

    none = recommend_vllm(total_bytes=0)
    rec = VllmRecommendation(NvidiaProbe(0, 0, "none", error="no_nvidia"), None, False, "no_nvidia")
    monkeypatch.setattr("hermes_cli.vllm_runtime.recommend.recommend_vllm", lambda **k: rec)
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.ensure_vllm_venv", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.start_active_engine", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.activate_vllm",
        lambda: {"ok": True, "base_url": "http://127.0.0.1:9/v1"})
    r = client.post("/api/local-models/quickstart", json={})
    assert r.status_code == 200, r.text
    assert r.json()["model_id"] == none.model
    assert "/" in r.json()["model_id"]


def test_vllm_quickstart_refuses_when_gpu_infeasible(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    _feasible_rec(monkeypatch, feasible=False)
    r = client.post("/api/local-models/quickstart", json={})
    assert r.status_code == 409
    assert "64k" in r.json()["detail"]


def test_vllm_quickstart_ignores_leftover_gated_search_hit(tmp_path, monkeypatch):
    """Set up for me downloads the official recommend, never leftover Gemma."""
    client, home = _client(tmp_path, monkeypatch)
    rec = _feasible_rec(monkeypatch)
    leftover = "google/gemma-3-27b-it"
    _write_engine(home, "vllm", extra={"vllm": {
        "model": leftover, "served_model_name": "gemma",
    }})
    hub = tmp_path / "hf-hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.ensure_vllm_venv", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: pulled.append(repo) or _fake_hub_download(hub, repo, job))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.start_active_engine", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.activate_vllm",
        lambda: {"ok": True, "base_url": "http://127.0.0.1:9/v1"})

    r = client.post("/api/local-models/quickstart", json={"model_id": leftover})
    assert r.status_code == 200, r.text
    assert r.json()["model_id"] == rec.model
    assert leftover not in r.json()["model_id"]
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert pulled == [rec.model]
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["vllm"]["model"] == rec.model


def test_delete_last_cached_resets_leftover_gemma_config(tmp_path, monkeypatch):
    """Deleting the last hub dir must not leave gated Gemma as the configured id."""
    client, home = _client(tmp_path, monkeypatch)
    leftover = "google/gemma-3-27b-it"
    cached_id = "acme/sideload-awq"
    rec = _feasible_rec(monkeypatch)
    _write_engine(home, "vllm", extra={"enabled": True, "vllm": {
        "model": leftover, "served_model_name": "gemma",
    }})
    hub = tmp_path / "hf-hub"
    dest = hub / "models--acme--sideload-awq"
    dest.mkdir(parents=True)
    (dest / "weights.bin").write_bytes(b"x" * 64)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))

    listed = client.get("/api/local-models/vllm/models")
    assert listed.status_code == 200
    ids = {m["id"] for m in listed.json()["models"]}
    assert leftover not in ids
    assert cached_id in ids

    gone = client.delete(f"/api/local-models/vllm/models/{cached_id}")
    assert gone.status_code == 200, gone.text
    from hermes_cli.config import load_config

    cfg = load_config()
    assert cfg["local_runtime"]["vllm"]["model"] == rec.model
    assert leftover not in (cfg["local_runtime"]["vllm"].get("model") or "")
    assert cfg["local_runtime"]["enabled"] is False
    after = client.get("/api/local-models/vllm/models")
    after_ids = {m["id"] for m in after.json()["models"]}
    assert leftover not in after_ids
    assert rec.model in after_ids


def test_vllm_quickstart_empty_body_ignores_leftover_nemotron(tmp_path, monkeypatch):
    """Set up for me (null/empty body) still plans the official recommend."""
    client, home = _client(tmp_path, monkeypatch)
    rec = _feasible_rec(monkeypatch)
    leftover = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"
    _write_engine(home, "vllm", extra={"vllm": {
        "model": leftover, "served_model_name": "nemotron",
    }})
    hub = tmp_path / "hf-hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.ensure_vllm_venv", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: pulled.append(repo) or _fake_hub_download(hub, repo, job))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.start_active_engine", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.activate_vllm",
        lambda: {"ok": True, "base_url": "http://127.0.0.1:9/v1"})

    for body in ({}, {"model_id": None}):
        r = client.post("/api/local-models/quickstart", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["model_id"] == rec.model
        assert leftover not in r.json()["model_id"]
        job = _wait_job(client, r.json()["job_id"])
        assert job["status"] == "done", job

    assert rec.model in pulled
    assert leftover not in pulled
    llama = client.get("/api/local-models/catalog")
    assert llama.status_code == 400
    assert "llama.cpp" in llama.json()["detail"]
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["vllm"]["model"] == rec.model


def test_vllm_quickstart_stops_leftover_running_id_starts_official(tmp_path, monkeypatch):
    """Restore recommended setup: leftover Nemotron still serving must be stopped,
    then official recommend starts with that row's argv — not leftover flags."""
    client, home = _client(tmp_path, monkeypatch)
    rec = _feasible_rec(monkeypatch)
    leftover = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"
    _write_engine(home, "vllm", extra={"enabled": True, "vllm": {
        "model": leftover,
        "served_model_name": "nemotron",
        "quantization": "",
        "max_model_len": 8192,
        "tool_call_parser": "llama3_json",
        "kv_cache_dtype": "fp8",
    }})
    hub = tmp_path / "hf-hub"
    dest = hub / ("models--" + rec.model.replace("/", "--"))
    dest.mkdir(parents=True)
    (dest / "weights.bin").write_bytes(b"w" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    order: list = []

    def _stop():
        order.append("stop")

    class _Sup:
        base_url = "http://127.0.0.1:9/v1"

    def _ensure(cfg=None, force=False, **k):
        from hermes_cli.config import load_config
        from hermes_cli.vllm_runtime.supervisor import vllm_settings

        settings = vllm_settings(cfg if cfg is not None else load_config())
        order.append(("start", settings.get("model"),
                      settings.get("quantization"),
                      int(settings.get("max_model_len") or 0)))
        return _Sup()

    occupied: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.occupancy.require_gpu_free",
        lambda: occupied.append("occupancy"))
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", _stop)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime", _ensure)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:9/v1")
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.ensure_vllm_venv", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: order.append(("download", repo)))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.activate_vllm",
        lambda: {"ok": True, "base_url": "http://127.0.0.1:9/v1"})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: None)

    r = client.post("/api/local-models/quickstart", json={})
    assert r.status_code == 200, r.text
    assert r.json()["model_id"] == rec.model
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert occupied  # foreign occupancy before leftover stop, and again at start
    assert order[0] == "stop"
    starts = [step for step in order if isinstance(step, tuple) and step[0] == "start"]
    assert len(starts) == 1
    assert starts[0][1] == rec.model
    assert leftover not in str(starts[0])
    assert starts[0][2] == rec.quantization
    assert starts[0][3] == rec.max_model_len
    from hermes_cli.config import load_config

    vllm = load_config()["local_runtime"]["vllm"]
    assert vllm["model"] == rec.model
    assert vllm["quantization"] == rec.quantization
    assert int(vllm["max_model_len"]) == rec.max_model_len
    assert vllm["served_model_name"] == rec.served_model_name
    assert "download" not in {step[0] for step in order if isinstance(step, tuple)}


def test_vllm_quickstart_surfaces_last_error_when_serve_dies(tmp_path, monkeypatch):
    """If ``vllm serve`` really dies, the job error is last_error — not a generic 502."""
    client, home = _client(tmp_path, monkeypatch)
    rec = _feasible_rec(monkeypatch)
    _write_engine(home, "vllm", extra={"vllm": {"model": rec.model}})
    hub = tmp_path / "hf-hub"
    dest = hub / ("models--" + rec.model.replace("/", "--"))
    dest.mkdir(parents=True)
    (dest / "weights.bin").write_bytes(b"w" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    from hermes_cli.vllm_runtime.supervisor import write_last_error

    def _ensure(*a, **k):
        write_last_error("vllm serve was killed (SIGKILL) starting Qwen/Qwen3-14B-AWQ")
        return None

    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime", _ensure)
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.ensure_vllm_venv", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: None)

    r = client.post("/api/local-models/quickstart", json={})
    assert r.status_code == 200, r.text
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "error", job
    assert "SIGKILL" in (job.get("error") or "")


def test_vllm_install_ignores_leftover_gated_id(tmp_path, monkeypatch):
    """Install job downloads official recommend, never leftover Nemotron 401."""
    client, home = _client(tmp_path, monkeypatch)
    rec = _feasible_rec(monkeypatch)
    leftover = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"
    _write_engine(home, "vllm", extra={"vllm": {
        "model": leftover, "served_model_name": "nemotron",
    }})
    hub = tmp_path / "hf-hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.ensure_vllm_venv", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: pulled.append(repo))

    r = client.post("/api/local-models/vllm/install")
    assert r.status_code == 200
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert pulled == [rec.model]
    assert leftover not in pulled
    from hermes_cli.config import load_config as _load

    assert _load()["local_runtime"]["vllm"]["model"] == rec.model


def test_vllm_download_keeps_explicit_search_hit(tmp_path, monkeypatch):
    """Download of an explicit leftover/search-hit id is Use-path, not failsafe."""
    client, home = _client(tmp_path, monkeypatch)
    leftover = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"
    _write_engine(home, "vllm", extra={"vllm": {
        "model": leftover, "served_model_name": "nemotron",
    }})
    hub = tmp_path / "hf-hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: pulled.append(repo) or _fake_hub_download(hub, repo, job))

    r = client.post("/api/local-models/vllm/download", json={"model": leftover})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == leftover
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert pulled == [leftover]


def test_vllm_quickstart_falls_back_when_official_id_401s(tmp_path, monkeypatch):
    """If recommend's HF id is gated/401, Set up for me uses the next public official row."""
    client, home = _client(tmp_path, monkeypatch)
    rec = _feasible_rec(monkeypatch)
    leftover = "google/gemma-3-27b-it"
    _write_engine(home, "vllm", extra={"vllm": {"model": leftover}})
    from hermes_cli.vllm_runtime.inventory import GATED_DOWNLOAD_MSG
    from hermes_cli.vllm_runtime.recommend import catalog_tiers

    fallback = next(t.model for t in catalog_tiers() if t.model != rec.model)

    def _issue(hid):
        if hid == rec.model:
            return GATED_DOWNLOAD_MSG
        return None

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.hf_repo_access_issue", _issue)
    hub = tmp_path / "hf-hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    pulled: list[str] = []
    monkeypatch.setattr("hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.ensure_vllm_venv", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.download_hf_repo",
        lambda repo, job=None: pulled.append(repo) or _fake_hub_download(hub, repo, job))
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.start_active_engine", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.activate_vllm",
        lambda: {"ok": True, "base_url": "http://127.0.0.1:9/v1"})
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)

    r = client.post("/api/local-models/quickstart", json={"model_id": leftover})
    assert r.status_code == 200, r.text
    assert r.json()["model_id"] == fallback
    assert leftover not in r.json()["model_id"]
    job = _wait_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert pulled == [fallback]
    assert rec.model not in pulled
    assert "not publicly downloadable" in (job.get("detail") or "")


def test_vllm_use_local_400_is_not_502_or_hf_reject(tmp_path, monkeypatch):
    """Local vLLM 400 must stay 400 with a human string — never 502 nested Bad Request."""
    client, home = _client(tmp_path, monkeypatch)
    hid = "Qwen/Qwen3-8B-AWQ"
    _write_engine(home, "vllm")
    hub = tmp_path / "hf-hub"
    dest = hub / "models--Qwen--Qwen3-8B-AWQ"
    dest.mkdir(parents=True)
    (dest / "w.bin").write_bytes(b"q" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))

    def _boom(_repo):
        raise urllib.error.HTTPError(
            "http://127.0.0.1:40689/v1/chat/completions", 400, "Bad Request",
            hdrs=None, fp=BytesIO())

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.gated_repo_reason", _boom)

    used = client.post("/api/local-models/vllm/use", json={"model": hid})
    assert used.status_code == 400, used.text
    detail = used.json()["detail"]
    assert "502" not in used.text
    assert "Hugging Face" not in detail
    assert "Bad Request" not in detail


def test_vllm_use_official_cached_after_leftover_search_hit(tmp_path, monkeypatch):
    """Use Qwen3-8B-AWQ must 200 even when config still holds a search hit."""
    client, home = _client(tmp_path, monkeypatch)
    leftover = "nvidia/Nemotron-3-Nano-30B-A3B-BF16"
    catalog = "Qwen/Qwen3-8B-AWQ"
    _write_engine(home, "vllm", extra={"vllm": {
        "model": leftover, "served_model_name": "nemotron", "quantization": "awq",
    }})
    hub = tmp_path / "hf-hub"
    dest = hub / "models--Qwen--Qwen3-8B-AWQ"
    dest.mkdir(parents=True)
    (dest / "w.bin").write_bytes(b"q" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.gated_repo_reason", lambda hid: None)
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)
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

    used = client.post("/api/local-models/vllm/use", json={"model": catalog})
    assert used.status_code == 200, used.text
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["vllm"]["model"] == catalog


def test_vllm_use_gated_repo_is_plain_language_400(tmp_path, monkeypatch):
    """Use on a gated Gemma must not toast a raw Bad Request."""
    client, home = _client(tmp_path, monkeypatch)
    hid = "google/gemma-3-27b-it"
    _write_engine(home, "vllm")
    hub = tmp_path / "hf-hub"
    dest = hub / "models--google--gemma-3-27b-it"
    dest.mkdir(parents=True)
    (dest / "w.bin").write_bytes(b"g" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    from hermes_cli.vllm_runtime.inventory import GATED_DOWNLOAD_MSG

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.gated_repo_reason",
        lambda repo: GATED_DOWNLOAD_MSG)
    started: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: started.append("start"))

    used = client.post("/api/local-models/vllm/use", json={"model": hid})
    assert used.status_code == 400, used.text
    assert "gated" in used.json()["detail"].lower()
    assert "Bad Request" not in used.json()["detail"]
    assert started == []


def test_vllm_use_urllib_400_is_plain_language_400_not_502(tmp_path, monkeypatch):
    """HF urllib 400 must not become FastAPI 502 with HTTP Error 400: Bad Request."""
    client, home = _client(tmp_path, monkeypatch)
    hid = "solidrust/Hermes-3-Llama-3.1-8B-AWQ"
    _write_engine(home, "vllm")
    hub = tmp_path / "hf-hub"
    dest = hub / "models--solidrust--Hermes-3-Llama-3.1-8B-AWQ"
    dest.mkdir(parents=True)
    (dest / "w.bin").write_bytes(b"g" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))

    def _boom(_repo):
        raise urllib.error.HTTPError(
            "https://huggingface.co/api/models/x", 400, "Bad Request",
            hdrs=None, fp=BytesIO())

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory.gated_repo_reason", _boom)

    used = client.post("/api/local-models/vllm/use", json={"model": hid})
    assert used.status_code == 400, used.text
    detail = used.json()["detail"]
    assert "Bad Request" not in detail
    assert "502" not in used.text


def test_vllm_use_nous_bf16_is_needs_awq_400(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    hid = "NousResearch/Hermes-3-Llama-3.1-8B"
    _write_engine(home, "vllm")
    rec = _feasible_rec(monkeypatch)
    hub = tmp_path / "hf-hub"
    dest = hub / "models--NousResearch--Hermes-3-Llama-3.1-8B"
    dest.mkdir(parents=True)
    (dest / "w.bin").write_bytes(b"g" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    started: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: started.append("start"))

    used = client.post("/api/local-models/vllm/use", json={"model": hid})
    assert used.status_code == 400, used.text
    detail = used.json()["detail"].lower()
    assert "awq" in detail or "too big" in detail
    assert "Bad Request" not in used.json()["detail"]
    assert started == []
    assert rec.model != hid


def test_vllm_use_nous_awq_cached_starts(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    hid = "solidrust/Hermes-3-Llama-3.1-8B-AWQ"
    _write_engine(home, "vllm")
    _feasible_rec(monkeypatch)
    hub = tmp_path / "hf-hub"
    dest = hub / "models--solidrust--Hermes-3-Llama-3.1-8B-AWQ"
    dest.mkdir(parents=True)
    (dest / "w.bin").write_bytes(b"g" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)
    started: list[str] = []

    class _Sup:
        base_url = "http://127.0.0.1:9/v1"

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: started.append("start") or _Sup())
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:9/v1")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bench.verify_tool_calls",
        lambda *a, **k: {"ok": True, "tool_calls": True})

    used = client.post("/api/local-models/vllm/use", json={"model": hid})
    assert used.status_code == 200, used.text
    assert started == ["start"]
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["vllm"]["model"] == hid


def test_vllm_search_hf_400_is_not_502(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")

    def _boom(*_a, **_k):
        raise urllib.error.HTTPError(
            "https://huggingface.co/api/models", 400, "Bad Request",
            hdrs=None, fp=BytesIO())

    monkeypatch.setattr("hermes_cli.vllm_runtime.inventory._hf_json", _boom)
    search = client.get("/api/local-models/vllm/search?q=nous")
    assert search.status_code == 400, search.text
    assert "Bad Request" not in search.json()["detail"]


def test_delete_uncached_configured_search_hit_is_not_404(tmp_path, monkeypatch):
    """A 401 leftover in config with no hub dir must still clear."""
    client, home = _client(tmp_path, monkeypatch)
    leftover = "google/gemma-3-27b-it"
    rec = _feasible_rec(monkeypatch)
    _write_engine(home, "vllm", extra={"enabled": True, "vllm": {
        "model": leftover, "served_model_name": "gemma",
    }})
    hub = tmp_path / "hf-hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))

    gone = client.delete(f"/api/local-models/vllm/models/{leftover}")
    assert gone.status_code == 200, gone.text
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["vllm"]["model"] == rec.model
    listed = client.get("/api/local-models/vllm/models")
    ids = {m["id"] for m in listed.json()["models"]}
    assert leftover not in ids


def test_vllm_use_reloads_when_switching_cached_models(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm", extra={"vllm": {"model": "acme/old-awq"}})
    hub = tmp_path / "hf-hub"
    for name in ("acme--old-awq", "acme--sideload-awq"):
        dest = hub / f"models--{name}"
        dest.mkdir(parents=True)
        (dest / "weights.bin").write_bytes(b"y" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    order: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: {"base_url": "http://127.0.0.1:9/v1", "pid": 1})
    monkeypatch.setattr(
        "hermes_cli.local_engines.stop_vllm_engine",
        lambda: order.append("stop"))
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)

    class _Sup:
        base_url = "http://127.0.0.1:9/v1"

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: order.append("start") or _Sup())
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:9/v1")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bench.verify_tool_calls",
        lambda *a, **k: {"ok": True, "tool_calls": True})

    used = client.post("/api/local-models/vllm/use", json={"model": "acme/sideload-awq"})
    assert used.status_code == 200, used.text
    assert used.json()["needs_download"] is False
    assert "stop" in order and "start" in order
    assert order.index("stop") < order.index("start")
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["vllm"]["model"] == "acme/sideload-awq"


def test_vllm_use_failed_start_restores_previous(tmp_path, monkeypatch):
    """Failed Use of a new id must put In use back on the previous serve."""
    client, home = _client(tmp_path, monkeypatch)
    previous = "Qwen/Qwen3-8B-AWQ"
    doomed = "acme/doomed-awq"
    rec = _feasible_rec(monkeypatch)
    _write_engine(home, "vllm", extra={"enabled": True, "vllm": {
        "model": previous, "served_model_name": "qwen3:8b",
    }})
    hub = tmp_path / "hf-hub"
    for repo in (previous, doomed, rec.model):
        dest = hub / ("models--" + repo.replace("/", "--"))
        dest.mkdir(parents=True)
        (dest / "w.bin").write_bytes(b"y" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: {"base_url": "http://127.0.0.1:9/v1", "pid": 1})
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)
    starts: list[str] = []

    class _Sup:
        base_url = "http://127.0.0.1:9/v1"

    def _ensure(*_a, **_k):
        from hermes_cli.config import load_config
        from hermes_cli.vllm_runtime.supervisor import disable_auto_start, write_last_error

        model = load_config()["local_runtime"]["vllm"]["model"]
        starts.append(model)
        if model == doomed:
            write_last_error(f"vllm serve was killed (SIGKILL) starting {doomed}")
            disable_auto_start()
            raise RuntimeError(f"vllm serve was killed (SIGKILL) starting {doomed}")
        return _Sup()

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime", _ensure)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:9/v1")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bench.verify_tool_calls",
        lambda *a, **k: {"ok": True, "tool_calls": True})

    used = client.post("/api/local-models/vllm/use", json={"model": doomed})
    assert used.status_code == 400, used.text
    assert "SIGKILL" in used.json()["detail"]
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.supervisor import read_last_error

    assert load_config()["local_runtime"]["vllm"]["model"] == previous
    assert load_config()["local_runtime"]["vllm"]["served_model_name"] == "qwen3:8b"
    assert starts[0] == doomed
    assert previous in starts
    assert "SIGKILL" in (read_last_error() or "")


def test_vllm_use_failed_restore_falls_back_to_recommend(tmp_path, monkeypatch):
    """If the previous serve cannot come back, start the official recommend."""
    client, home = _client(tmp_path, monkeypatch)
    previous = "acme/old-awq"
    doomed = "acme/doomed-awq"
    rec = _feasible_rec(monkeypatch)
    _write_engine(home, "vllm", extra={"enabled": True, "vllm": {
        "model": previous, "served_model_name": "old-awq",
    }})
    hub = tmp_path / "hf-hub"
    for repo in (previous, doomed, rec.model):
        dest = hub / ("models--" + repo.replace("/", "--"))
        dest.mkdir(parents=True)
        (dest / "w.bin").write_bytes(b"y" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: {"base_url": "http://127.0.0.1:9/v1", "pid": 1})
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)
    starts: list[str] = []

    class _Sup:
        base_url = "http://127.0.0.1:9/v1"

    def _ensure(*_a, **_k):
        from hermes_cli.config import load_config
        from hermes_cli.vllm_runtime.supervisor import disable_auto_start, write_last_error

        model = load_config()["local_runtime"]["vllm"]["model"]
        starts.append(model)
        if model in {doomed, previous}:
            write_last_error(f"vllm serve was killed (SIGKILL) starting {model}")
            disable_auto_start()
            raise RuntimeError(f"vllm serve was killed (SIGKILL) starting {model}")
        return _Sup()

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime", _ensure)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:9/v1")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bench.verify_tool_calls",
        lambda *a, **k: {"ok": True, "tool_calls": True})

    used = client.post("/api/local-models/vllm/use", json={"model": doomed})
    assert used.status_code == 400, used.text
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.supervisor import read_last_error

    assert load_config()["local_runtime"]["vllm"]["model"] == rec.model
    assert starts[0] == doomed
    assert previous in starts
    assert rec.model in starts
    assert "SIGKILL" in (read_last_error() or "")


def test_server_start_vllm_succeeds_when_already_running(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    _seed_default_vllm_cache(tmp_path, monkeypatch)
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: {"base_url": "http://127.0.0.1:18435/v1", "pid": 3})
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:18435/v1")

    r = client.post("/api/local-models/server", json={"action": "start"})
    assert r.status_code == 200, r.text


def test_vllm_use_and_start_refuse_exl2_without_spawning(tmp_path, monkeypatch):
    """EXL2 must not kick a serve loop — Use/Start refuse and leave Download/Delete free."""
    client, home = _client(tmp_path, monkeypatch)
    hid = "org/Dolphin-8B-exl2"
    _write_engine(home, "vllm", extra={"enabled": True, "vllm": {
        "model": hid, "served_model_name": "dolphin-exl2", "quantization": "awq",
    }})
    hub = tmp_path / "hf-hub"
    cached = hub / "models--org--Dolphin-8B-exl2"
    cached.mkdir(parents=True)
    (cached / "w.bin").write_bytes(b"y" * 32)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    started: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: started.append("start"))
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)

    used = client.post("/api/local-models/vllm/use", json={"model": hid})
    assert used.status_code == 400, used.text
    assert "cannot serve this format" in used.json()["detail"]
    assert started == []

    started.clear()
    boot = client.post("/api/local-models/server", json={"action": "start"})
    assert boot.status_code == 400, boot.text
    assert "cannot serve this format" in boot.json()["detail"]
    assert started == []
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.supervisor import read_last_error

    assert load_config()["local_runtime"]["enabled"] is False
    assert "cannot serve this format" in (read_last_error() or "")


def test_failed_start_then_delete_and_use_catalog(tmp_path, monkeypatch):
    """After a doomed start, Delete + Use Qwen3-8B-AWQ must work without a CLI."""
    client, home = _client(tmp_path, monkeypatch)
    doomed = "dphn/dolphin-2.9.1-llama-3-8b"
    catalog = "Qwen/Qwen3-8B-AWQ"
    _write_engine(home, "vllm", extra={"enabled": True, "vllm": {
        "model": doomed, "served_model_name": "dolphin", "quantization": "awq",
    }})
    hub = tmp_path / "hf-hub"
    dest = hub / "models--dphn--dolphin-2.9.1-llama-3-8b"
    dest.mkdir(parents=True)
    (dest / "w.bin").write_bytes(b"x" * 64)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    monkeypatch.setattr("hermes_cli.local_engines.stop_llama_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.local_engines.stop_vllm_engine", lambda: None)
    monkeypatch.setattr("hermes_cli.vllm_runtime.occupancy.require_gpu_free", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: None)

    failed = client.post("/api/local-models/server", json={"action": "start"})
    assert failed.status_code == 502, failed.text
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["enabled"] is False

    gone = client.delete(f"/api/local-models/vllm/models/{doomed}")
    assert gone.status_code == 200, gone.text
    assert not dest.exists()

    qwen = hub / "models--Qwen--Qwen3-8B-AWQ"
    qwen.mkdir(parents=True)
    (qwen / "w.bin").write_bytes(b"q" * 32)
    started: list[str] = []

    class _Sup:
        base_url = "http://127.0.0.1:9/v1"

    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.ensure_vllm_runtime",
        lambda *a, **k: started.append("start") or _Sup())
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bootstrap.activate_vllm_provider",
        lambda cfg=None: "http://127.0.0.1:9/v1")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.bench.verify_tool_calls",
        lambda *a, **k: {"ok": True, "tool_calls": True})

    used = client.post("/api/local-models/vllm/use", json={"model": catalog})
    assert used.status_code == 200, used.text
    assert started == ["start"]
    cfg = load_config()
    assert cfg["local_runtime"]["vllm"]["model"] == catalog
    assert cfg["local_runtime"]["vllm"].get("quantization") == "awq"


def test_apply_search_hit_clears_leftover_awq(tmp_path, monkeypatch):
    """Using Dolphin after a Qwen3-AWQ recommend must not keep --quantization awq."""
    _, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm", extra={"vllm": {
        "model": "Qwen/Qwen3-8B-AWQ",
        "quantization": "awq",
        "kv_cache_dtype": "fp8",
    }})
    from hermes_cli.config import load_config
    from hermes_cli.vllm_runtime.inventory import apply_vllm_model

    apply_vllm_model("dphn/dolphin-2.9.1-llama-3-8b")
    vllm = load_config()["local_runtime"]["vllm"]
    assert vllm["model"] == "dphn/dolphin-2.9.1-llama-3-8b"
    assert not (vllm.get("quantization") or "").strip()


def test_vllm_job_timeout_covers_supervisor_ready_wait():
    """Use / quickstart / server-start must not 60–90s-fail while CUDA graphs capture."""
    from hermes_cli.vllm_runtime.supervisor import READY_TIMEOUT_S
    from hermes_cli.web_routers.local_models_engine import VLLM_START_TIMEOUT_S

    assert VLLM_START_TIMEOUT_S >= READY_TIMEOUT_S
    assert READY_TIMEOUT_S >= 180


def test_vllm_log_phase_sniffs_warmup_and_cuda_graphs(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
    log = home / "runtimes" / "vllm" / "vllm-server.log"
    log.parent.mkdir(parents=True)
    log.write_text("Loading weights took 12s\nWarming up Mamba kernels\n", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.venv.venv_ready", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.endpoint.resolve_vllm_endpoint",
        lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models_engine.occupancy_payload",
        lambda: {"occupancy": [], "occupancy_message": None})

    data = client.get("/api/local-models/status").json()
    assert data["start_phase"] == "Warming up GPU"

    log.write_text(
        "Warming up Mamba kernels\nCapturing CUDA graphs (decode, 32)\n", encoding="utf-8")
    data = client.get("/api/local-models/status").json()
    assert data["start_phase"] == "Capturing CUDA graphs"
