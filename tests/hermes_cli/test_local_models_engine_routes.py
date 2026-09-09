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
    hermes = next(h for h in search.json()["hits"] if "Hermes-3" in h["repo"])
    # 8B in the id, no quant on the card → unknown, never a lying Fits badge.
    assert hermes["fit"] == "unknown"
    assert "tools" in hermes["capabilities"]


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


def test_vllm_use_downloads_then_starts_when_uncached(tmp_path, monkeypatch):
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

    used = client.post("/api/local-models/vllm/use", json={"model": "acme/fresh-awq"})
    assert used.status_code == 200, used.text
    body = used.json()
    assert body["needs_download"] is True
    assert body["already_downloaded"] is False
    assert body["job_id"]
    job = _wait_job(client, body["job_id"])
    assert job["status"] == "done", job
    assert job["kind"] == "model-download"
    assert pulled == ["acme/fresh-awq"]
    assert started == ["start"]
    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["vllm"]["model"] == "acme/fresh-awq"


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

    fresh = client.post("/api/local-models/vllm/download", json={"model": "acme/fresh-awq"})
    assert fresh.status_code == 200
    assert fresh.json()["already_downloaded"] is False
    job = _wait_job(client, fresh.json()["job_id"])
    assert job["status"] == "done", job
    assert pulled == ["acme/fresh-awq"]


def test_vllm_install_downloads_recommended_weights(tmp_path, monkeypatch):
    client, home = _client(tmp_path, monkeypatch)
    _write_engine(home, "vllm")
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
    monkeypatch.setattr(
        "hermes_cli.vllm_runtime.inventory._hf_json",
        lambda url: [
            {"id": "Qwen/Qwen3-8B-AWQ", "downloads": 9, "likes": 2,
             "lastModified": "", "gated": False, "tags": ["awq", "instruct"]},
            {"id": "Qwen/Qwen3-14B-AWQ", "downloads": 5, "likes": 1,
             "lastModified": "", "gated": False, "tags": ["awq"]},
            {"id": "someone/mystery-weights", "downloads": 1, "likes": 0,
             "lastModified": "", "gated": False, "tags": []},
            {"id": "Qwen/Qwen3-32B-AWQ", "downloads": 3, "likes": 0,
             "lastModified": "", "gated": False, "tags": ["awq"]},
        ])

    search = client.get("/api/local-models/vllm/search?q=qwen")
    assert search.status_code == 200
    by_repo = {h["repo"]: h for h in search.json()["hits"]}
    assert by_repo["Qwen/Qwen3-8B-AWQ"]["fit"] == "fits-gpu"
    assert "awq" in by_repo["Qwen/Qwen3-8B-AWQ"]["capabilities"]
    assert "instruct" in by_repo["Qwen/Qwen3-8B-AWQ"]["capabilities"]
    # 24 GB card recommends the 24gb tier (14B), not the 16gb 8B row.
    assert by_repo["Qwen/Qwen3-8B-AWQ"]["recommended"] is False
    assert by_repo["Qwen/Qwen3-14B-AWQ"]["recommended"] is True
    assert by_repo["someone/mystery-weights"]["fit"] == "unknown"
    assert by_repo["Qwen/Qwen3-32B-AWQ"]["fit"] == "too-big"
