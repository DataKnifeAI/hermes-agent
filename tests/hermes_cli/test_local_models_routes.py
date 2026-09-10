"""Contract tests for the local-models dashboard routes (Rollout 4).

Real FastAPI TestClient against the real router; the runtime pieces
underneath are exercised against temp HERMES_HOME (autouse fixture). Network
downloads are stubbed at the urllib boundary — never live."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    test_client = TestClient(web_server.app)
    # Same auth pattern as the git-route tests: present the session token.
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return test_client


def test_local_models_routes_require_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    unauth = TestClient(web_server.app)
    assert unauth.get("/api/local-models/status").status_code == 401


def _write_fake_gguf(path: Path, size: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\x00" * size)


# ── status ───────────────────────────────────────────────────


def test_status_shape_and_defaults(client):
    r = client.get("/api/local-models/status")
    assert r.status_code == 200
    data = r.json()
    # Contract: every key the pane's first paint needs, present and typed.
    assert isinstance(data["enabled"], bool)
    assert isinstance(data["tag"], str) and data["tag"].startswith("b")
    assert isinstance(data["runtime_installed"], bool)
    assert isinstance(data["server_running"], bool)
    assert isinstance(data["models"], list)


def test_status_lists_staged_models_with_labels(client, tmp_path):
    from hermes_cli.local_runtime.bootstrap import models_dir

    _write_fake_gguf(models_dir() / "Some-Model.gguf", size=2048)
    data = client.get("/api/local-models/status").json()
    ids = [m["id"] for m in data["models"]]
    assert "Some-Model" in ids
    row = data["models"][ids.index("Some-Model")]
    assert row["size_bytes"] > 0
    assert row["size_label"].endswith("GB")


# ── hardware ─────────────────────────────────────────────────


def test_hardware_plain_facts(client):
    data = client.get("/api/local-models/hardware").json()
    assert isinstance(data["uma"], bool)
    assert data["ram_total_bytes"] > 0
    assert data["vram_total_bytes"] >= 0
    # GPU fields are None-able (non-NVIDIA machines) but must exist.
    assert "gpu_name" in data and "gpu_util_percent" in data and "vram_used_bytes" in data
    assert "ram_used_bytes" in data and "vllm_version" in data
    # Used without total is useless — when both are present, used cannot exceed total.
    if data["vram_used_bytes"] is not None:
        assert data["vram_used_bytes"] <= data["vram_total_bytes"]
    if data["ram_used_bytes"] is not None:
        assert data["ram_used_bytes"] <= data["ram_total_bytes"]
    if data["vllm_version"] is not None:
        assert isinstance(data["vllm_version"], str) and data["vllm_version"]
    if data.get("vram_free_bytes") is not None:
        assert data["vram_free_bytes"] <= data["vram_total_bytes"]
    if data.get("vram_engine_bytes") is not None and data.get("vram_other_bytes") is not None:
        assert data["vram_engine_bytes"] + data["vram_other_bytes"] >= 0
        if data["vram_total_bytes"]:
            assert data["vram_engine_bytes"] + data["vram_other_bytes"] <= data["vram_total_bytes"] + (16 << 20)
    for key in (
        "vram_free_bytes", "vram_engine_bytes", "vram_other_bytes",
        "gpu_driver_version", "cuda_compute_capability", "engine",
        "models_dir", "models_dir_display", "models_storage_bytes",
        "disk_free_bytes", "disk_total_bytes", "runtime_dir",
        "runtime_dir_display", "occupancy_foreign", "ctx_64k_feasible",
    ):
        assert key in data
    assert data["models_storage_bytes"] >= 0
    assert data["disk_free_bytes"] >= 0
    assert data["engine"] in ("llamacpp", "vllm")


def test_hardware_cache_path_is_profile_aware_and_engine_specific(tmp_path, monkeypatch):
    """llama.cpp cache is the machine models dir; vLLM is the HF hub — never a
    hardcoded ~/.hermes string, and both live on the same hardware payload."""
    import yaml

    home = tmp_path / ".hermes"
    home.mkdir()
    hf = tmp_path / "hf-home"
    hf.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HF_HOME", str(hf))
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    from hermes_cli import web_server

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.vllm_runtime.inventory import hf_hub_dir
    from hermes_constants import get_default_hermes_root

    llama = client.get("/api/local-models/hardware").json()
    assert llama["engine"] == "llamacpp"
    assert Path(llama["models_dir"]) == models_dir()
    assert models_dir() == get_default_hermes_root() / "models"
    assert "~/.hermes" not in llama["models_dir"]
    assert Path(llama["models_dir"]).is_relative_to(get_default_hermes_root())

    (home / "config.yaml").write_text(
        yaml.dump({"local_runtime": {"enabled": True, "engine": "vllm"}}),
        encoding="utf-8")
    vllm = client.get("/api/local-models/hardware").json()
    assert vllm["engine"] == "vllm"
    assert Path(vllm["models_dir"]) == hf_hub_dir()
    assert hf_hub_dir().is_relative_to(hf)
    assert vllm["models_dir"] != llama["models_dir"]
    assert vllm["models_storage_bytes"] >= 0


# ── catalog ──────────────────────────────────────────────────


def test_catalog_prices_every_entry_for_this_machine(client):
    data = client.get("/api/local-models/catalog").json()
    assert len(data["models"]) >= 3
    for row in data["models"]:
        # The three user questions, answered on every row:
        assert row["size_label"].endswith("GB")            # how big
        assert isinstance(row["fits"], bool)               # will it fit
        assert row["fit_summary"]                          # what shape
        if row["fits"]:
            assert row["start_window"] >= 1
            assert row["start_window_label"].endswith("K")
        else:
            assert "memory" in row["fit_summary"].lower()
        assert isinstance(row["downloaded"], bool)


def test_catalog_never_hides_unaffordable_models(client, monkeypatch):
    """Unaffordable entries stay visible with a plain reason — hiding them
    is how users conclude the feature is broken."""
    from hermes_cli.local_runtime.estimator import HardwareBudget

    tiny = HardwareBudget(usable_vram_bytes=1 << 30, total_device_bytes=1 << 30,
                          ram_available_bytes=1 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: tiny)
    data = client.get("/api/local-models/catalog").json()
    from hermes_cli.local_runtime.catalog import CATALOG

    assert len(data["models"]) == len(CATALOG)
    refused = [m for m in data["models"] if not m["fits"]]
    assert refused, "a 1 GiB machine must refuse the 20 GB models"
    for row in refused:
        assert row["fit_detail"] or row["fit_summary"]


# ── downloads ────────────────────────────────────────────────


def test_download_unknown_model_404s(client):
    r = client.post("/api/local-models/download", json={"model_id": "nope"})
    assert r.status_code == 404


def test_download_short_of_server_length_errors_and_cleans_up(client, monkeypatch):
    """Catalog sizes are advisory (upstream re-uploads may make them
    stale — a mismatch against the CATALOG must not fail a download).
    The server's own declared length is the only completeness check:
    fewer bytes than the server promised means a dropped connection, so
    the job errors and nothing is staged."""

    class FakeResponse(io.BytesIO):
        # Body is 17 bytes; the server promises 32 — a truncated stream.
        headers = {"Content-Length": "32"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: FakeResponse(b"not the real body"))

    # Pin a generous budget: variant selection prices against the machine
    # running the test, and a GPU-less CI runner honestly refuses every
    # build (409) — this test is about the download path, not selection.
    from hermes_cli.local_runtime.estimator import HardwareBudget

    budget = HardwareBudget(usable_vram_bytes=64 << 30,
                            total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)

    from hermes_cli.local_runtime.catalog import CATALOG

    entry_id = CATALOG[0].id
    r = client.post("/api/local-models/download", json={"model_id": entry_id})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert job_id

    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status is not None and status["status"] == "error"
    assert "bytes" in status["error"].lower()

    from hermes_cli.local_runtime.bootstrap import models_dir

    assert not (models_dir() / f"{entry_id}.gguf").exists()
    assert not (models_dir() / f"{entry_id}.part").exists()


def test_download_already_downloaded_short_circuits(client, monkeypatch):
    from hermes_cli.local_runtime.bootstrap import models_dir
    from hermes_cli.local_runtime.catalog import CATALOG, select_variant
    from hermes_cli.local_runtime.estimator import HardwareBudget

    # Pin the budget so the selected variant is deterministic in the test.
    budget = HardwareBudget(usable_vram_bytes=64 << 30, total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)
    choice = select_variant(CATALOG[0], budget)
    assert choice is not None
    _write_fake_gguf(models_dir() / choice.variant.files[0].local_name)
    r = client.post("/api/local-models/download", json={"model_id": CATALOG[0].id})
    assert r.status_code == 200
    assert r.json()["already_downloaded"] is True


def test_delete_model(client):
    from hermes_cli.local_runtime.bootstrap import models_dir

    _write_fake_gguf(models_dir() / "Doomed.gguf")
    assert client.delete("/api/local-models/models/Doomed").status_code == 200
    assert not (models_dir() / "Doomed.gguf").exists()
    assert client.delete("/api/local-models/models/Doomed").status_code == 404


# ── runtime install ──────────────────────────────────────────


def test_runtime_install_rejects_impossible_combo(client, monkeypatch):
    """Impossible platform/backend combos fail the POST itself with the
    resolver's honest message — not a background job that dies silently.
    (win-arm64-vulkan; the old cuda case became real upstream at ~b1036x.)"""
    monkeypatch.setattr(
        "hermes_cli.local_runtime.binaries._host_os_arch", lambda: ("win", "arm64"))
    r = client.post("/api/local-models/runtime/install", json={"backend": "vulkan"})
    assert r.status_code == 400
    assert "arm64" in r.json()["detail"]


def test_job_poll_unknown_404s(client):
    assert client.get("/api/local-models/jobs/deadbeef").status_code == 404


def test_eject_without_supervisor_is_not_a_500(client, monkeypatch):
    """Eject on an ADOPTED server (no in-process supervisor — the shape
    every backend restart produces, since boot adopts the running server
    via the state file) must route through the persisted endpoint, not
    crash. Regression: _state_endpoint was only imported inside the
    status route, so eject raised NameError -> 500 for every adopted-
    server session."""
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.get_supervisor", lambda: None)
    # No running server either: the route must answer 409 (no server),
    # never a NameError 500.
    monkeypatch.setattr(
        "hermes_cli.web_routers.local_models._state_endpoint", lambda: None)
    r = client.post("/api/local-models/eject", json={"model_id": "anything"})
    assert r.status_code == 409, (r.status_code, r.text)


def test_download_tolerates_stale_catalog_size(client, monkeypatch):
    """Upstream re-uploads make catalog sizes stale; a download whose
    delivered bytes are self-consistent with the SERVER's declared length
    must succeed even when the catalog said something else. (This is the
    tolerance the sha removal was for — being out of date must not break
    downloads.)"""

    body = b"x" * 48  # server-consistent: Content-Length == body length

    class FakeResponse(io.BytesIO):
        headers = {"Content-Length": str(len(body))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: FakeResponse(body))

    from hermes_cli.local_runtime.estimator import HardwareBudget

    budget = HardwareBudget(usable_vram_bytes=64 << 30,
                            total_device_bytes=64 << 30,
                            ram_available_bytes=64 << 30)
    monkeypatch.setattr("hermes_cli.local_runtime.hardware.probe_budget",
                        lambda **kw: budget)
    # Keep the post-download server bounce out of this unit.
    monkeypatch.setattr(
        "hermes_cli.local_runtime.bootstrap.refresh_local_runtime",
        lambda: False)

    from hermes_cli.local_runtime.catalog import CATALOG

    # Catalog size for this entry is in the tens of GB — wildly stale
    # versus our 48-byte body. The download must still land.
    entry_id = CATALOG[0].id
    r = client.post("/api/local-models/download", json={"model_id": entry_id})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    deadline = time.time() + 10
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/local-models/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert status is not None and status["status"] == "done", status.get("error")
