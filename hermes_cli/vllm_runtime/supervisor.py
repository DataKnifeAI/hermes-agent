"""Supervision of one ``vllm serve`` process.

Readiness is ``GET /v1/models`` (process still alive). Health checks always
dial ``127.0.0.1`` — resolving ``localhost`` pays an IPv6 fallback tax (same
lesson as the llama.cpp supervisor). Default bind is loopback; ``0.0.0.0``
only when the user set ``local_runtime.vllm.host``.
"""

from __future__ import annotations

from contextlib import suppress
import json
import logging
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from hermes_cli.vllm_runtime.venv import runtimes_root, vllm_executable

logger = logging.getLogger(__name__)

_RESTART_BACKOFF_S = (1, 5, 15, 60)
# llama.cpp's managed listen port — never share it. Sessions persist base_url per
# engine; TIME_WAIT after a switch would also collide. Same reason llama.cpp
# avoids 8000/8080.
LLAMA_CPP_PORT = 18434
# Preferred vLLM bind when ``local_runtime.vllm.port`` is 0 (pick at spawn).
DEFAULT_LISTEN_PORT = 18435
_LOOPBACK = "127.0.0.1"


def state_path() -> Path:
    return runtimes_root() / "server.json"


def vllm_settings(config: dict | None) -> dict:
    """``local_runtime.vllm`` merged over DEFAULT_CONFIG so a partial section still serves."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    defaults = dict(DEFAULT_CONFIG["local_runtime"]["vllm"])
    override = ((config or {}).get("local_runtime") or {}).get("vllm") or {}
    if isinstance(override, dict):
        defaults.update(override)
    return defaults


def bind_host(settings: dict) -> str:
    host = str(settings.get("host") or _LOOPBACK).strip() or _LOOPBACK
    return host


def client_host(settings: dict) -> str:
    """Host the OpenAI client should dial. Wildcard binds are reached via loopback."""
    host = bind_host(settings)
    if host in ("0.0.0.0", "::", "[::]"):
        return _LOOPBACK
    return host


def openai_base_url(settings: dict, *, port: int | None = None) -> str:
    if port is not None:
        p = int(port)
    else:
        p = int(settings.get("port") or 0) or DEFAULT_LISTEN_PORT
    return f"http://{client_host(settings)}:{p}/v1"


def serve_argv(executable: str | Path, settings: dict) -> list[str]:
    """``vllm serve`` argv. Bind host comes from settings; 1-click default is loopback."""
    model = str(settings.get("model") or "").strip()
    if not model:
        raise ValueError("local_runtime.vllm.model is required")
    port = int(settings.get("port") or 0) or DEFAULT_LISTEN_PORT
    argv = [
        str(executable), "serve", model,
        "--host", bind_host(settings),
        "--port", str(port),
        "--max-model-len", str(int(settings.get("max_model_len") or 65536)),
        "--gpu-memory-utilization", str(settings.get("gpu_memory_utilization") or 0.75),
        "--enable-auto-tool-choice",
        "--tool-call-parser", str(settings.get("tool_call_parser") or "hermes"),
    ]
    quant = str(settings.get("quantization") or "").strip()
    if quant:
        argv.extend(["--quantization", quant])
    kv = str(settings.get("kv_cache_dtype") or "").strip()
    if kv:
        argv.extend(["--kv-cache-dtype", kv])
    served = str(settings.get("served_model_name") or "").strip()
    if served:
        argv.extend(["--served-model-name", served])
    return argv


def _quiet(fn) -> None:
    with suppress(Exception):
        fn()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((_LOOPBACK, 0))
        return s.getsockname()[1]


def pick_listen_port(preferred: int = 0) -> int:
    """Like llama.cpp: try the engine's stable default, else an ephemeral port.

    ``preferred`` 0 means DEFAULT_LISTEN_PORT. Never bind llama.cpp's 18434.
    """
    candidate = preferred if preferred > 0 else DEFAULT_LISTEN_PORT
    if candidate == LLAMA_CPP_PORT:
        candidate = DEFAULT_LISTEN_PORT
    try:
        with socket.socket() as s:
            s.bind((_LOOPBACK, candidate))
            return candidate
    except OSError:
        logger.warning(
            "port %d busy; managed vLLM falling back to an ephemeral "
            "port — existing sessions may need a model re-pick", candidate)
        port = _free_port()
        if port == LLAMA_CPP_PORT:
            port = _free_port()
        return port


class VllmSupervisor:
    """Own one ``vllm serve`` process for the life of a Hermes session."""

    def __init__(self, settings: dict, *, executable: Path | None = None,
                 log_path: Path | None = None):
        self.settings = dict(settings)
        preferred = int(self.settings.get("port") or 0)
        self.port = pick_listen_port(preferred)
        self.settings["port"] = self.port
        self.executable = Path(executable) if executable else vllm_executable()
        self.log_path = log_path or (runtimes_root() / "vllm-server.log")
        self.proc: subprocess.Popen | None = None
        self._restarts = 0
        self._stopping = False
        self._watchdog: threading.Thread | None = None
        self._log_handle = None

    @property
    def base_url(self) -> str:
        return openai_base_url(self.settings, port=self.port)

    def _health_url(self) -> str:
        return f"http://{_LOOPBACK}:{self.port}/v1/models"

    def _spawn(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(self.log_path, "ab")  # noqa: SIM115
        cmd = serve_argv(self.executable, self.settings)
        env = os.environ.copy()
        bindir = str(Path(self.executable).parent)
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
        self.proc = subprocess.Popen(
            cmd, stdout=self._log_handle, stderr=subprocess.STDOUT, env=env)
        logger.info("vllm serve spawned pid=%s port=%s", self.proc.pid, self.port)
        self._write_state()

    def start(self, timeout_s: int = 120) -> None:
        self._stopping = False
        self._spawn()
        self._wait_ready(timeout_s)
        self._write_state()
        self._watchdog = threading.Thread(
            target=self._watch, daemon=True, name="vllm-supervisor")
        self._watchdog.start()

    def _write_state(self) -> None:
        path = state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "base_url": self.base_url,
            "pid": self.proc.pid if self.proc else None,
            "port": self.port,
            "bind": bind_host(self.settings),
            "served_model_name": self.settings.get("served_model_name"),
        }), encoding="utf-8")

    def _wait_ready(self, timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited rc={self.proc.returncode} during startup "
                    f"(log: {self.log_path})")
            with suppress(urllib.error.URLError, OSError, TimeoutError):
                with urllib.request.urlopen(self._health_url(), timeout=3) as r:
                    if r.status == 200:
                        return
            time.sleep(0.25)
        raise TimeoutError(
            f"vllm serve not ready after {timeout_s}s (log: {self.log_path})")

    def _watch(self) -> None:
        while not self._stopping:
            proc = self.proc
            if proc is None:
                return
            rc = proc.poll()
            if rc is None:
                time.sleep(2)
                continue
            if self._stopping:
                return
            backoff = _RESTART_BACKOFF_S[min(self._restarts, len(_RESTART_BACKOFF_S) - 1)]
            logger.warning("vllm serve exited rc=%s; restart #%s in %ss",
                           rc, self._restarts + 1, backoff)
            time.sleep(backoff)
            self._restarts += 1
            try:
                self._spawn()
                self._wait_ready(120)
            except Exception as exc:  # noqa: BLE001
                logger.error("vllm serve restart failed: %s", exc)

    def stop(self) -> None:
        self._stopping = True
        state_path().unlink(missing_ok=True)
        if self.proc and self.proc.poll() is None:
            self._terminate_tree(self.proc)
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    pause = stop  # v1: explicit stop only; no llama-style idle unload.

    @staticmethod
    def _terminate_tree(proc: subprocess.Popen) -> None:
        """SIGTERM the serve process and its CUDA children, then SIGKILL.

        Children hold the weights; killing only the parent orphans VRAM.
        """
        children: list = []
        with suppress(Exception):
            import psutil

            children = psutil.Process(proc.pid).children(recursive=True)
        proc.terminate()
        for child in children:
            _quiet(child.terminate)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        for child in children:
            _quiet(lambda: child.is_running() and child.kill())
