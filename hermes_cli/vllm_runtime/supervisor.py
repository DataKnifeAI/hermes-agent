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
_MAX_CRASH_RESTARTS = len(_RESTART_BACKOFF_S)
# GET /v1/models must wait through Mamba warmup + CUDA graph capture, not just
# process spawn. Nemotron is ~98s cold; 120s left no margin. Zip/venv install
# is a separate job and does not share this budget.
READY_TIMEOUT_S = 180
MODEL_REMOVED_MSG = "model was removed — Download to use again"
LEFTOVER_AWQ_MSG = (
    "vLLM cannot start this model as AWQ — it has no AWQ config. "
    "Stop, then Use an AWQ or FP8 instruct model"
)
_FATAL_CRASH_NEEDLES = (
    "cannot find the config file",
    "unknown quantization method",
    "is not a supported model",
    "exl2",
    "exllamav2",
    "gguf",
)
# llama.cpp's managed listen port — never share it. Sessions persist base_url per
# engine; TIME_WAIT after a switch would also collide. Same reason llama.cpp
# avoids 8000/8080.
LLAMA_CPP_PORT = 18434
# Preferred vLLM bind when ``local_runtime.vllm.port`` is 0 (pick at spawn).
DEFAULT_LISTEN_PORT = 18435
_LOOPBACK = "127.0.0.1"


def state_path() -> Path:
    return runtimes_root() / "server.json"


def last_error_path() -> Path:
    return runtimes_root() / "last_error.json"


def write_last_error(message: str) -> None:
    path = last_error_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"error": message}, ensure_ascii=False), encoding="utf-8")


def state_served_model_name() -> str:
    """Served id recorded by the last spawn. Empty when no state file / pid."""
    path = state_path()
    if not path.is_file():
        return ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("served_model_name") or "").strip()


def read_last_error() -> str | None:
    path = last_error_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(raw, dict):
        text = str(raw.get("error") or "").strip()
        return text or None
    return None


def clear_last_error() -> None:
    last_error_path().unlink(missing_ok=True)


def disable_auto_start() -> None:
    """A failed start must not leave ``enabled`` kicking boot forever."""
    with suppress(Exception):
        from cli import save_config_value

        save_config_value("local_runtime.enabled", False)


def configured_unservable_reason(settings: dict | None) -> str | None:
    hid = configured_model_id(settings)
    if not hid:
        return None
    from hermes_cli.vllm_runtime.inventory import unservable_reason

    return unservable_reason(hid)


def serve_quantization(settings: dict) -> str:
    """Drop leftover ``--quantization`` when the HF id does not claim that method.

    Recommend writes ``awq``. Using a BF16 search hit (Dolphin) then passes
    ``--quantization awq`` and vLLM dies looking for an AWQ config file.
    """
    quant = str(settings.get("quantization") or "").strip()
    if not quant:
        return ""
    from hermes_cli.vllm_runtime.inventory import parse_quantization

    parsed = parse_quantization(configured_model_id(settings))
    return quant if parsed == quant else ""


def humanize_serve_error(raw: str | None, *, model: str = "") -> str | None:
    blocked = configured_unservable_reason({"model": model} if model else None)
    if blocked:
        return blocked
    text = (raw or "").strip()
    lower = text.lower()
    if "cannot find the config file for awq" in lower:
        return LEFTOVER_AWQ_MSG
    if "out of memory" in lower or "oom" in lower or "sigkill" in lower:
        label = model or "this model"
        return (
            f"{label} is too big for this GPU (OOM / SIGKILL) — "
            "Use an AWQ or FP8 instruct model"
        )
    return text[-500:] or None


def is_fatal_serve_crash(message: str | None, *, model: str = "") -> bool:
    if configured_unservable_reason({"model": model} if model else None):
        return True
    lower = (message or "").lower()
    return any(needle in lower for needle in _FATAL_CRASH_NEEDLES)


def configured_model_id(settings: dict | None) -> str:
    return str((settings or {}).get("model") or "").strip()


def configured_cache_missing(settings: dict | None) -> bool:
    """True when a configured HF id is set but its hub cache dir is gone."""
    hid = configured_model_id(settings)
    if not hid:
        return False
    from hermes_cli.vllm_runtime.inventory import repo_is_cached

    return not repo_is_cached(hid)


def last_serve_error_line(log_path: Path | None = None) -> str | None:
    """Last real error line from vllm-server.log — not a restart-loop breadcrumb."""
    path = Path(log_path) if log_path else (runtimes_root() / "vllm-server.log")
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[-12000:]
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if (
            "validationerror" in lower
            or "value error" in lower
            or "error:" in lower
            or "out of memory" in lower
            or "oom" in lower
        ):
            return stripped[-500:]
    return None


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


def probe_served_model_name(base_url: str, timeout_s: float = 1.5) -> str:
    """First id from GET ``{base_url}/models`` 200. Empty while CUDA graphs run.

    Spawn-time ``server.json`` is not readiness — only this probe is.
    """
    url = str(base_url or "").rstrip("/") + "/models"
    if not str(base_url or "").strip():
        return ""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            if int(getattr(resp, "status", 200) or 200) != 200:
                return ""
            body = json.loads(resp.read() or b"{}")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError,
            json.JSONDecodeError, ValueError, TypeError):
        return ""
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return ""
    for row in data:
        if isinstance(row, dict):
            hid = str(row.get("id") or "").strip()
            if hid:
                return hid
    return ""


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
    quant = serve_quantization(settings)
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
        # Qwen3-8B-AWQ (the 16 GB shipped default) derives 40960; Hermes' tool
        # loop is 64k. vLLM refuses the override unless this is set.
        env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
        self.proc = subprocess.Popen(
            cmd, stdout=self._log_handle, stderr=subprocess.STDOUT, env=env)
        logger.info("vllm serve spawned pid=%s port=%s", self.proc.pid, self.port)
        self._write_state()

    def start(self, timeout_s: int = READY_TIMEOUT_S) -> None:
        if self._stopping:
            raise RuntimeError("vllm serve stopped during startup")
        hid = configured_model_id(self.settings)
        blocked = configured_unservable_reason(self.settings)
        if blocked:
            write_last_error(blocked)
            disable_auto_start()
            raise RuntimeError(blocked)
        if configured_cache_missing(self.settings):
            write_last_error(MODEL_REMOVED_MSG)
            disable_auto_start()
            raise RuntimeError(MODEL_REMOVED_MSG)
        clear_last_error()
        try:
            self._spawn()
            self._wait_ready(timeout_s)
        except Exception as exc:
            # Restore/stop of an in-flight boot must not poison last_error or
            # flip enabled=false while the replacement serve is starting.
            if self._stopping:
                raise
            crash = last_serve_error_line(self.log_path)
            if configured_cache_missing(self.settings):
                write_last_error(MODEL_REMOVED_MSG)
            else:
                write_last_error(
                    humanize_serve_error(crash or str(exc), model=hid)
                    or str(exc).strip()
                    or "vllm serve failed"
                )
            disable_auto_start()
            raise
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
        """Poll GET /v1/models until 200. Connection refused / 5xx is not failure
        while the pid is alive — CUDA graph capture looks like that for minutes."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._stopping:
                raise RuntimeError("vllm serve stopped during startup")
            if self.proc and self.proc.poll() is not None:
                rc = self.proc.returncode
                hid = configured_model_id(self.settings)
                if rc == -9:
                    raise RuntimeError(
                        f"vllm serve was killed (SIGKILL) starting {hid or 'the model'}"
                    )
                raise RuntimeError(
                    f"vllm serve exited rc={rc} during startup "
                    f"(log: {self.log_path})")
            try:
                with urllib.request.urlopen(self._health_url(), timeout=3) as r:
                    if r.status == 200:
                        return
            except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError):
                pass
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
            hid = configured_model_id(self.settings)
            if configured_cache_missing(self.settings):
                write_last_error(MODEL_REMOVED_MSG)
                logger.error("%s", MODEL_REMOVED_MSG)
                disable_auto_start()
                self.stop()
                return
            blocked = configured_unservable_reason(self.settings)
            crash = last_serve_error_line(self.log_path)
            if blocked:
                write_last_error(blocked)
                logger.error("%s", blocked)
                disable_auto_start()
                self.stop()
                return
            if crash:
                write_last_error(humanize_serve_error(crash, model=hid) or crash)
            if is_fatal_serve_crash(crash, model=hid):
                logger.error("vllm serve fatal; not restarting: %s", crash)
                disable_auto_start()
                self.stop()
                return
            if self._restarts >= _MAX_CRASH_RESTARTS:
                write_last_error(
                    humanize_serve_error(crash, model=hid)
                    or f"vllm serve crashed {self._restarts} times — Stop, then Use another model"
                )
                logger.error("vllm serve restart budget exhausted")
                disable_auto_start()
                self.stop()
                return
            backoff = _RESTART_BACKOFF_S[min(self._restarts, len(_RESTART_BACKOFF_S) - 1)]
            logger.warning("vllm serve exited rc=%s; restart #%s in %ss",
                           rc, self._restarts + 1, backoff)
            time.sleep(backoff)
            if self._stopping:
                return
            if configured_cache_missing(self.settings):
                write_last_error(MODEL_REMOVED_MSG)
                logger.error("%s", MODEL_REMOVED_MSG)
                disable_auto_start()
                self.stop()
                return
            self._restarts += 1
            try:
                self._spawn()
                self._wait_ready(READY_TIMEOUT_S)
                clear_last_error()
            except Exception as exc:  # noqa: BLE001
                logger.error("vllm serve restart failed: %s", exc)
                write_last_error(humanize_serve_error(str(exc), model=hid) or str(exc))
                if configured_cache_missing(self.settings) or is_fatal_serve_crash(str(exc), model=hid):
                    if configured_cache_missing(self.settings):
                        write_last_error(MODEL_REMOVED_MSG)
                    disable_auto_start()
                    self.stop()
                    return

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
