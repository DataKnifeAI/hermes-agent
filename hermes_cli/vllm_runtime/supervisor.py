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

from hermes_cli.vllm_runtime.device import (
    CPU, CPU_LISTEN_PORT, GPU_LISTEN_PORT, LLAMA_CPP_PORT, RESERVED_PORTS,
    USER_SERVER_PORTS, default_listen_port, normalize_device,
)
from hermes_cli.vllm_runtime.venv import (
    _intel_openmp_library, runtimes_root, server_log_path, vllm_executable,
)

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
# vLLM wraps the EngineCore AttributeError; the pane must not hang on this.
_LOG_WRAPPER_NEEDLES = (
    "see root cause above",
    "engine core initialization failed",
    "failed core proc",
)
_FATAL_CRASH_NEEDLES = (
    "cannot find the config file",
    "unknown quantization method",
    "is not a supported model",
    "exl2",
    "exllamav2",
    "gguf",
    "draft_model_config",
    "qwen3_dspark",
)
# Preferred GPU bind when ``local_runtime.vllm.port`` is 0 (pick at spawn).
DEFAULT_LISTEN_PORT = GPU_LISTEN_PORT
_LOOPBACK = "127.0.0.1"


def state_path(device: str = "gpu") -> Path:
    return runtimes_root(device) / "server.json"


def last_error_path(device: str = "gpu") -> Path:
    return runtimes_root(device) / "last_error.json"


def write_last_error(message: str, device: str = "gpu") -> None:
    path = last_error_path(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"error": message}, ensure_ascii=False), encoding="utf-8")


def state_served_model_name(device: str = "gpu") -> str:
    """Served id recorded by the last spawn. Empty when no state file / pid."""
    path = state_path(device)
    if not path.is_file():
        return ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("served_model_name") or "").strip()


def read_last_error(device: str = "gpu") -> str | None:
    path = last_error_path(device)
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


def clear_last_error(device: str = "gpu") -> None:
    last_error_path(device).unlink(missing_ok=True)


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
    """CLI ``--quantization`` only when config.json ``quant_method`` is AutoAWQ/GPTQ.

    Recommend writes ``awq``. An id that merely contains AWQ (llmcompressor
    Hermes-*-AWQ-4bit) is compressed-tensors — passing ``awq`` dies. A BF16
    search hit (Dolphin) with leftover ``awq`` dies looking for an AWQ file.
    """
    hid = configured_model_id(settings)
    from hermes_cli.vllm_runtime.inventory import (
        cached_model_config, parse_quantization, repo_quant_method)

    method = repo_quant_method(hid, cached_model_config(hid))
    if method == "compressed-tensors":
        return ""
    if method in {"awq", "gptq"}:
        return method
    quant = str(settings.get("quantization") or "").strip()
    if not quant:
        return ""
    parsed = parse_quantization(hid)
    return quant if parsed == quant else ""


def humanize_serve_error(raw: str | None, *, model: str = "") -> str | None:
    blocked = configured_unservable_reason({"model": model} if model else None)
    if blocked:
        return blocked
    text = (raw or "").strip()
    lower = text.lower()
    if "cannot find the config file for awq" in lower:
        return LEFTOVER_AWQ_MSG
    if "draft_model_config" in lower or "qwen3_dspark" in lower:
        from hermes_cli.vllm_runtime.inventory import UNSERVABLE_DSPARK_MSG

        return UNSERVABLE_DSPARK_MSG
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


def last_serve_error_line(log_path: Path | None = None, device: str = "gpu") -> str | None:
    """Last real error line from the engine log — not a restart-loop breadcrumb."""
    if log_path:
        path = Path(log_path)
    else:
        from hermes_cli.vllm_runtime.venv import server_log_read_paths

        path = next(
            (p for p in server_log_read_paths(device) if p.is_file()),
            server_log_path(device),
        )
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
        if any(needle in lower for needle in _LOG_WRAPPER_NEEDLES):
            continue
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
    """``local_runtime.vllm`` merged over DEFAULT_CONFIG so a partial section still serves.

    Device ``cpu`` (or a legacy ``engine: vllm-cpu``) serves the BF16 default
    when the model is empty or still the GPU shipped AWQ id. A legacy
    ``vllm-cpu`` config with any other id keeps that explicit choice. A nested
    ``vllm.cpu.model`` is the CPU checkpoint and does not replace the GPU model.
    """
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.local_engines import vllm_device_from_config
    from hermes_cli.vllm_runtime.device import ENGINE_CPU

    defaults = dict(DEFAULT_CONFIG["local_runtime"]["vllm"])
    local = (config or {}).get("local_runtime") or {}
    if not isinstance(local, dict):
        local = {}
    override = local.get("vllm") or {}
    cpu_block: dict = {}
    if isinstance(override, dict):
        if isinstance(override.get("cpu"), dict):
            cpu_block = dict(override["cpu"])
        defaults.update({k: v for k, v in override.items() if k not in ("cpu", "device")})
    defaults.pop("cpu", None)
    defaults.pop("device", None)
    if vllm_device_from_config(config) != CPU:
        return defaults
    from hermes_cli.vllm_runtime.recommend import (
        as_vllm_config, gpu_shipped_model, recommend_vllm_cpu,
    )

    if str(cpu_block.get("model") or "").strip():
        defaults.update({k: v for k, v in cpu_block.items() if k != "device"})
        return defaults
    model = str(defaults.get("model") or "").strip()
    legacy = str(local.get("engine") or "").strip().lower().replace("_", "-") == ENGINE_CPU
    if legacy and model and model != gpu_shipped_model():
        return defaults
    defaults.update(as_vllm_config(recommend_vllm_cpu()))
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


def openai_base_url(settings: dict, *, port: int | None = None,
                    device: str = "gpu") -> str:
    if port is not None:
        p = int(port)
    else:
        p = int(settings.get("port") or 0) or default_listen_port(device)
    return f"http://{client_host(settings)}:{p}/v1"


_ALLOW_LONG_ENV = "VLLM_ALLOW_LONG_MAX_MODEL_LEN"


def _requested_max_model_len(settings: dict) -> int:
    from hermes_cli.vllm_runtime.recommend import MIN_CONTEXT

    raw = settings.get("max_model_len")
    try:
        requested = int(raw) if raw not in (None, "") else MIN_CONTEXT
    except (TypeError, ValueError):
        requested = MIN_CONTEXT
    return requested if requested > 0 else MIN_CONTEXT


def serve_len_and_rope(settings: dict) -> tuple[int, dict | None]:
    """``--max-model-len`` and optional YaRN ``--rope-scaling``.

    Hermes' 64k floor is a *request*. Serve ``min(request, native)`` unless the
    checkpoint's own ``rope_scaling`` yarn dict documents a higher length —
    never ALLOW_LONG to fake 64k over a 40960 Qwen3. 30B-A3B-2507 is also
    capped at 64k so leftover ``max_model_len: 262144`` cannot over-serve.
    """
    requested = _requested_max_model_len(settings)
    hid = configured_model_id(settings)
    from hermes_cli.vllm_runtime.recommend import serve_len_cap

    cap = serve_len_cap(hid)
    if cap is not None:
        requested = min(requested, cap)
    from hermes_cli.vllm_runtime.inventory import (
        cached_model_config, documented_yarn_rope, native_max_model_len)

    config = cached_model_config(hid)
    native = native_max_model_len(config)
    yarn = documented_yarn_rope(config, requested=requested, native=native)
    if yarn is not None:
        return requested, yarn
    if native is None:
        return requested, None
    return min(requested, native), None


def serve_environ(executable: str | Path, base: dict | None = None, *,
                  device: str = "gpu") -> dict[str, str]:
    """Child env for ``vllm serve``. Never inherit ALLOW_LONG — it starts then dies.

    CPU sets ``VLLM_TARGET_DEVICE=cpu`` before vLLM imports so
    ``platforms/cuda.py`` is not loaded. That module imports
    ``vllm._C_stable_libtorch``, which needs ``libtorch_cuda.so`` — the CPU
    wheel does not ship it. Unset means vLLM defaults the target to cuda.
    GPU never forces the CPU platform, including when the parent inherited it.
    """
    env = dict(os.environ if base is None else base)
    bindir = str(Path(executable).parent)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env.pop(_ALLOW_LONG_ENV, None)
    if normalize_device(device) == CPU:
        env["VLLM_TARGET_DEVICE"] = "cpu"
        lib = _intel_openmp_library(Path(executable))
        if lib:
            prior = [part for part in env.get("LD_PRELOAD", "").split(":") if part]
            if lib not in prior:
                prior.insert(0, lib)
            env["LD_PRELOAD"] = ":".join(prior)
    elif env.get("VLLM_TARGET_DEVICE", "").strip().lower() == "cpu":
        env.pop("VLLM_TARGET_DEVICE", None)
    return env


def serve_argv(executable: str | Path, settings: dict, *, device: str = "gpu") -> list[str]:
    """``vllm serve`` argv. Bind host comes from settings; 1-click default is loopback.

    CPU uses the official CPU-built ``vllm`` — never ``--device cpu`` (that is
    the CUDA-wheel trap). GPU-only flags stay off the CPU argv.
    """
    model = str(settings.get("model") or "").strip()
    if not model:
        raise ValueError("local_runtime.vllm.model is required")
    device = normalize_device(device)
    port = int(settings.get("port") or 0) or default_listen_port(device)
    max_len, rope = serve_len_and_rope(settings)
    argv = [
        str(executable), "serve", model,
        "--host", bind_host(settings),
        "--port", str(port),
        "--max-model-len", str(max_len),
        "--enable-auto-tool-choice",
        "--tool-call-parser", str(settings.get("tool_call_parser") or "hermes"),
    ]
    if device != CPU:
        argv.extend([
            "--gpu-memory-utilization",
            str(settings.get("gpu_memory_utilization") or 0.75),
        ])
    if rope:
        argv.extend(["--rope-scaling", json.dumps(rope, separators=(",", ":"))])
    quant = serve_quantization(settings)
    if quant:
        argv.extend(["--quantization", quant])
    from hermes_cli.vllm_runtime.recommend import is_qwen3_30b_a3b_2507

    kv = str(settings.get("kv_cache_dtype") or "").strip()
    # 30B-A3B-2507 BF16 KV is ~26 GiB with weights — leftover empty/bf16
    # must not drop --kv-cache-dtype on a 24 GB card. CPU has no GPU KV pool.
    if device != CPU and is_qwen3_30b_a3b_2507(model):
        kv = "fp8"
    if device != CPU and kv:
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


def pick_listen_port(preferred: int = 0, *, device: str = "gpu") -> int:
    """Like llama.cpp: try the engine's stable default, else an ephemeral port.

    GPU prefers 18435, CPU 18436. Never 18434 / 8000 / 8080, and never the
    sibling vLLM default (GPU must not steal 18436; CPU must not steal 18435).
    """
    device = normalize_device(device)
    sibling = CPU_LISTEN_PORT if device != CPU else GPU_LISTEN_PORT
    candidate = preferred if preferred > 0 else default_listen_port(device)
    if candidate in RESERVED_PORTS and candidate != default_listen_port(device):
        candidate = default_listen_port(device)
    if candidate == sibling:
        candidate = default_listen_port(device)
    try:
        with socket.socket() as s:
            s.bind((_LOOPBACK, candidate))
            return candidate
    except OSError:
        logger.warning(
            "port %d busy; managed vLLM falling back to an ephemeral "
            "port — existing sessions may need a model re-pick", candidate)
        port = _free_port()
        while port in RESERVED_PORTS or port in USER_SERVER_PORTS:
            port = _free_port()
        return port


class VllmSupervisor:
    """Own one ``vllm serve`` process for the life of a Hermes session."""

    def __init__(self, settings: dict, *, executable: Path | None = None,
                 log_path: Path | None = None, device: str = "gpu"):
        self.device = normalize_device(device)
        self.settings = dict(settings)
        preferred = int(self.settings.get("port") or 0)
        self.port = pick_listen_port(preferred, device=self.device)
        self.settings["port"] = self.port
        self.executable = Path(executable) if executable else vllm_executable(self.device)
        self.log_path = log_path or server_log_path(self.device)
        self.proc: subprocess.Popen | None = None
        self._restarts = 0
        self._stopping = False
        self._watchdog: threading.Thread | None = None
        self._log_handle = None

    @property
    def base_url(self) -> str:
        return openai_base_url(self.settings, port=self.port, device=self.device)

    def _health_url(self) -> str:
        return f"http://{_LOOPBACK}:{self.port}/v1/models"

    def _spawn(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(self.log_path, "ab")  # noqa: SIM115
        cmd = serve_argv(self.executable, self.settings, device=self.device)
        env = serve_environ(self.executable, device=self.device)
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
            write_last_error(blocked, self.device)
            disable_auto_start()
            raise RuntimeError(blocked)
        if configured_cache_missing(self.settings):
            write_last_error(MODEL_REMOVED_MSG, self.device)
            disable_auto_start()
            raise RuntimeError(MODEL_REMOVED_MSG)
        clear_last_error(self.device)
        try:
            self._spawn()
            self._wait_ready(timeout_s)
        except Exception as exc:
            # Restore/stop of an in-flight boot must not poison last_error or
            # flip enabled=false while the replacement serve is starting.
            if self._stopping:
                raise
            crash = last_serve_error_line(self.log_path, device=self.device)
            if configured_cache_missing(self.settings):
                write_last_error(MODEL_REMOVED_MSG, self.device)
            else:
                write_last_error(
                    humanize_serve_error(crash or str(exc), model=hid)
                    or str(exc).strip()
                    or "vllm serve failed",
                    self.device,
                )
            disable_auto_start()
            raise
        self._write_state()
        self._watchdog = threading.Thread(
            target=self._watch, daemon=True, name="vllm-supervisor")
        self._watchdog.start()

    def _write_state(self) -> None:
        path = state_path(self.device)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "base_url": self.base_url,
            "pid": self.proc.pid if self.proc else None,
            "port": self.port,
            "bind": bind_host(self.settings),
            "device": self.device,
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
                write_last_error(MODEL_REMOVED_MSG, self.device)
                logger.error("%s", MODEL_REMOVED_MSG)
                disable_auto_start()
                self.stop()
                return
            blocked = configured_unservable_reason(self.settings)
            crash = last_serve_error_line(self.log_path, device=self.device)
            if blocked:
                write_last_error(blocked, self.device)
                logger.error("%s", blocked)
                disable_auto_start()
                self.stop()
                return
            if crash:
                write_last_error(humanize_serve_error(crash, model=hid) or crash, self.device)
            if is_fatal_serve_crash(crash, model=hid):
                logger.error("vllm serve fatal; not restarting: %s", crash)
                disable_auto_start()
                self.stop()
                return
            if self._restarts >= _MAX_CRASH_RESTARTS:
                write_last_error(
                    humanize_serve_error(crash, model=hid)
                    or f"vllm serve crashed {self._restarts} times — Stop, then Use another model",
                    self.device,
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
                write_last_error(MODEL_REMOVED_MSG, self.device)
                logger.error("%s", MODEL_REMOVED_MSG)
                disable_auto_start()
                self.stop()
                return
            self._restarts += 1
            try:
                self._spawn()
                self._wait_ready(READY_TIMEOUT_S)
                clear_last_error(self.device)
            except Exception as exc:  # noqa: BLE001
                logger.error("vllm serve restart failed: %s", exc)
                write_last_error(humanize_serve_error(str(exc), model=hid) or str(exc), self.device)
                if configured_cache_missing(self.settings) or is_fatal_serve_crash(str(exc), model=hid):
                    if configured_cache_missing(self.settings):
                        write_last_error(MODEL_REMOVED_MSG, self.device)
                    disable_auto_start()
                    self.stop()
                    return

    def stop(self) -> None:
        self._stopping = True
        state_path(self.device).unlink(missing_ok=True)
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
