"""Detect a foreign LLM that would fight managed vLLM for the GPU.

Managed llama.cpp is stopped by the coordinator before this runs. Anything still
resident — another vLLM, Ollama, llama-server, LM Studio — must be stopped by
the user; one GPU cannot hold two weight files.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Well-known *other* servers, plus Hermes preferred ports when a leftover is
# not our state pid. Never treat 8000/8080 as ours.
FOREIGN_HTTP_PORTS = (8000, 8080, 11434, 1234, 18434, 18435)

_LLM_NAME_NEEDLES = (
    "vllm", "llama-server", "ollama", "enginecore",
    "text-generation-webui", "lmstudio", "lm-studio", "kobold",
)
_DESKTOP_SKIP = (
    "kwin", "brave", "chrome", "chromium", "firefox", "xorg", "xwayland",
    "gnome-shell", "plasmashell", "cursor", "code", "electron",
)


@dataclass(frozen=True)
class OccupyingLlm:
    kind: str
    detail: str
    port: int | None = None
    pid: int | None = None


class OccupyingLlmError(RuntimeError):
    """A foreign LLM still holds the GPU; managed vLLM must not start beside it."""


def occupancy_stop_message(hits: list[OccupyingLlm]) -> str | None:
    if not hits:
        return None
    details = "; ".join(h.detail for h in hits)
    return (
        f"Another LLM is already running ({details}). "
        "Stop it so managed vLLM can use the GPU."
    )


def gpu_process_is_foreign_llm(name: str, pid: int, our_pids: set[int]) -> bool:
    """Host-independent: process name as data, no sys.platform."""
    if pid in our_pids or pid <= 0:
        return False
    lower = name.lower()
    if any(skip in lower for skip in _DESKTOP_SKIP):
        return False
    return any(needle in lower for needle in _LLM_NAME_NEEDLES)


def parse_compute_app_usage(csv_text: str) -> list[tuple[int, str, int]]:
    """nvidia-smi ``pid,process_name,used_memory`` rows → (pid, name, used_bytes).

    Accepts ``nounits`` (bare MiB) or ``1234 MiB``. Missing/N/A memory is 0.
    """
    rows: list[tuple[int, str, int]] = []
    for raw in csv_text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("pid"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        with suppress(ValueError):
            pid = int(parts[0])
            name = parts[1]
            used = 0
            if len(parts) >= 3:
                token = parts[2].split()[0]
                if token.lower() not in {"n/a", "[n/a]", ""}:
                    used = int(float(token)) << 20
            if pid > 0 and name:
                rows.append((pid, name, used))
    return rows


def parse_compute_app_rows(csv_text: str) -> list[tuple[int, str]]:
    """nvidia-smi ``pid,process_name,used_memory`` rows, header optional."""
    return [(pid, name) for pid, name, _used in parse_compute_app_usage(csv_text)]


def _pid_alive(pid: int) -> bool:
    if not pid or pid < 0:
        return False
    with suppress(Exception):
        import psutil  # type: ignore

        return psutil.pid_exists(pid)
    return True


def _read_state(path) -> dict | None:
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return None
    return state if isinstance(state, dict) else None


def _port_from_state(state: dict) -> int | None:
    raw = state.get("port")
    with suppress(TypeError, ValueError):
        port = int(raw or 0)
        if port > 0:
            return port
    from urllib.parse import urlparse

    with suppress(ValueError):
        parsed = urlparse(str(state.get("base_url") or ""))
        if parsed.port:
            return int(parsed.port)
    return None


def descendant_pids(pid: int) -> set[int]:
    """Children of a supervised serve pid. EngineCore (not the parent) holds VRAM."""
    found: set[int] = set()
    if not pid or pid <= 0:
        return found
    with suppress(Exception):
        import psutil  # type: ignore

        found = {int(c.pid) for c in psutil.Process(pid).children(recursive=True)}
    return found


def expand_managed_pids(root_pids: set[int], *, children_of=None) -> set[int]:
    """Include serve descendants so occupancy does not flag our own EngineCore."""
    out = set(root_pids)
    lookup = descendant_pids if children_of is None else children_of
    for pid in root_pids:
        out |= set(lookup(pid) or ())
    return out


def _collect_managed_state(path) -> tuple[set[int], set[int]]:
    pids: set[int] = set()
    ports: set[int] = set()
    state = _read_state(path)
    if not state:
        return pids, ports
    pid = int(state.get("pid") or 0)
    if pid and _pid_alive(pid):
        pids.add(pid)
        port = _port_from_state(state)
        if port:
            ports.add(port)
    return pids, ports


def _our_managed() -> tuple[set[int], set[int]]:
    """Pids and listen ports of *this install's* supervised servers (skip as foreign)."""
    pids: set[int] = set()
    ports: set[int] = set()
    with suppress(Exception):
        from hermes_cli.vllm_runtime.supervisor import state_path as vllm_state

        vp, vo = _collect_managed_state(vllm_state())
        pids |= vp
        ports |= vo
    with suppress(Exception):
        from hermes_cli.local_runtime.supervisor import state_path as llama_state

        lp, lo = _collect_managed_state(llama_state())
        pids |= lp
        ports |= lo
    return expand_managed_pids(pids), ports


def _http_json(url: str, timeout_s: float = 1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError, ValueError):
        return 0, None


def probe_foreign_http(port: int) -> OccupyingLlm | None:
    """Fingerprint a loopback OpenAI-compat / llama-server / Ollama listener."""
    from hermes_cli.local_runtime.detect import probe_port

    llama = probe_port(port)
    if llama:
        return OccupyingLlm(
            "llamacpp", f"llama-server on http://127.0.0.1:{port}", port)
    root = f"http://127.0.0.1:{port}"
    models_status, body = _http_json(f"{root}/v1/models")
    if models_status == 0:
        models_status, body = _http_json(f"{root}/api/tags")  # Ollama native
        if models_status == 200:
            return OccupyingLlm("ollama", f"Ollama on {root}", port)
        return None
    if models_status in (200, 401):
        kind = "openai-compat"
        if isinstance(body, dict):
            owned = ""
            data = body.get("data")
            if isinstance(data, list) and data and isinstance(data[0], dict):
                owned = str(data[0].get("owned_by") or "").lower()
            if owned == "vllm" or "vllm" in json.dumps(body).lower():
                kind = "vllm"
        if port == 11434:
            kind = "ollama"
        return OccupyingLlm(kind, f"{kind} API on {root}", port)
    return None


def _gpu_occupants(our_pids: set[int],
                   rows: list[tuple[int, str]] | None) -> list[OccupyingLlm]:
    if rows is None:
        rows = _live_compute_apps()
    hits = []
    for pid, name in rows:
        if gpu_process_is_foreign_llm(name, pid, our_pids):
            hits.append(OccupyingLlm("gpu-process", f"{name} (pid {pid})", pid=pid))
    return hits


def _live_compute_app_usage() -> list[tuple[int, str, int]] | None:
    """Live compute-app rows, or None when nvidia-smi is missing/fails (not the same as idle)."""
    from hermes_cli.local_runtime.hardware import _nvidia_smi_path

    exe = _nvidia_smi_path()
    if not exe:
        return None
    import subprocess

    with suppress(OSError, subprocess.TimeoutExpired):
        out = subprocess.run(
            [exe, "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return parse_compute_app_usage(out.stdout)
    return None


def _live_compute_apps() -> list[tuple[int, str]]:
    rows = _live_compute_app_usage()
    if rows is None:
        return []
    return [(pid, name) for pid, name, _used in rows]


def gpu_vram_attribution(
        *,
        rows: list[tuple[int, str, int]] | None = None,
        our_pids: set[int] | None = None,
) -> dict[str, int | bool | None]:
    """Split live compute-app VRAM into this managed engine vs everyone else.

    ``None`` used-bytes means the probe failed — do not render as 0. Inject
    rows/pids in tests; a live miss must not raise.
    """
    live = rows if rows is not None else _live_compute_app_usage()
    if live is None:
        return {
            "vram_engine_bytes": None,
            "vram_other_bytes": None,
            "occupancy_foreign": False,
        }
    ours = our_pids if our_pids is not None else _our_managed()[0]
    engine = other = 0
    foreign = False
    for pid, name, used in live:
        if pid in ours:
            engine += used
        else:
            other += used
            if gpu_process_is_foreign_llm(name, pid, ours):
                foreign = True
    return {
        "vram_engine_bytes": engine,
        "vram_other_bytes": other,
        "occupancy_foreign": foreign,
    }


def discover_occupying_llms(
        *,
        ports: tuple[int, ...] | None = None,
        gpu_rows: list[tuple[int, str]] | None = None,
        our_pids: set[int] | None = None,
        our_ports: set[int] | None = None,
) -> list[OccupyingLlm]:
    """Foreign listeners + GPU processes. Inject ports/rows in tests; live GPU is optional."""
    managed_pids, managed_ports = _our_managed()
    ours = our_pids if our_pids is not None else managed_pids
    skip_ports = our_ports if our_ports is not None else managed_ports
    hits: list[OccupyingLlm] = []
    seen: set[str] = set()
    for port in (FOREIGN_HTTP_PORTS if ports is None else ports):
        if port in skip_ports:
            continue
        hit = probe_foreign_http(port)
        if hit and hit.detail not in seen:
            hits.append(hit)
            seen.add(hit.detail)
    for hit in _gpu_occupants(ours, gpu_rows):
        if hit.pid in ours:
            continue
        if hit.detail not in seen:
            hits.append(hit)
            seen.add(hit.detail)
    return hits


def require_gpu_free() -> None:
    msg = occupancy_stop_message(discover_occupying_llms())
    if msg:
        raise OccupyingLlmError(msg)
