"""Isolated venv for the managed vLLM engine.

Machine-scoped (same rule as llama.cpp zips): profiles share one PyTorch wheel
tree and must not fight over the port. Hermes creates this venv — the user does
not pip-install vLLM, does not pick a Python, and does not clone a third-party
installer. Empty ``local_runtime.vllm.python`` uses Hermes' CPython when it is
3.12 or 3.13. Python 3.14 is outside vLLM's supported range; uv fetches 3.12.

The CPU venv installs the official ``+cpu`` wheel from
``https://wheels.vllm.ai/<version>/cpu``. The PyPI ``vllm`` package is the CUDA
build; ``--torch-backend=cpu`` only swaps torch and still leaves
``libtorch_cuda.so`` on the import path.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from hermes_cli.vllm_runtime.device import (
    CPU, install_log_name, normalize_device, runtime_leaf, server_log_name,
)

logger = logging.getLogger(__name__)


def runtimes_root(device: str = "gpu") -> Path:
    """``<default hermes root>/runtimes/vllm`` or ``…/vllm-cpu`` — not profile-scoped."""
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "runtimes" / runtime_leaf(device)


def venv_dir(device: str = "gpu") -> Path:
    return runtimes_root(device) / ".venv"


def venv_python(device: str = "gpu") -> Path:
    if os.name == "nt":
        return venv_dir(device) / "Scripts" / "python.exe"
    return venv_dir(device) / "bin" / "python"


def vllm_executable(device: str = "gpu") -> Path:
    """``vllm`` console script inside the isolated venv (missing until install)."""
    if os.name == "nt":
        return venv_dir(device) / "Scripts" / "vllm.exe"
    return venv_dir(device) / "bin" / "vllm"


def venv_ready(device: str = "gpu") -> bool:
    return vllm_executable(device).is_file()


def hermes_logs_dir() -> Path:
    """Same directory ``hermes logs`` lists and llama-server.log uses.

    Wheels stay under ``runtimes/vllm/`` or ``runtimes/vllm-cpu/``. Only the
    log files join the central log dir (profile-aware via ``get_hermes_home()``).
    """
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "logs"


def server_log_path(device: str = "gpu") -> Path:
    return hermes_logs_dir() / server_log_name(device)


def install_log_path(device: str = "gpu") -> Path:
    return hermes_logs_dir() / install_log_name(device)


def server_log_read_paths(device: str = "gpu") -> tuple[Path, ...]:
    """Write target first; leftover runtime-dir files from older builds last."""
    return (server_log_path(device), runtimes_root(device) / server_log_name(device))


def install_log_read_paths(device: str = "gpu") -> tuple[Path, ...]:
    return (install_log_path(device), runtimes_root(device) / "install.log")


def server_log_hint(device: str = "gpu") -> str:
    from hermes_constants import display_hermes_home

    return f"{display_hermes_home()}/logs/{server_log_name(device)}"


def manifest_path(device: str = "gpu") -> Path:
    return runtimes_root(device) / "manifest.json"


# vLLM documents CPython 3.10–3.13. 3.14 warns at startup and is not a supported
# runtime (the CPU venv on 3.14.7 still imported the CUDA extension). FlashInfer,
# pulled in by the GPU wheel, crashes on 3.11. The managed venv is 3.12 or 3.13.
_PY_MIN = (3, 12)
_PY_MAX = (3, 14)  # exclusive
_CPU_WHEEL_INDEX = "https://wheels.vllm.ai/{version}/cpu"
_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _py_pair(text: str) -> tuple[int, int] | None:
    bits = (text or "").strip().split(".")
    if len(bits) < 2 or not bits[0].isdigit() or not bits[1].isdigit():
        return None
    return int(bits[0]), int(bits[1])


def _py_supported(pair: tuple[int, int] | None) -> bool:
    return pair is not None and _PY_MIN <= pair < _PY_MAX


def _py_unsupported_message(got: str) -> str:
    return (
        "vLLM needs CPython >=3.12,<3.14 (3.14 is unsupported; FlashInfer "
        f"crashes on 3.11); got {got}. Clear local_runtime.vllm.python to let "
        "Hermes fetch 3.12 with uv"
    )


def _running_pair() -> tuple[int, int]:
    return (sys.version_info[0], sys.version_info[1])


def resolve_venv_python(pin: str | None = "") -> str:
    """Interpreter spec for the isolated venv: a path, or a ``3.12`` pin for uv.

    Empty pin uses Hermes' CPython when it is 3.12 or 3.13, otherwise the first
    ``python3.13`` / ``python3.12`` on PATH, otherwise ``3.12`` for
    ``uv --managed-python``. Never 3.14, even when that is the Hermes interpreter.
    The user does not install a second Python — uv fetches one.
    """
    pin = (pin or "").strip()
    if pin:
        as_path = Path(pin).expanduser()
        if as_path.is_file():
            have = _venv_python_version(as_path)
            if not _py_supported(_py_pair(have)):
                raise RuntimeError(_py_unsupported_message(have or str(as_path)))
            return str(as_path)
        names = [pin, f"python{pin}"] if pin[0].isdigit() else [pin]
        for name in names:
            found = shutil.which(name)
            if found:
                have = _venv_python_version(Path(found))
                if not _py_supported(_py_pair(have)):
                    raise RuntimeError(_py_unsupported_message(have or found))
                return found
        pair = _py_pair(pin) if pin[0].isdigit() else None
        if pair is not None and not _py_supported(pair):
            raise RuntimeError(_py_unsupported_message(pin))
        if pin[0].isdigit() and shutil.which("uv"):
            return pin
        running = f"{_running_pair()[0]}.{_running_pair()[1]}"
        raise RuntimeError(
            f"need CPython {pin} to create the vLLM venv (supported >=3.12,<3.14); "
            f"install that interpreter or clear local_runtime.vllm.python "
            f"(running {running})"
        )
    if _py_supported(_running_pair()):
        return str(Path(sys.executable))
    for minor in (13, 12):
        found = shutil.which(f"python3.{minor}")
        if found:
            return found
    if shutil.which("uv"):
        return "3.12"
    running = f"{_running_pair()[0]}.{_running_pair()[1]}"
    raise RuntimeError(
        "need CPython >=3.12,<3.14 for the vLLM venv (3.14 is unsupported; "
        "FlashInfer crashes on 3.11). Install Python 3.12 or 3.13, or uv "
        f"(which can fetch 3.12). Running {running}"
    )


def _hermes_prefix() -> Path:
    return Path(sys.prefix).resolve()


def _assert_isolated(target: Path) -> None:
    """Refuse to install wheels into the Hermes interpreter."""
    resolved = target.resolve()
    hermes = _hermes_prefix()
    try:
        resolved.relative_to(hermes)
    except ValueError:
        return
    raise RuntimeError(
        f"refusing to install vLLM into the Hermes venv ({hermes}); "
        "the managed engine uses an isolated runtimes/vllm/.venv or runtimes/vllm-cpu/.venv"
    )


def _uv_install_env() -> dict[str, str]:
    """Don't inherit Hermes' project uv config (``exclude-newer = 14 days``).

    Desktop ``serve`` cwd is the git checkout. Bare ``uv pip install`` then
    reads ``pyproject.toml`` and silently skips a PyPI release newer than the
    rolling cutoff — the Update button "succeeds" and vLLM stays put.
    """
    env = os.environ.copy()
    env["UV_NO_CONFIG"] = "1"
    return env


def _stream(cmd: list[str], log_path: Path, *, cwd: Path | None = None,
            env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("vLLM install: %s", " ".join(cmd))
    with log_path.open("ab") as log:
        log.write(f"+ {' '.join(cmd)}\n".encode())
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"vLLM install command failed rc={proc.returncode} ({log_path}): {' '.join(cmd)}"
        )


def _write_manifest(python_exe: Path, device: str = "gpu") -> None:
    payload = {
        "python": str(python_exe),
        "python_version": subprocess.check_output(
            [str(python_exe), "-c",
             "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            text=True,
        ).strip(),
        "device": normalize_device(device),
    }
    dest = manifest_path(device)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _venv_python_version(exe: Path) -> str:
    try:
        return subprocess.check_output(
            [str(exe), "-c",
             "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            text=True, timeout=30,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _managed_serve_alive(device: str) -> bool:
    """True when this device's supervised ``vllm serve`` still has a live pid.

    Recreating a venv under a running server deletes the interpreter it is
    using. CPU install must not do that to the GPU engine.
    """
    from hermes_cli.vllm_runtime.supervisor import state_path

    path = state_path(device)
    if not path.is_file():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not isinstance(raw, dict):
        return False
    try:
        pid = int(raw.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    return _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    """Process exists. ``os.kill(pid, 0)`` on Windows calls TerminateProcess."""
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError:
        if os.name == "nt":
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    return bool(psutil.pid_exists(pid))


def _needs_recreate(creator: str, device: str = "gpu") -> bool:
    py = venv_python(device)
    if not py.is_file():
        return True
    have = _venv_python_version(py)
    if Path(creator).is_file():
        want = _venv_python_version(Path(creator))
    else:
        want = creator
    return bool(have) and bool(want) and not have.startswith(want)


def _create_venv(creator: str, dest: Path, log: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "venv", "--python", creator, "--seed"]
        if not Path(creator).is_file():
            cmd.append("--managed-python")
        cmd.append(str(dest))
        _stream(cmd, log)
        return
    if not Path(creator).is_file():
        raise RuntimeError(
            f"need CPython {creator} on PATH to create the vLLM venv without uv"
        )
    _stream([creator, "-m", "venv", str(dest)], log)


def _assert_cuda(py: Path, device: str = "gpu") -> None:
    probe = subprocess.run(
        [str(py), "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
        capture_output=True, text=True, timeout=120,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "vLLM venv PyTorch has no CUDA — GPU inference would run on CPU. "
            f"See {install_log_path(device)}"
        )


# Version check first: forcing VLLM_TARGET_DEVICE=cpu on the CUDA wheel can
# skip platforms/cuda.py and still pass an import, then die later on
# libtorch_cuda.so. The official CPU artifact is published as ``<ver>+cpu``.
_CPU_IMPORT_PROBE = """
import os, sys
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
import importlib.metadata as m
ver = m.version("vllm")
if "+cpu" not in ver.lower():
    sys.stderr.write("cuda-wheel %s\\n" % ver)
    raise SystemExit(2)
import vllm
from vllm.platforms import current_platform
if not current_platform.is_cpu():
    sys.stderr.write("platform %s\\n" % type(current_platform).__name__)
    raise SystemExit(3)
import torch
if torch.cuda.is_available():
    sys.stderr.write("torch cuda wheel\\n")
    raise SystemExit(4)
"""


def _assert_cpu(py: Path, device: str = "cpu") -> None:
    """CPU venv must be the official ``+cpu`` wheel, importing as the CPU platform.

    Torch's CPU index is not enough: the PyPI ``vllm`` wheel is still the CUDA
    build and ``import vllm`` loads ``libtorch_cuda.so``.
    """
    env = os.environ.copy()
    env["VLLM_TARGET_DEVICE"] = "cpu"
    lib = _intel_openmp_library(py)
    if lib:
        prior = [p for p in env.get("LD_PRELOAD", "").split(":") if p]
        if lib not in prior:
            prior.insert(0, lib)
        env["LD_PRELOAD"] = ":".join(prior)
    probe = subprocess.run(
        [str(py), "-c", _CPU_IMPORT_PROBE],
        capture_output=True, text=True, timeout=180, env=env,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"rc={probe.returncode}"
        raise RuntimeError(
            "vLLM CPU venv is not the official CPU build "
            "(wheels.vllm.ai <version>+cpu, VLLM_TARGET_DEVICE=cpu). "
            f"{tail}. See {install_log_path(device)}"
        )


def _intel_openmp_library(python_or_vllm: Path) -> str:
    """``libiomp5.so`` from the intel-openmp wheel, if this venv shipped it.

    Official CPU wheels require it on ``LD_PRELOAD`` before the first import.
    """
    if os.name == "nt":
        return ""
    # bin/python is often a symlink into uv's managed CPython. Don't resolve —
    # libiomp5 lives in this venv's site-packages.
    root = Path(python_or_vllm).parent.parent
    found = sorted(root.glob("**/libiomp5.so"))
    return str(found[0]) if found else ""


def _ninja_executable(device: str = "gpu") -> Path:
    if os.name == "nt":
        return venv_dir(device) / "Scripts" / "ninja.exe"
    return venv_dir(device) / "bin" / "ninja"


def _install_env(device: str) -> dict[str, str]:
    env = _uv_install_env()
    if normalize_device(device) == CPU:
        env["VLLM_TARGET_DEVICE"] = "cpu"
    elif env.get("VLLM_TARGET_DEVICE", "").strip().lower() == "cpu":
        env.pop("VLLM_TARGET_DEVICE", None)
    return env


def _vllm_requirement(version: str | None, *, device: str) -> str:
    """Package spec. CPU pins ``vllm==<ver>+cpu``, which PyPI does not publish."""
    ver = (version or "").strip().split("+", 1)[0]
    if normalize_device(device) == CPU:
        if not ver:
            ver = latest_vllm_pypi_version().split("+", 1)[0]
        if not ver:
            raise RuntimeError(
                "could not resolve a vLLM release for the CPU wheel "
                "(https://wheels.vllm.ai/<version>/cpu). Refusing to pip install "
                "the PyPI CUDA wheel into the CPU venv"
            )
        return f"vllm=={ver}+cpu"
    if ver:
        return f"vllm=={ver}"
    return "vllm"


def _cpu_wheel_index(requirement: str) -> str:
    ver = requirement.split("==", 1)[1]
    base = ver.split("+", 1)[0]
    return _CPU_WHEEL_INDEX.format(version=base)


def _cpu_wheel_missing(device: str) -> bool:
    """True when the CPU venv has vLLM installed but it is not the ``+cpu`` build."""
    if normalize_device(device) != CPU or not venv_ready(device):
        return False
    return "+cpu" not in installed_vllm_version(device).lower()


def _install_packages(py: Path, dest: Path, log: Path, packages: list[str],
                      *, device: str, uv: str | None) -> None:
    env = _install_env(device)
    cpu = normalize_device(device) == CPU
    if uv:
        cmd = [uv, "--no-config", "pip", "install", "--python", str(py), "--upgrade"]
        if cpu:
            cmd.extend([
                "--extra-index-url", _cpu_wheel_index(packages[0]),
                "--index-strategy", "first-index",
                "--torch-backend=cpu",
            ])
        else:
            cmd.append("--torch-backend=auto")
        cmd.extend(packages)
        _stream(cmd, log, cwd=dest.parent, env=env)
        return
    _stream([str(py), "-m", "pip", "install", "--upgrade", "pip"], log, env=env)
    if cpu:
        _stream(
            [str(py), "-m", "pip", "install", "--upgrade", "torch",
             "--index-url", _TORCH_CPU_INDEX],
            log, env=env,
        )
        _stream(
            [str(py), "-m", "pip", "install", "--upgrade", *packages,
             "--extra-index-url", _cpu_wheel_index(packages[0])],
            log, env=env,
        )
        return
    _stream([str(py), "-m", "pip", "install", "--upgrade", *packages], log, env=env)


def ensure_vllm_venv(python_pin: str | None = "", *, upgrade: bool = False,
                     version: str | None = None, device: str = "gpu") -> Path:
    """Create the isolated venv if needed and pip-install ``vllm``. Returns the ``vllm`` exe.

    GPU uses CUDA torch (``--torch-backend=auto``) and ``_assert_cuda``.
    CPU installs ``vllm==<ver>+cpu`` from ``https://wheels.vllm.ai/<ver>/cpu``
    with ``--torch-backend=cpu`` and ``VLLM_TARGET_DEVICE=cpu``. Never
    ``pip install vllm`` (that is the CUDA wheel) and never ``--device cpu``
    on that wheel. Never installs into ``sys.prefix``.
    """
    device = normalize_device(device)
    creator = resolve_venv_python(python_pin)
    dest = venv_dir(device)
    _assert_isolated(dest)
    log = install_log_path(device)
    uv = shutil.which("uv")

    if _needs_recreate(creator, device) and dest.exists():
        if _managed_serve_alive(device):
            logger.warning(
                "not recreating the vLLM %s venv while its server is still running",
                device,
            )
        else:
            logger.info("recreating vLLM %s venv (Python pin changed)", device)
            shutil.rmtree(dest)

    if not venv_python(device).is_file():
        _create_venv(creator, dest, log)

    py = venv_python(device)
    _assert_isolated(py)
    need_wheels = not venv_ready(device) or upgrade or not _ninja_executable(device).is_file()
    if _cpu_wheel_missing(device):
        need_wheels = True
    if need_wheels:
        packages = [_vllm_requirement(version, device=device), "ninja"]
        _install_packages(py, dest, log, packages, device=device, uv=uv)
        if device == CPU:
            _assert_cpu(py, device)
        else:
            _assert_cuda(py, device)
    exe = vllm_executable(device)
    if not exe.is_file():
        raise RuntimeError(f"vLLM install finished but {exe} is missing")
    _write_manifest(py, device)
    return exe


def ensure_both_vllm_venvs(python_pin: str | None = "", *, upgrade: bool = False,
                           version: str | None = None) -> dict[str, Path | Exception]:
    """Install GPU and CPU isolated venvs. One failure does not skip the other."""
    out: dict[str, Path | Exception] = {}
    for device in ("gpu", "cpu"):
        try:
            out[device] = ensure_vllm_venv(
                python_pin, upgrade=upgrade, version=version, device=device)
        except Exception as exc:  # noqa: BLE001 — install both, report each
            logger.warning("vLLM %s venv install failed: %s", device, exc)
            out[device] = exc
    return out


# Status and hardware poll both venvs. Re-probe only when pip rewrites the
# venv python or the ``vllm`` script (an upgrade of one device leaves the other).
_installed_version_cache: dict[str, tuple[int, int, str]] = {}


def installed_vllm_version(device: str = "gpu") -> str:
    """Version of the ``vllm`` package inside the isolated venv — never Hermes ``sys.prefix``."""
    py = venv_python(device)
    if not py.is_file():
        return ""
    _assert_isolated(py)
    exe = vllm_executable(device)
    try:
        py_ns = py.stat().st_mtime_ns
    except OSError:
        return ""
    try:
        exe_ns = exe.stat().st_mtime_ns if exe.is_file() else 0
    except OSError:
        exe_ns = 0
    key = str(py)
    cached = _installed_version_cache.get(key)
    if cached is not None and cached[0] == py_ns and cached[1] == exe_ns:
        return cached[2]
    try:
        version = subprocess.check_output(
            [str(py), "-c", "import importlib.metadata as m; print(m.version('vllm'))"],
            text=True, timeout=30,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    if version:
        _installed_version_cache[key] = (py_ns, exe_ns, version)
    return version


def _parse_pep440_head(s: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(s).split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits or 0))
    return tuple(parts)


def latest_vllm_pypi_version(*, timeout_s: float = 8) -> str:
    """Current vLLM release on PyPI. Empty on network failure."""
    import json
    import urllib.request

    req = urllib.request.Request(
        "https://pypi.org/pypi/vllm/json",
        headers={"User-Agent": "hermes-local-models"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            data = json.load(r)
    except (OSError, json.JSONDecodeError, TimeoutError):
        return ""
    info = data.get("info") if isinstance(data, dict) else None
    if not isinstance(info, dict):
        return ""
    return str(info.get("version") or "").strip()


def version_check_path(device: str = "gpu") -> Path:
    return runtimes_root(device) / "version-check.json"


def read_version_check(device: str = "gpu") -> dict:
    path = version_check_path(device)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_version_check(installed: str, latest: str, device: str = "gpu") -> dict:
    payload = {
        "installed": installed,
        "latest": latest,
        "update_available": bool(
            installed and latest and _parse_pep440_head(latest) > _parse_pep440_head(installed)
        ),
    }
    path = version_check_path(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def vllm_version_fields(*, check: bool = False, device: str = "gpu") -> dict:
    """Status extras: installed tag, last (or fresh) PyPI check, update flag."""
    installed = installed_vllm_version(device)
    if check:
        latest = latest_vllm_pypi_version()
        if latest:
            return {
                "tag": installed,
                "configured_tag": latest,
                **write_version_check(installed, latest, device),
            }
        remembered = read_version_check(device)
        return {
            "tag": installed,
            "configured_tag": str(remembered.get("latest") or installed),
            "update_available": False,
            "installed": installed,
            "latest": "",
        }
    remembered = read_version_check(device)
    latest = str(remembered.get("latest") or "")
    update = bool(remembered.get("update_available"))
    if installed and latest:
        update = _parse_pep440_head(latest) > _parse_pep440_head(installed)
    return {
        "tag": installed,
        "configured_tag": latest or installed,
        "update_available": update,
        "installed": installed,
        "latest": latest,
    }
