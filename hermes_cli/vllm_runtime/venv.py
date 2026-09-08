"""Isolated venv for the managed vLLM engine.

Machine-scoped (same rule as llama.cpp zips): profiles share one PyTorch wheel
tree and must not fight over the port. Hermes creates this venv — the user does
not pip-install vLLM, does not pick a Python, and does not clone a third-party
installer. Empty ``local_runtime.vllm.python`` uses the interpreter already
running Hermes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def runtimes_root() -> Path:
    """``<default hermes root>/runtimes/vllm`` — not profile-scoped."""
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "runtimes" / "vllm"


def venv_dir() -> Path:
    return runtimes_root() / ".venv"


def venv_python() -> Path:
    if os.name == "nt":
        return venv_dir() / "Scripts" / "python.exe"
    return venv_dir() / "bin" / "python"


def vllm_executable() -> Path:
    """``vllm`` console script inside the isolated venv (missing until install)."""
    if os.name == "nt":
        return venv_dir() / "Scripts" / "vllm.exe"
    return venv_dir() / "bin" / "vllm"


def venv_ready() -> bool:
    return vllm_executable().is_file()


def install_log_path() -> Path:
    return runtimes_root() / "install.log"


def manifest_path() -> Path:
    return runtimes_root() / "manifest.json"


def resolve_venv_python(pin: str | None = "") -> str:
    """Interpreter spec for the isolated venv: a path, or a ``3.12`` pin for uv.

    Empty pin uses Hermes' CPython when it is 3.12+, otherwise the first
    ``python3.{12,13,14}`` on PATH, otherwise ``3.12`` for ``uv --managed-python``.
    FlashInfer (pulled in by current vLLM) subscripts ``array.array`` and crashes
    on 3.11. The user does not install a second Python — uv fetches one.
    """
    pin = (pin or "").strip()
    if pin:
        as_path = Path(pin).expanduser()
        if as_path.is_file():
            return str(as_path)
        names = [pin, f"python{pin}"] if pin[0].isdigit() else [pin]
        for name in names:
            found = shutil.which(name)
            if found:
                return found
        if pin[0].isdigit() and shutil.which("uv"):
            return pin
        running = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            f"need CPython {pin} to create the vLLM venv; install that interpreter "
            f"or clear local_runtime.vllm.python to let Hermes pick 3.12+ "
            f"(running {running})"
        )
    if sys.version_info >= (3, 12):
        return str(Path(sys.executable))
    for minor in (12, 13, 14):
        found = shutil.which(f"python3.{minor}")
        if found:
            return found
    if shutil.which("uv"):
        return "3.12"
    raise RuntimeError(
        "need CPython 3.12+ for the vLLM venv (FlashInfer). Install Python 3.12+ "
        "or uv (which can fetch it), or set local_runtime.vllm.python to that interpreter"
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
        "the managed engine uses an isolated runtimes/vllm/.venv"
    )


def _stream(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("vLLM install: %s", " ".join(cmd))
    with log_path.open("ab") as log:
        log.write(f"+ {' '.join(cmd)}\n".encode())
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"vLLM install command failed rc={proc.returncode} ({log_path}): {' '.join(cmd)}"
        )


def _write_manifest(python_exe: Path) -> None:
    payload = {
        "python": str(python_exe),
        "python_version": subprocess.check_output(
            [str(python_exe), "-c",
             "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            text=True,
        ).strip(),
    }
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _venv_python_version(exe: Path) -> str:
    try:
        return subprocess.check_output(
            [str(exe), "-c",
             "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            text=True, timeout=30,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _needs_recreate(creator: str) -> bool:
    py = venv_python()
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


def _assert_cuda(py: Path) -> None:
    probe = subprocess.run(
        [str(py), "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
        capture_output=True, text=True, timeout=120,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "vLLM venv PyTorch has no CUDA — GPU inference would run on CPU. "
            f"See {install_log_path()}"
        )


def _ninja_executable() -> Path:
    if os.name == "nt":
        return venv_dir() / "Scripts" / "ninja.exe"
    return venv_dir() / "bin" / "ninja"


def ensure_vllm_venv(python_pin: str | None = "", *, upgrade: bool = False) -> Path:
    """Create the isolated venv if needed and pip-install ``vllm``. Returns the ``vllm`` exe.

    Uses ``uv`` when it is already on PATH (CUDA torch via ``--torch-backend=auto``).
    Otherwise stdlib ``venv`` + ``python -m pip``. Never installs into ``sys.prefix``.
    """
    creator = resolve_venv_python(python_pin)
    dest = venv_dir()
    _assert_isolated(dest)
    log = install_log_path()
    uv = shutil.which("uv")

    if _needs_recreate(creator) and dest.exists():
        logger.info("recreating vLLM venv (Python pin changed)")
        shutil.rmtree(dest)

    if not venv_python().is_file():
        _create_venv(creator, dest, log)

    py = venv_python()
    _assert_isolated(py)
    need_wheels = not venv_ready() or upgrade or not _ninja_executable().is_file()
    if need_wheels:
        packages = ["vllm", "ninja"]
        if uv:
            _stream([uv, "pip", "install", "--python", str(py),
                     "--upgrade", *packages, "--torch-backend=auto"], log)
        else:
            _stream([str(py), "-m", "pip", "install", "--upgrade", "pip"], log)
            _stream([str(py), "-m", "pip", "install", "--upgrade", *packages], log)
        _assert_cuda(py)
    exe = vllm_executable()
    if not exe.is_file():
        raise RuntimeError(f"vLLM install finished but {exe} is missing")
    _write_manifest(py)
    return exe
