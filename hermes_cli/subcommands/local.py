"""``hermes local`` subcommand parser — managed llama.cpp / vLLM engines."""

from __future__ import annotations

from typing import Callable


def build_local_parser(subparsers, *, cmd_local: Callable) -> None:
    """Attach the ``local`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "local",
        help="Managed local inference (llama.cpp default, vLLM sibling)",
        description=(
            "Install, start, stop, and point the provider at a managed local "
            "engine. llama.cpp stays the default. vLLM is the Python/CUDA "
            "sibling Hermes installs into an isolated venv — you do not pip "
            "or edit config.yaml by hand. Stop frees VRAM for games without "
            "killing unrelated processes."
        ),
    )
    sub = parser.add_subparsers(dest="local_command")
    sub.add_parser(
        "status", aliases=["list", "ls"],
        help="Show engine, server, venv, and GPU probe (default)")
    engine = sub.add_parser(
        "engine", help="Show or set local_runtime.engine (llamacpp | vllm)")
    engine.add_argument(
        "engine_name", nargs="?", choices=["llamacpp", "vllm"],
        help="Engine to select; omit to print the current value")
    recommend = sub.add_parser(
        "recommend",
        help="Print VRAM tier + HF model; --apply writes local_runtime.vllm.*")
    recommend.add_argument(
        "--apply", action="store_true",
        help="Write recommendation into config.yaml (never .env)")
    sub.add_parser(
        "install",
        help="Install the configured engine (llama.cpp zip or isolated vLLM venv)")
    sub.add_parser(
        "start",
        help="Stop the other engine, start this one, activate the provider")
    sub.add_parser(
        "stop",
        help="Stop the managed server and free VRAM (config is unchanged)")
    sub.add_parser(
        "bench",
        help="Smoke-check GET /v1/models on the managed endpoint")
    sub.add_parser(
        "use",
        help="Point model.provider at the managed engine (loopback URL only)")
    parser.set_defaults(func=cmd_local)
