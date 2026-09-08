"""Doctor rows for the managed local engines (llama.cpp / vLLM)."""

from __future__ import annotations

from hermes_cli.doctor_report import Finding, check_info, check_ok, check_warn, doctor_check


@doctor_check(on_error="Managed local engine")
def _check_managed_local_engine(should_fix: bool, f: Finding) -> None:  # noqa: ARG001
    """venv, nvidia-smi, /v1/models, last bench — informational unless engine is vllm and broken."""
    from hermes_cli.config import load_config
    from hermes_cli.local_engines import engine_from_config
    from hermes_cli.local_runtime.hardware import _nvidia_smi_path
    from hermes_cli.vllm_runtime.bench import last_bench
    from hermes_cli.vllm_runtime.endpoint import resolve_vllm_endpoint
    from hermes_cli.vllm_runtime.venv import venv_ready

    cfg = load_config()
    engine = engine_from_config(cfg)
    check_info(f"engine: {engine}")
    smi = _nvidia_smi_path()
    if smi:
        check_ok("nvidia-smi", smi)
    else:
        check_warn("nvidia-smi", "not on PATH (vLLM needs an NVIDIA driver)")
        if engine == "vllm":
            f.manual_issues.append("nvidia-smi not found — managed vLLM needs an NVIDIA GPU")

    if engine == "vllm" or venv_ready():
        if venv_ready():
            check_ok("vLLM venv", "isolated runtimes/vllm/.venv")
        else:
            check_warn("vLLM venv", "missing — hermes local install")
            if engine == "vllm":
                f.issues.append("vLLM venv missing — run `hermes local install`")

    endpoint = resolve_vllm_endpoint()
    if endpoint:
        check_ok("/v1/models", endpoint.get("base_url", ""))
    elif engine == "vllm":
        check_warn("/v1/models", "managed vLLM not running — hermes local start")

    bench = last_bench()
    if not bench:
        return
    url = bench.get("url") or ""
    if bench.get("ok"):
        check_ok("last bench", url)
    else:
        check_warn("last bench", bench.get("error") or url)
