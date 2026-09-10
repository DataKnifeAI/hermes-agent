# Managed vLLM as a first-class local engine

Status: **CLI follow-up** — `feat/hermes-local-cli` / [#8](https://github.com/DataKnifeAI/hermes-agent/issues/8)
Audience: `hermes local` argparse parity with Desktop managed vLLM

Engine + Desktop ship on `feat/managed-vllm-local-engine`. This branch
keeps the CLI wrapper so use / status / start can catch up.

## What shipped

llama.cpp stays the default 1-click engine (`local_runtime.engine: llamacpp`,
port **18434**). vLLM is the managed sibling on Linux NVIDIA
(`engine: vllm`, preferred port **18435**, then an ephemeral port). Never
18434 (llama) and never 8000/8080 (a user's own vLLM/Ollama).

Official catalog is Qwen3 AWQ (8B on 16 GB, 14B on 24 GB, plus larger
tiers). Hermes keeps a **64k** tool-loop floor — cards that cannot hold
it stay infeasible; there is no silent ctx shrink. vLLM **hides**
unfitting official rows until Desktop **Show models that don't fit** or
`hermes local ls --show-unfitting`. llama.cpp still shows them.

Logs live under the profile Hermes home (`get_hermes_home()/logs/`,
user-facing `{display_hermes_home()}/logs/vllm-server.log`). Never
hardcode `~/.hermes`. The isolated venv is machine-scoped
(`get_default_hermes_root()/runtimes/vllm/`). HF weights stay in the
Hugging Face cache.

Desktop **Settings → Providers → Local Models** is the 1-click path
(engine switch → Install → Use). Switching engines stops the other
supervised server.

## Code

| Piece | Where |
|---|---|
| Config (`engine`, `local_runtime.vllm.*`) | `hermes_cli/config_defaults.py` |
| Stop-the-other-engine | `hermes_cli/local_engines.py` |
| venv / supervisor / recommend / inventory | `hermes_cli/vllm_runtime/` |
| Desktop HTTP | `hermes_cli/web_routers/local_models_engine.py` |
| Desktop UI | `apps/desktop/src/app/settings/local-models-settings.tsx`, `vllm-models-pane.tsx` |
| User guide | `website/docs/user-guide/local-models.md` |
| CLI argparse | `hermes_cli/subcommands/local.py`, `hermes_cli/local_cmd.py` |

`hermes_cli/local_runtime/` remains llama.cpp only.

## Still true (do not regress)

- Do not vendor [enodios](https://github.com/DataKnifeAI/enodios).
- Do not replace llama.cpp as the default.
- Do not make Docker the default; native host vLLM only.
- Do not regex-edit `config.yaml`. Use `save_config_value()`.
- No `HERMES_VLLM_*` behavioral env vars.
- Do not stuff Python/CUDA into `hermes_cli/local_runtime/`.
- Activate overwrites loopback/empty `base_url` only — a remote vLLM
  endpoint stays remote.
- Mid-conversation engine swaps are next-session / explicit restart.

## This branch

Close these gaps before merging `hermes local` as product:

- CLI `hermes local use` still only pins the provider. Desktop **Use**
  serves the cached id. Do not rename `use`.
- `hermes local status` does not print Starting/Ready / In use
  (`engine_state`).
- `hermes local start` can still fail the tool-call bench even when
  serve is up. Do not change CLI start semantics in a drive-by.

Call the same engine helpers as Desktop (`local_engines`,
`vllm_runtime`, `/api/local-models/*`). Do not fork them.
