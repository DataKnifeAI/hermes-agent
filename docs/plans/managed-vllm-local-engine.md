# Managed vLLM as a first-class local engine

Status: design proposal (not yet implemented)
Audience: Desktop + CLI local-inference work
Reference clone (patterns only): `/mnt/game2/git/enodios` — [DataKnifeAI/enodios](https://github.com/DataKnifeAI/enodios)

## Goal

Seamless local inference in Hermes Desktop and the CLI. llama.cpp remains the
**default 1-click engine**. vLLM is a **managed alternative engine**, not a
“user starts vLLM themselves” detour.

Hermes installs the Python/CUDA stack, picks a Hugging Face model that fits
the GPU, stands up and supervises a **localhost** vLLM instance, and wires
the existing provider. The user never copies a `vllm serve` line, never
regex-edits `config.yaml`, and never clones a third-party installer.

vLLM stays the right tool for remote / high-throughput serving too. `--host`
is the only difference: `127.0.0.1` for the managed local case, a LAN or
public bind when the operator is running a remote endpoint. llama.cpp already
is a local OpenAI-compatible gateway; vLLM is the same pattern with a
Python/CUDA stack — more flexible on Linux NVIDIA, where official ggml
release zips ship **no CUDA asset**.

## Why now

Linux NVIDIA + the pinned llama.cpp tag (`local_runtime.tag`, currently
`b10679`) has no `ubuntu-cuda` asset. `hermes_cli/local_runtime/binaries.py`
raises `BinaryResolutionError` and falls through to Vulkan or CPU. Vulkan
works; it is not CUDA.

vLLM uses real CUDA via PyTorch. That is the point of offering it as a
managed engine on Linux NVIDIA, not a replacement for llama.cpp on Metal,
Windows CUDA zips, or CPU-only boxes.

Related upstream:

- [#3238](https://github.com/NousResearch/hermes-agent/issues/3238) /
  [PR #3242](https://github.com/NousResearch/hermes-agent/pull/3242) —
  command generation only (print a `vllm serve` snippet). **Not this work.**
- [#85852](https://github.com/NousResearch/hermes-agent/issues/85852) —
  managed llama.cpp runtime (the charter we must not overwrite).
- [#102565](https://github.com/NousResearch/hermes-agent/issues/102565) —
  no Linux CUDA zip at current llama.cpp tags.
- [DataKnifeAI/enodios](https://github.com/DataKnifeAI/enodios) — external
  installer that already proves the vLLM + Hermes loop on Linux NVIDIA.

## What we are not doing

- **Do not vendor** [enodios](https://github.com/DataKnifeAI/enodios). Do not
  add it as a submodule, do not shell out to `enodios`, do not copy the
  bash installer into the tree. **Port the patterns** listed below into
  Hermes-owned Python.
- **Do not replace llama.cpp.** It stays the default 1-click engine. The
  Local Models page keeps the GGUF catalog, quickstart, sideload, and eject
  flows for `engine: llamacpp`.
- **Do not make Docker the default.** Enodios documents
  `CUDA_ERROR_UNKNOWN` on `vllm/vllm-openai` images; native host vLLM is
  the managed path.
- **Do not regex-edit `config.yaml`.** Enodios `configure` does that.
  Hermes already has `save_config_value()` and the `provider: vllm` →
  `custom` alias. Use those.
- **Do not add `HERMES_*` behavioral env vars.** Knobs live in
  `config.yaml`. Secrets stay in `.env`. If an internal process needs an
  env mirror (e.g. `CUDA_VISIBLE_DEVICES` for the child), bridge it in
  code from config.
- **Do not stuff vLLM into `hermes_cli/local_runtime/`.** That package’s
  charter is llama.cpp: official release zips, GGUF catalog, router-mode
  `llama-server`, touch-generation readiness. A Python/CUDA venv and HF
  weights do not belong there.

## Patterns to port from enodios (not the repo)

Read `/mnt/game2/git/enodios` as a reference implementation. Re-express
each pattern in Hermes modules, config, and tests.

| Enodios pattern | Hermes home |
|---|---|
| VRAM tiers / `recommend` (`8gb` / `12gb` / `16gb` / `24gb` class; `nvidia-smi` total + free + other compute apps) | `hermes_cli/vllm_runtime/recommend.py` — data, not env exports |
| `vllm serve --host 127.0.0.1` + `--enable-auto-tool-choice --tool-call-parser hermes` | Supervisor argv. Default bind is loopback. Remote bind is an explicit `--host` / config override |
| Isolated venv under `~/.local/share/enodios/.venv` | Isolated venv under `get_hermes_home()`-rooted `runtimes/vllm/` (see Paths) |
| Readiness `GET /v1/models` | Supervisor ready-check. Optional touch / tool-call bench after first start |
| Tool-call bench (`calculator` tool, `tool_choice=required`) | Shared helper; CLI `hermes local bench` and Desktop “Verify tools” |
| `stop` / `pause` to free GPU | Supervisor stop. Desktop server toggle + CLI `stop`. Switching engines stops the other managed server |
| `configure` wiring `http://127.0.0.1:8000/v1` | Activate via existing `provider: vllm` → custom + that base URL. **No regex.** |

Enodios defaults worth keeping as starting points (all overridable in
`config.yaml`, none as `HERMES_*`):

- Model: `solidrust/Hermes-3-Llama-3.1-8B-AWQ` served as `hermes3:8b`
- Port: `8000` (vLLM’s own default; llama.cpp already avoids `8080` /
  `8000` by using `18434`)
- Context: `65536` (Hermes agent tool-loop floor)
- KV cache: `fp8` when the GPU class allows
- GPU util: tiered (tighter on 12 GB, more headroom on 24 GB shared
  desktops)
- Quant: `awq` for the default 8B; BF16 alt only on ≥40 GB clean cards

## Architecture

### Package split

```
hermes_cli/local_runtime/     # UNCHANGED charter: llama.cpp only
hermes_cli/vllm_runtime/      # NEW sibling
    __init__.py               # public: ensure / shutdown / recommend / activate
    venv.py                   # create / upgrade isolated venv + pip vllm
    supervisor.py             # spawn vllm serve, pid/log/state, stop/pause
    recommend.py              # nvidia-smi → VRAM tier → model + flags
    endpoint.py               # resolve managed or already-running vLLM
    bootstrap.py              # ensure_vllm_runtime(config) — CLI + Desktop
    bench.py                  # /v1/models + tool-call smoke
```

`local_runtime/__init__.py` today says “Managed llama.cpp runtime.” Keep
that sentence true. Cross-engine policy lives in a thin coordinator, not
inside either package:

- Config key: `local_runtime.engine: llamacpp | vllm` (default
  `llamacpp`). New keys deep-merge; bump `_config_version` only if a
  migration is required (it should not be — `engine` is additive).
- Switching engines **stops the other managed server** before starting
  the new one. One GPU, one resident weights file.
- CLI and Desktop call the **same** helpers (`ensure_vllm_runtime`,
  `shutdown_vllm_runtime`, `recommend_vllm`, `activate_vllm_provider`).
  Desktop is not a special case; the Local Models page is another
  caller.

Reuse what is already generic:

- `hermes_cli/local_runtime/hardware.py` (`nvidia-smi` PATH ladder,
  `probe_budget`) — import from the vLLM sibling; do not fork a second
  GPU probe.
- `save_config_value()` / `DEFAULT_CONFIG` — all behavioral knobs.
- Provider alias `vllm` → `custom` (`hermes_cli/auth.py`,
  `hermes_cli/providers.py`). Managed activate writes
  `model.provider: vllm`, `model.default: <served name>`,
  `model.base_url: http://127.0.0.1:8000/v1` (or the configured port),
  plus a `custom_providers` entry if one is not already present.

Do **not** invent a parallel `provider: vllm-managed`. Reachability is
the credential, same as llamacpp.

### Paths

llama.cpp binaries are **machine-scoped** on purpose
(`get_default_hermes_root() / "runtimes" / "llamacpp"`): a second
profile must not re-download the engine or fight over the port.
vLLM is the same class of asset (multi-GB PyTorch wheel + HF weights).

- Venv + wheel cache: `<default hermes root>/runtimes/vllm/`
  (`get_hermes_home()` after profile override still points here when
  the default profile is active; implementation should follow the
  llama.cpp `get_default_hermes_root()` rule so profiles share one
  venv).
- HF weights: Hugging Face cache (default `~/.cache/huggingface`), not
  a second copy under the profile. Optional later: pin
  `HF_HOME` under the same runtimes tree if we want Hermes-owned
  eviction.
- Supervisor state: `<default hermes root>/runtimes/vllm/server.json`
  (pid, port, bind, served name, started_at). Logs beside it.
- Profile-scoped: `local_runtime.engine`, `local_runtime.enabled`,
  `model.provider` / `model.default` / `model.base_url`. A profile
  chooses the engine; the machine holds the venv.

Never hardcode `~/.hermes`.

### Config shape (additive)

```yaml
local_runtime:
  enabled: false
  engine: llamacpp          # llamacpp | vllm
  # existing llama.cpp keys (tag, backend, models_max, port, detect_ports)
  vllm:
    port: 8000
    host: 127.0.0.1         # 0.0.0.0 only when the user opts into remote
    model: solidrust/Hermes-3-Llama-3.1-8B-AWQ
    served_model_name: hermes3:8b
    max_model_len: 65536
    gpu_memory_utilization: 0.75
    quantization: awq
    kv_cache_dtype: fp8
    tool_call_parser: hermes
    python: "3.12"          # venv interpreter pin; see Risks
```

No `HERMES_VLLM_*`. If vLLM’s process needs `CUDA_VISIBLE_DEVICES`,
read it from this block (or leave unset).

### Supervisor contract

Spawn (localhost default):

```text
vllm serve <model> \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.75 \
  --quantization awq \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --served-model-name hermes3:8b
```

`--host` is the only switch that turns this into a remote/LAN server.
Managed 1-click never binds `0.0.0.0` unless the user set
`local_runtime.vllm.host`.

Readiness:

1. Process still alive.
2. `GET http://127.0.0.1:<port>/v1/models` succeeds (enodios
   `vllm_ready`).
3. After first install / model change: tool-call bench (required
   `calculator` tool). Fail the activate job if the parser is not
   actually producing `tool_calls`.

Stop / pause: SIGTERM the supervised pid, wait, then SIGKILL. Goal is
VRAM back for games / other GPU work. Hermes config is unchanged; the
next start reuses the same provider URL.

Idle unload: llama.cpp unloads after 15 minutes. vLLM sleep/pause is
optional follow-up; v1 can be explicit stop only. Do not pretend
llama.cpp’s router idle-unload maps 1:1 onto a single-model vLLM
process.

Restart backoff: copy the llama.cpp supervisor idea (bounded backoff,
persisted port + identity) without importing its GGUF/router internals.

Always dial `127.0.0.1` for local health checks. Do not resolve
`localhost` (IPv6 fallback tax, learned on the llama.cpp supervisor).

### Provider activation

Existing chain already treats `vllm` as a local OpenAI-compatible
alias (`auth.py`: `"vllm": "custom"`; `providers.py` `local` group).
Activate means:

1. Ensure venv + weights + supervised server.
2. `save_config_value("model.provider", "vllm")`
3. `save_config_value("model.default", served_name)`
4. `save_config_value("model.base_url", "http://127.0.0.1:8000/v1")`
5. Upsert `custom_providers` via the config API (not a regex).

A user who already pointed `provider: vllm` at a remote host keeps that
URL. Managed start only overwrites `base_url` when the configured host
is loopback / empty / the managed port. Remote vLLM is still “just
`--host`” on the GPU box; Hermes on the controller stays a custom
endpoint.

### Local Models page (one page, two engines)

Same Settings → Providers → Local Models surface. Add an **engine
switch** (llama.cpp | vLLM). Default remains llama.cpp.

| Engine | User sees |
|---|---|
| llama.cpp | Today’s GGUF catalog, hardware fit, quickstart, HF GGUF browse, sideload `.gguf`, eject |
| vLLM | HF model id (recommended from VRAM tier), isolated-venv install progress, start / stop / pause, tool-call verify, “Use” activate |

Do **not** overload llama-only routes without engine guards:

- `GET /api/local-models/catalog`
- `POST /api/local-models/quickstart`
- `POST /api/local-models/sideload`
- `POST /api/local-models/eject`

Those are GGUF / llama-server operations. Unguarded calls while
`engine: vllm` would download a GGUF into the llama router or eject
the wrong tree. Guard with `local_runtime.engine` (400 + plain-language
reason) or split vLLM routes (`/api/local-models/vllm/...`) and keep
the llama routes llama-only.

Shared routes that stay engine-aware:

- `GET /api/local-models/status` — report which engine is active,
  whether *that* server is up, VRAM, last error.
- `GET /api/local-models/hardware` — already generic enough; vLLM
  recommend consumes the same probe.
- `POST /api/local-models/server` — start/stop must dispatch to the
  active engine’s supervisor and stop the other.

Desktop (`apps/desktop/src/app/settings/local-models-settings.tsx`)
calls the same HTTP API the dashboard already uses. No Electron-only
venv installer. No `HERMES_DESKTOP` gate — engine availability is
session/config, not “was I spawned by Electron.”

### CLI

One helper, two skins:

```text
hermes local status
hermes local engine llamacpp|vllm
hermes local recommend          # print tier + model; --apply writes config.yaml
hermes local install            # venv + vllm wheel (vLLM) or zip (llama)
hermes local start | stop
hermes local bench
hermes local use                # activate provider (same helper as Desktop Use)
```

Argparse alias dispatch: accept `ls` if we add it. Interactive pickers
use curses, not a new prompt_toolkit wizard.

`hermes setup` / `hermes model` keep “Custom endpoint” for people who
already run vLLM themselves. Managed 1-click is `hermes local` + the
Local Models page.

## Phases

### Phase 1 — config + supervisor

- Add `local_runtime.engine` and `local_runtime.vllm.*` to
  `DEFAULT_CONFIG` (deep-merge; no version bump unless a rename is
  required).
- New `hermes_cli/vllm_runtime/` with venv create, `vllm serve`
  supervisor, `GET /v1/models` readiness, stop/pause, recommend
  tiers.
- Coordinator: selecting `engine: vllm` stops llama-server; selecting
  `llamacpp` stops vLLM.
- Unit tests against a temp `HERMES_HOME`: argv construction, bind
  default `127.0.0.1`, recommend table contracts (VRAM → model
  relationship, not frozen snapshot strings), stop-other-engine
  ordering. Mock `nvidia-smi` / subprocess at the helper boundary;
  do not fake `sys.platform`.

Exit: `ensure_vllm_runtime(config)` brings up a supervised localhost
server on a machine that already has the venv; `shutdown` frees the
port. No UI yet.

### Phase 2 — CLI

- `hermes local …` subcommand (new `hermes_cli/subcommands/` module)
  calling Phase 1 helpers only.
- `hermes local use` writes provider via `save_config_value`.
- `hermes local recommend --apply` writes `local_runtime.vllm.*`,
  never a `.env`.
- Doctor row: venv present, `nvidia-smi`, `/v1/models`, last bench.

Exit: a Linux NVIDIA box can `install → recommend --apply → start →
use → chat` without opening Desktop.

### Phase 3 — Desktop / dashboard UI

- Engine switch on the existing Local Models page.
- vLLM panel: recommend summary, install job (POST → poll, same job
  pattern as llama runtime install), start/stop, Use.
- Guard llama-only routes; add vLLM routes or an engine query param
  with explicit 400s.
- Desktop vitest for the switch (does not call llama catalog APIs
  while `engine: vllm`). Python router tests for the guards.

Exit: Desktop 1-click on vLLM matches llama.cpp’s “Install → Use”
shape, without a terminal.

### Phase 4 — fit catalog

- Curated HF ids per VRAM tier (start with enodios’s Hermes 3 8B AWQ
  + documented alts). Relationship tests: every catalog row has a
  min-VRAM and a tool-parser, and recommend never returns a row
  whose min-VRAM exceeds probed total.
- Optional: browse HF (safetensors / AWQ), not GGUF sideload.
- Document in `website/docs/user-guide/local-models.md`: llama.cpp
  default, vLLM alternative, Linux CUDA motivation, stop-to-play-
  games.

Exit: recommend is data, not a hardcoded single model, and the user
guide states both engines.

## Test plan

Always `scripts/run_tests.sh` (never bare pytest). Isolation already
redirects `HERMES_HOME`.

- **Invariant tests (1–2 per phase, proven red on base):**
  - Default engine is `llamacpp`; missing `engine` key deep-merges to
    that default.
  - vLLM supervisor argv includes `--host 127.0.0.1`,
    `--enable-auto-tool-choice`, `--tool-call-parser hermes`.
  - `activate` writes `model.provider == "vllm"` and a loopback
    `base_url` ending in `/v1` — relationship, not a frozen port
    snapshot if port is configurable.
  - Switching `engine` to `vllm` calls llama `shutdown` before vLLM
    start (and the reverse).
  - Llama-only routes (`/catalog`, `/quickstart`, `/sideload`,
    `/eject`) reject `engine: vllm` without touching GGUF state.
- **E2E against a temp `HERMES_HOME`:** real imports, real config
  write, supervisor given a fake `vllm` executable (script that
  serves `/v1/models`). No live GPU required in CI.
- **Linux-only marker** for tests that must see `nvidia-smi` or a
  real CUDA wheel. Do not `patch sys.platform`. Host-independent
  recommend math takes VRAM numbers as data.
- **Do not** read source files in tests. Do not snapshot
  `_config_version` or catalog length.
- Desktop: vitest on the engine switch and “llama routes not called
  while vLLM is selected.” A Python test that greps
  `package.json` / `.tsx` will not run on a JS-only PR — keep UI
  contracts in vitest.

## Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| **PyTorch / vLLM wheel size** | Several GB; first install is not “a few hundred MB” like llama.cpp zips | Honest progress UI; isolated venv so Hermes’ own interpreter is untouched; machine-scoped cache so profiles do not duplicate |
| **Python pin** | Enodios defaults to 3.14; vLLM wheels lag. Hermes itself may run on a different minor | Isolated venv with an explicit `local_runtime.vllm.python`; recreate venv on pin change; fail install with “need CPython X.Y” rather than pip-on-the-Hermes-venv |
| **HF weights vs GGUF** | AWQ / safetensors are not the llama catalog. Sideload `.gguf` is meaningless on vLLM | Separate catalog; no shared quickstart; do not reuse `/sideload` |
| **VRAM vs 64k floor** | Hermes tool loops want 64k. 8–12 GB cards often cannot hold 8B AWQ + 64k | Recommend `feasible: false` with a cloud fallback message (enodios does this). Do not silently drop to 8k and call it 1-click |
| **Two servers on one GPU** | Starting vLLM while llama-server is resident OOMs | Engine switch always stops the other supervisor first |
| **Route mix-ups** | Catalog/quickstart/sideload/eject are llama-shaped | Engine guards or split routes; tests in Phase 3 |
| **Remote vs managed URL clobber** | A user with `provider: vllm` at a GPU host must not be rewritten to 127.0.0.1 | Activate overwrites loopback/empty only |
| **Docker temptation** | Images look 1-click; enodios warns `CUDA_ERROR_UNKNOWN` | Out of scope for v1; native venv only |

## Out of scope (recap)

- Replacing llama.cpp as the default engine
- Docker-as-default (or any Docker path in v1)
- Regex-editing `config.yaml`
- Bundling enodios as a submodule or calling its CLI
- New `HERMES_*` behavioral environment variables
- Teaching `hermes_cli/local_runtime/` to spawn Python
- Mid-conversation engine swaps that rebuild the system prompt
  (engine change is a next-session / explicit restart, like any
  provider change)

## Success

A Linux NVIDIA user opens Desktop → Local Models → switches the engine
to vLLM → clicks Install → clicks Use, or runs the same sequence via
`hermes local`, and chats against a supervised `127.0.0.1` vLLM with
Hermes tool-calling. llama.cpp remains one click for everyone else.
Remote vLLM is still the same engine with a different `--host`.
)
