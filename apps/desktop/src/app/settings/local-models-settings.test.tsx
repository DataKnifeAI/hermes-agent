import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $localRuntimeJobs } from '@/store/local-runtime-jobs'
import type { LocalCatalogModel, LocalHardware, LocalModelsStatus, LocalRuntimeJob } from '@/types/hermes'

import { LocalModelsSettings } from './local-models-settings'

// Mock the API layer — the pane's contract is what it RENDERS from these
// payloads, not transport.
vi.mock('@/hermes', () => ({
  activateLocalModel: vi.fn(),
  checkVllmUpdate: vi.fn(),
  deleteLocalModel: vi.fn(),
  deleteVllmModel: vi.fn(),
  downloadBrowsedModel: vi.fn(),
  downloadVllmModel: vi.fn(),
  downloadLocalModel: vi.fn(),
  ejectLocalModel: vi.fn(),
  getLocalCatalog: vi.fn(),
  getLocalHardware: vi.fn(),
  getLocalModelsJobs: vi.fn(),
  getLocalModelsStatus: vi.fn(),
  getLocalRuntimeJob: vi.fn(),
  getVllmModels: vi.fn(),
  getVllmRecommend: vi.fn(),
  installLocalRuntime: vi.fn(),
  installVllm: vi.fn(),
  listHFRepoFiles: vi.fn(),
  quickstartLocalModels: vi.fn(),
  searchHFModels: vi.fn(),
  searchVllmModels: vi.fn(),
  setLocalEngine: vi.fn(),
  setLocalServer: vi.fn(),
  setVllmModel: vi.fn(),
  sideloadLocalModel: vi.fn(),
  updateVllm: vi.fn(),
  useVllm: vi.fn()
}))

import * as hermes from '@/hermes'

const mocked = vi.mocked(hermes)

const BASE_STATUS: LocalModelsStatus = {
  enabled: true,
  engine: 'llamacpp',
  tag: 'b10290',
  configured_tag: 'b10290',
  update_available: false,
  runtime_installed: false,
  runtime_backend: null,
  server_running: false,
  server_base_url: null,
  active_model_id: null,
  loaded_models: {},
  models: [],
  models_dir: 'C:/somewhere/models'
}

const VLLM_STATUS: LocalModelsStatus = {
  ...BASE_STATUS,
  engine: 'vllm',
  tag: '',
  configured_tag: '',
  runtime_installed: false,
  venv_ready: false,
  occupancy: [],
  occupancy_message: null,
  served_model_name: 'hermes3:8b',
  model: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ'
}

const BASE_HARDWARE: LocalHardware = {
  uma: false,
  vram_total_bytes: 32 * 2 ** 30,
  vram_usable_bytes: 26 * 2 ** 30,
  ram_total_bytes: 256 * 2 ** 30,
  ram_available_bytes: 200 * 2 ** 30,
  vram_label: '32.0 GB',
  gpu_name: 'NVIDIA GeForce RTX 5090',
  gpu_util_percent: 12,
  vram_used_bytes: 6 * 2 ** 30,
  vram_free_bytes: 26 * 2 ** 30,
  vram_engine_bytes: 5 * 2 ** 30,
  vram_other_bytes: 1 * 2 ** 30,
  gpu_driver_version: '560.35.03',
  cuda_compute_capability: '8.9',
  engine: 'llamacpp',
  models_dir: '/tmp/hermes-home/models',
  models_dir_display: '~/.hermes/models',
  models_storage_bytes: 17.6 * 2 ** 30,
  disk_free_bytes: 800 * 2 ** 30,
  disk_total_bytes: 2000 * 2 ** 30,
  runtime_dir: '/tmp/hermes-home/runtimes/llamacpp',
  runtime_dir_display: '~/.hermes/runtimes/llamacpp',
  occupancy_foreign: false,
  ctx_64k_feasible: true
}

const FITTING_MODEL: LocalCatalogModel = {
  id: 'Qwen3.6-27B-UD-Q4_K_XL',
  display_name: 'Qwen3.6 27B',
  description: 'Best all-round agent model; long context stays fast',
  size_bytes: 17.6 * 2 ** 30,
  size_label: '17.6 GB',
  native_context: 262144,
  native_context_label: '256K',
  recommended: true,
  downloaded: false,
  mtp: false,
  fits: true,
  fit_summary: 'runs at its full 256K context',
  start_window: 262144,
  start_window_label: '256K',
  spilled: false
}

const SPILLED_MODEL: LocalCatalogModel = {
  ...FITTING_MODEL,
  id: 'Spilled-Model',
  display_name: 'Spilled Model',
  recommended: false,
  fits: true,
  spilled: true,
  start_window: 65536,
  start_window_label: '64K',
  fit_summary: 'starts at 64K and grows toward 256K as you use it (larger than your GPU memory — runs slower)'
}

const REFUSED_MODEL: LocalCatalogModel = {
  ...FITTING_MODEL,
  id: 'Huge-Model',
  display_name: 'Huge Model',
  recommended: false,
  fits: false,
  fit_summary: 'Needs more memory than this machine has',
  fit_detail: 'needs ~60 GiB at the 64K floor',
  start_window: undefined,
  start_window_label: undefined
}

function renderPane() {
  return render(
    <MemoryRouter>
      <I18nProvider>
        <LocalModelsSettings />
      </I18nProvider>
    </MemoryRouter>
  )
}

// The fresh-machine states these tests exercise now lead with the
// quickstart card; the full pane (runtime rows, model list, browser)
// is one 'Configure…' click away. Render and click through.
async function renderFullPane() {
  const result = renderPane()
  const configure = await screen.findByRole('button', { name: /configure/i })

  fireEvent.click(configure)

  return result
}

beforeEach(() => {
  mocked.getLocalModelsStatus.mockResolvedValue(BASE_STATUS)
  mocked.getLocalHardware.mockResolvedValue(BASE_HARDWARE)
  mocked.getLocalCatalog.mockResolvedValue({ models: [FITTING_MODEL, SPILLED_MODEL, REFUSED_MODEL] })
  mocked.getLocalModelsJobs.mockResolvedValue({ jobs: [] })
  mocked.getVllmRecommend.mockResolvedValue({
    config: {},
    feasible: true,
    gpu_memory_utilization: 0.75,
    kv_cache_dtype: 'fp8',
    max_model_len: 65536,
    model: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
    quantization: 'awq',
    reason: 'ok',
    served_model_name: 'hermes3:8b',
    tier: '24gb'
  })
  mocked.setLocalEngine.mockResolvedValue({ engine: 'vllm', ok: true })
  mocked.installVllm.mockResolvedValue({ job_id: 'v1' })
  mocked.useVllm.mockResolvedValue({ base_url: 'http://127.0.0.1:18435/v1', ok: true })
  mocked.getVllmModels.mockResolvedValue({
    models: [
      {
        active: true,
        added_by_you: false,
        cached: false,
        capabilities: ['awq', 'instruct', 'tools'],
        display_name: 'hermes3:8b',
        fit: 'fits-gpu',
        fits: true,
        id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
        recommended: true,
        served_model_name: 'hermes3:8b',
        size_bytes: 0,
        size_label: '—'
      }
    ]
  })
  mocked.downloadVllmModel.mockResolvedValue({ already_downloaded: false, job_id: 'vd1', model: 'x' })
  mocked.searchVllmModels.mockResolvedValue({ hits: [] })
  mocked.setVllmModel.mockResolvedValue({
    model: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
    ok: true,
    served_model_name: 'hermes3:8b'
  })
  mocked.deleteVllmModel.mockResolvedValue({ ok: true })
  mocked.checkVllmUpdate.mockResolvedValue({
    configured_tag: '0.10.0',
    installed: '0.10.0',
    latest: '0.10.0',
    tag: '0.10.0',
    update_available: false
  })
  mocked.updateVllm.mockResolvedValue({ job_id: 'vu1' })
  $localRuntimeJobs.set([])
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('LocalModelsSettings', () => {
  it('offers the runtime install with a plain-language explanation', async () => {
    await renderFullPane()

    expect(await screen.findByText('Install the local runtime')).toBeTruthy()
    expect(screen.getByText(/runs? entirely on this machine/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: /install runtime/i })).toBeTruthy()
  })

  it('shows every catalog model with fit pills; unaffordable ones stay visible with the reason', async () => {
    await renderFullPane()

    expect(await screen.findByText('Qwen3.6 27B')).toBeTruthy()
    // The fitting model reads as pills, not prose: green memory pill +
    // green full-context pill (start_window == native, resident on GPU).
    expect(screen.getByText('Fits your GPU')).toBeTruthy()
    expect(screen.getByText('Full 256K context').className).toContain('emerald')

    // The refused model is NOT hidden (discoverability rule): red memory
    // pill, plus the ceiling it would have had.
    expect(screen.getByText('Huge Model')).toBeTruthy()
    expect(screen.getByText('Too big for this machine')).toBeTruthy()

    // The spilled model reads amber + ONE quiet ceiling pill — the same
    // 'Up to' shape the refused row wears; no start/grow pair.
    expect(screen.getByText('Spilled Model')).toBeTruthy()
    expect(screen.getByText('Uses system RAM')).toBeTruthy()
    expect(screen.getAllByText('Up to 256K context').length).toBe(2)
    expect(screen.queryByText(/Starts at/)).toBeNull()

    // Its download button is disabled; the fitting model's is enabled once
    // the runtime exists (here runtime_installed=false, so both disabled —
    // asserted separately below).
    const buttons = screen.getAllByRole('button', { name: /download · 17\.6 GB/i })
    expect(buttons.every(b => (b as HTMLButtonElement).disabled)).toBe(true)
  })

  it('orders the catalog by fit: resident first, then spilled, then too-big', async () => {
    // Scrambled input — the pane, not the backend, owns display order.
    mocked.getLocalCatalog.mockResolvedValue({ models: [REFUSED_MODEL, SPILLED_MODEL, FITTING_MODEL] })
    await renderFullPane()
    await screen.findByText('Qwen3.6 27B')

    // The matched element is the row-title span; the recommended row's
    // includes its nested pill copy — strip it before comparing order.
    const names = screen
      .getAllByText(/^(Qwen3\.6 27B|Spilled Model|Huge Model)$/)
      .map(el => el.textContent?.replace('Recommended', ''))

    expect(names).toEqual(['Qwen3.6 27B', 'Spilled Model', 'Huge Model'])
  })

  it('never greens the full-context pill on a system-RAM model', async () => {
    // Full native window, but earned by spilling into system RAM: the
    // pill must not wear the green that would recommend exactly the
    // wrong model.
    const spilledFull: LocalCatalogModel = {
      ...FITTING_MODEL,
      id: 'Spilled-Full',
      display_name: 'Spilled Full',
      recommended: false,
      spilled: true,
      fit_summary: 'runs its full 256K context, partly from system RAM'
    }

    mocked.getLocalCatalog.mockResolvedValue({ models: [spilledFull] })
    await renderFullPane()
    await screen.findByText('Spilled Full')

    expect(screen.getByText('Full 256K context').className).not.toContain('emerald')
  })

  it('explains the Recommended pick on hover', async () => {
    // The tooltip is the resolver's own reason, and it must actually OPEN:
    // Tip works by asChild-cloning hover handlers onto the pill, so a Pill
    // that swallows its rest props kills the tooltip silently (the pill
    // still renders, nothing appears on hover).
    mocked.getLocalCatalog.mockResolvedValue({
      models: [{ ...FITTING_MODEL, recommended_reason: 'speed-gated-quality' }]
    })
    await renderFullPane()
    await screen.findByText('Qwen3.6 27B')

    fireEvent.pointerMove(screen.getByText('Recommended'))
    fireEvent.pointerEnter(screen.getByText('Recommended'))

    await waitFor(() =>
      expect(screen.getAllByText(/would respond too slowly on its memory bandwidth/).length).toBeGreaterThan(0)
    )
  })

  it('enables downloads only once the runtime is installed', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...BASE_STATUS,
      runtime_installed: true,
      runtime_backend: 'cuda'
    })
    await renderFullPane()

    await screen.findByText('Qwen3.6 27B')
    const [fittingButton] = screen.getAllByRole('button', { name: /download · 17\.6 GB/i })
    expect((fittingButton as HTMLButtonElement).disabled).toBe(false)
  })

  it('shows hardware facts after backfill', async () => {
    await renderFullPane()

    expect(await screen.findByText(/NVIDIA GeForce RTX 5090/)).toBeTruthy()
    expect(screen.getByText(/6\.0 GB \/ 32\.0 GB used/)).toBeTruthy()
    expect(screen.getByText(/26\.0 GB free/)).toBeTruthy()
    expect(screen.getByText(/56\.0 GB \/ 256\.0 GB used/)).toBeTruthy()
    expect(screen.queryByText(/this engine 5\.0 GB/)).toBeNull()
    expect(screen.queryByText(/~\/\.hermes\/models/)).toBeNull()
    expect(screen.queryByText(/64k context fits this GPU/)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /^show more$/i }))
    expect(screen.getByText(/this engine 5\.0 GB/)).toBeTruthy()
    expect(screen.getByText(/~\/\.hermes\/models/)).toBeTruthy()
    expect(screen.getByText(/17\.6 GB on disk/)).toBeTruthy()
    expect(screen.getByText(/64k context fits this GPU/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /^show less$/i }))
    expect(screen.queryByText(/64k context fits this GPU/)).toBeNull()
  })

  it('tracks a download job to completion and refreshes', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...BASE_STATUS,
      runtime_installed: true,
      runtime_backend: 'cuda'
    })
    mocked.downloadLocalModel.mockResolvedValue({ job_id: 'j1' })

    const running: LocalRuntimeJob = {
      job_id: 'j1',
      kind: 'model-download',
      target: 'Qwen3.6 27B',
      model_id: FITTING_MODEL.id,
      status: 'running',
      phase: 'downloading',
      detail: 'Qwen3.6 27B — 17.6 GB',
      total_bytes: 100,
      done_bytes: 40,
      percent: 40,
      error: null
    }

    mocked.getLocalModelsJobs
      .mockResolvedValueOnce({ jobs: [running] })
      .mockResolvedValue({ jobs: [{ ...running, status: 'done', phase: 'done', done_bytes: 100, percent: 100 }] })

    await renderFullPane()
    await screen.findByText('Qwen3.6 27B')

    const [download] = screen.getAllByRole('button', { name: /download · 17\.6 GB/i })
    download.click()

    // The app-level watcher follows the job; when it settles the pane
    // refreshes (status + catalog re-fetched).
    await waitFor(() => {
      expect(mocked.getLocalModelsJobs).toHaveBeenCalled()
      expect(mocked.getLocalModelsStatus.mock.calls.length).toBeGreaterThanOrEqual(2)
    })
  })

  it('renders progress for a download discovered from the store (survives pane remount)', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...BASE_STATUS,
      runtime_installed: true,
      runtime_backend: 'cuda'
    })
    // A running job already in the app-level store — as after closing and
    // reopening the pane mid-download.
    $localRuntimeJobs.set([
      {
        job_id: 'j9',
        kind: 'model-download',
        target: 'Qwen3.6 27B',
        model_id: FITTING_MODEL.id,
        status: 'running',
        phase: 'downloading',
        detail: '',
        total_bytes: 100,
        done_bytes: 62,
        percent: 62,
        error: null
      }
    ])

    await renderFullPane()
    await screen.findByText('Qwen3.6 27B')

    // The fitting row shows byte progress; the remaining download
    // buttons belong to the other rows (spilled + refused).
    expect(screen.getAllByText(/0\.0 GB of 0\.0 GB|of/).length).toBeGreaterThan(0)
    const remaining = screen.queryAllByRole('button', { name: /download · 17\.6 GB/i })
    expect(remaining.length).toBe(2)
    expect(remaining.some(b => (b as HTMLButtonElement).disabled)).toBe(true)
  })

  it('surfaces a failed download with the backend message', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...BASE_STATUS,
      runtime_installed: true,
      runtime_backend: 'cuda'
    })
    $localRuntimeJobs.set([
      {
        job_id: 'j2',
        kind: 'model-download',
        target: 'Qwen3.6 27B',
        model_id: FITTING_MODEL.id,
        status: 'error',
        phase: 'verifying',
        detail: '',
        total_bytes: 100,
        done_bytes: 100,
        error: 'Downloaded file failed its integrity check and was removed — try again'
      }
    ])

    await renderFullPane()
    await screen.findByText('Qwen3.6 27B')

    expect(await screen.findByText(/integrity check/)).toBeTruthy()
  })
})

describe('quickstart', () => {
  it('leads with one button on a fresh machine and fires the quickstart job', async () => {
    mocked.quickstartLocalModels.mockResolvedValue({
      display_name: 'Qwen3.6 27B',
      download_bytes: FITTING_MODEL.size_bytes,
      job_id: 'q1',
      model_id: 'qwen3.6-27b',
      needs_download: true,
      needs_runtime: true
    })
    renderPane()

    // The card names the recommended model and the one-click action; the
    // runtime/model machinery is NOT on screen.
    expect(await screen.findByRole('button', { name: /set up for me/i })).toBeTruthy()
    expect(screen.queryByText('Install the local runtime')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /set up for me/i }))
    await waitFor(() => {
      expect(mocked.quickstartLocalModels).toHaveBeenCalled()
    })
  })

  it('pins the quickstart progress view while the job runs', async () => {
    $localRuntimeJobs.set([
      {
        job_id: 'q1',
        kind: 'quickstart',
        target: 'Qwen3.6 27B',
        model_id: 'qwen3.6-27b',
        status: 'running',
        phase: 'downloading',
        detail: 'Qwen3.6 27B — 17.6 GB',
        total_bytes: 100,
        done_bytes: 30,
        percent: 30,
        error: null
      }
    ])
    renderPane()

    expect(await screen.findByText('Qwen3.6 27B — 17.6 GB')).toBeTruthy()
    // One job, one view: no Set up / Configure buttons while it runs.
    expect(screen.queryByRole('button', { name: /set up for me/i })).toBeNull()
  })

  it('skips the card entirely once a model is staged', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...BASE_STATUS,
      runtime_installed: true,
      runtime_backend: 'cuda',
      models: [{ id: 'Qwen3.6-27B-UD-Q4_K_XL', size_bytes: 17 * 2 ** 30, size_label: '17.6 GB' }]
    })
    renderPane()

    // Straight to the full pane — no quickstart hero for a working setup.
    expect(await screen.findByText('Qwen3.6 27B')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /set up for me/i })).toBeNull()
  })
})

describe('BrowseSection', () => {
  it('searches HF after a pause and shows fit-priced files on demand', async () => {
    vi.useFakeTimers()

    try {
      vi.mocked(hermes.searchHFModels).mockResolvedValue({
        hits: [{ downloads: 872724, gated: false, likes: 47, repo: 'unsloth/Qwen3.8-27B-GGUF', updated: '2026-08-18' }]
      })
      vi.mocked(hermes.listHFRepoFiles).mockResolvedValue({
        files: [
          { fit: 'fits-gpu', label: 'Q4_K_M', paths: ['Qwen3.8-27B-Q4_K_M.gguf'], total_bytes: 17 * 2 ** 30 },
          { fit: 'too-big', label: 'F16', paths: ['Qwen3.8-27B-F16.gguf'], total_bytes: 56 * 2 ** 30 }
        ]
      })

      render(
        <MemoryRouter>
          <I18nProvider>
            <LocalModelsSettings />
          </I18nProvider>
        </MemoryRouter>
      )
      await act(async () => {
        await vi.runOnlyPendingTimersAsync()
      })
      // Fresh machine leads with the quickstart card — enter the full pane.
      fireEvent.click(screen.getByRole('button', { name: /configure/i }))

      const box = screen.getByPlaceholderText(/search models/i)
      fireEvent.change(box, { target: { value: 'qwen' } })
      // Debounce: no call until the pause elapses.
      expect(hermes.searchHFModels).not.toHaveBeenCalled()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(400)
      })
      expect(hermes.searchHFModels).toHaveBeenCalledWith('qwen')
      expect(screen.getByText('unsloth/Qwen3.8-27B-GGUF')).toBeTruthy()

      fireEvent.click(screen.getByRole('button', { name: /show files/i }))
      await act(async () => {
        await vi.runOnlyPendingTimersAsync()
      })
      expect(screen.getByText('Q4_K_M')).toBeTruthy()
      // Each tile has an explicit download button; the too-big quant's is
      // disabled, the fitting one is live and starts the download.
      const q4Btn = screen.getByRole('button', { name: 'Download Q4_K_M' })
      const f16Btn = screen.getByRole('button', { name: 'Download F16' })
      expect((f16Btn as HTMLButtonElement).disabled).toBe(true)
      expect((q4Btn as HTMLButtonElement).disabled).toBe(false)

      vi.mocked(hermes.downloadBrowsedModel).mockResolvedValue({ job_id: 'j1', model_id: 'Qwen3.8-27B-Q4_K_M' })
      fireEvent.click(q4Btn)
      await act(async () => {
        await vi.runOnlyPendingTimersAsync()
      })
      expect(hermes.downloadBrowsedModel).toHaveBeenCalledWith('unsloth/Qwen3.8-27B-GGUF', ['Qwen3.8-27B-Q4_K_M.gguf'])
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('added-by-you rows', () => {
  it('staged models outside the catalog get the full action set', async () => {
    vi.mocked(hermes.getLocalModelsStatus).mockResolvedValue({
      ...BASE_STATUS,
      loaded_models: { 'Hermes-4.3-36B-Q5_K_M': 'loaded' },
      models: [{ id: 'Hermes-4.3-36B-Q5_K_M', size_bytes: 25 * 2 ** 30, size_label: '25.0 GB' }],
      placement: {
        'Hermes-4.3-36B-Q5_K_M': {
          granted_window_label: '96K',
          spilled: false,
          window: 98304,
          window_label: '96K'
        }
      },
      server_running: true
    })
    vi.mocked(hermes.getLocalCatalog).mockResolvedValue({ models: [] })

    renderPane()
    await screen.findByText('Hermes-4.3-36B-Q5_K_M')

    // Full management surface: Use, eject, delete, live placement pill.
    expect(screen.getByText(/added by you/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: /use/i })).toBeTruthy()
    expect(screen.getByText(/96K/)).toBeTruthy()
    const buttons = screen.getAllByRole('button')
    expect(buttons.length).toBeGreaterThanOrEqual(3)
  })
})

describe('quickstart completion navigation', () => {
  it('lands on a new chat when a quickstart it watched finishes; stale done jobs on mount never navigate', async () => {
    const routeProbe = vi.fn()

    function Probe() {
      const loc = useLocation()
      routeProbe(loc.pathname)

      return null
    }

    const doneJob: LocalRuntimeJob = {
      done_bytes: 0,
      detail: '',
      error: null,
      job_id: 'stale-done',
      kind: 'quickstart',
      model_id: 'qwen3.8-27b',
      phase: 'done',
      status: 'done',
      target: 'Qwen3.8 27B',
      total_bytes: null
    }

    // A finished quickstart already in history when the pane mounts —
    // must NOT trigger navigation.
    $localRuntimeJobs.set([doneJob])

    render(
      <MemoryRouter initialEntries={['/settings']}>
        <I18nProvider>
          <LocalModelsSettings />
        </I18nProvider>
        <Probe />
      </MemoryRouter>
    )
    await act(async () => {})
    expect(routeProbe).not.toHaveBeenCalledWith('/')

    // A quickstart the pane SAW running that then completes -> navigate.
    const running: LocalRuntimeJob = { ...doneJob, job_id: 'live-run', phase: 'downloading', status: 'running' }
    await act(async () => {
      $localRuntimeJobs.set([doneJob, running])
    })
    await act(async () => {
      $localRuntimeJobs.set([doneJob, { ...running, phase: 'done', status: 'done' }])
    })
    expect(routeProbe).toHaveBeenCalledWith('/')
  })
})

describe('vLLM engine', () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn()
  })

  it('does not call llama-only APIs while vLLM is selected', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue(VLLM_STATUS)
    renderPane()

    expect(await screen.findByLabelText(/local backend/i)).toBeTruthy()
    await waitFor(() => {
      expect(mocked.getLocalModelsStatus).toHaveBeenCalled()
    })
    expect(mocked.getLocalCatalog).not.toHaveBeenCalled()
    expect(mocked.quickstartLocalModels).not.toHaveBeenCalled()
    expect(mocked.sideloadLocalModel).not.toHaveBeenCalled()
    expect(mocked.ejectLocalModel).not.toHaveBeenCalled()
    expect(screen.queryByText('Qwen3.6 27B')).toBeNull()
  })

  it('shows a running vLLM pane with Local inventory, not GGUF widgets', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      runtime_installed: true,
      venv_ready: true,
      server_running: true,
      server_base_url: 'http://127.0.0.1:18435/v1',
      active_model_id: 'hermes3:8b',
      tag: '0.10.0',
      venv_path: '/tmp/runtimes/vllm/.venv',
      models: [{ id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }]
    })
    renderPane()

    expect((await screen.findAllByText(/hermes3:8b/i)).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/127\.0\.0\.1:18435/).length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /turn off/i })).toBeTruthy()
    expect(screen.getByText('Local')).toBeTruthy()
    expect(screen.getAllByText(/0\.10\.0/).length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /check for update/i })).toBeTruthy()
    expect(screen.queryByText(/Download a model, then Use/i)).toBeNull()
    expect(screen.queryByText(/Hermes starts and manages the server for you/i)).toBeNull()
    expect(screen.queryByText('Install the local runtime')).toBeNull()
    expect(screen.queryByText('Qwen3.6 27B')).toBeNull()
    await waitFor(() => {
      expect(mocked.getVllmModels).toHaveBeenCalled()
    })
  })

  it('uses the same select tokens as other settings dropdowns', async () => {
    renderPane()
    const picker = await screen.findByLabelText(/local backend/i)

    expect(picker.getAttribute('data-slot')).toBe('select-trigger')
    expect(picker.getAttribute('role')).toBe('combobox')
    expect(picker.className.split(/\s+/)).toContain('text-xs')
    expect(picker.className.split(/\s+/)).toContain('desktop-input-chrome')
    expect(picker.tagName).not.toBe('SELECT')
  })

  it('writes the engine through setLocalEngine without stopping a server', async () => {
    renderPane()
    const picker = await screen.findByLabelText(/local backend/i)
    fireEvent.click(picker)
    fireEvent.click(await screen.findByRole('option', { name: 'vLLM' }))

    await waitFor(() => {
      expect(mocked.setLocalEngine).toHaveBeenCalledWith('vllm')
    })
    expect(mocked.setLocalServer).not.toHaveBeenCalled()
  })

  it('falls back to first-time setup when the vLLM inventory is empty', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      runtime_installed: true,
      venv_ready: true,
      tag: '0.27.1',
      models: []
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: false,
          added_by_you: false,
          cached: false,
          capabilities: ['awq'],
          display_name: 'qwen3:8b',
          fit: 'fits-gpu',
          fits: true,
          id: 'Qwen/Qwen3-8B-AWQ',
          recommended: true,
          served_model_name: 'qwen3:8b',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        }
      ]
    })
    renderPane()

    expect(await screen.findByRole('button', { name: /set up for me/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /configure/i })).toBeTruthy()
    expect(screen.getByText('Qwen/Qwen3-8B-AWQ')).toBeTruthy()
    const download = screen.getByRole('button', { name: /download · 5\.0 GB/i })
    expect((download as HTMLButtonElement).disabled).toBe(false)
  })

  it('still offers Download on the official row when it is configured but not cached', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      runtime_installed: true,
      venv_ready: true,
      tag: '0.27.1',
      models: []
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: true,
          added_by_you: false,
          cached: false,
          capabilities: ['awq'],
          display_name: 'qwen3:8b',
          fit: 'fits-gpu',
          fits: true,
          id: 'Qwen/Qwen3-8B-AWQ',
          recommended: true,
          served_model_name: 'qwen3:8b',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        }
      ]
    })
    renderPane()

    expect(await screen.findByRole('button', { name: /set up for me/i })).toBeTruthy()
    const download = screen.getByRole('button', { name: /download · 5\.0 GB/i })
    expect((download as HTMLButtonElement).disabled).toBe(false)
    expect(screen.queryByRole('button', { name: /^use$/i })).toBeNull()
  })

  it('Set up for me fires the engine-aware quickstart, not install-only', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue(VLLM_STATUS)
    mocked.quickstartLocalModels.mockResolvedValue({
      display_name: 'hermes3:8b',
      download_bytes: 0,
      job_id: 'vq1',
      model_id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
      needs_download: true,
      needs_runtime: true
    })
    renderPane()

    fireEvent.click(await screen.findByRole('button', { name: /set up for me/i }))
    await waitFor(() => {
      expect(mocked.quickstartLocalModels).toHaveBeenCalled()
    })
    expect(mocked.installVllm).not.toHaveBeenCalled()
  })

  it('pins vLLM quickstart progress while the job runs', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue(VLLM_STATUS)
    $localRuntimeJobs.set([
      {
        job_id: 'vq1',
        kind: 'quickstart',
        target: 'hermes3:8b',
        model_id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
        status: 'running',
        phase: 'downloading',
        detail: 'Downloading solidrust/Hermes-3-Llama-3.1-8B-AWQ',
        total_bytes: 100,
        done_bytes: 40,
        percent: 40,
        error: null
      }
    ])
    renderPane()

    expect(await screen.findByText(/Downloading solidrust/)).toBeTruthy()
    expect(screen.getByText('Model')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /set up for me/i })).toBeNull()
  })

  it('surfaces occupancy copy on the vLLM pane', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      runtime_installed: true,
      venv_ready: true,
      occupancy_message:
        'Another LLM is already running (Ollama on http://127.0.0.1:11434). Stop it so managed vLLM can use the GPU.'
    })
    renderPane()

    expect(await screen.findAllByText(/Another LLM is already running/)).not.toHaveLength(0)
  })

  it('shows fit / capability tags and Download then Use on vLLM catalog rows', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      models: [{ id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: false,
          added_by_you: false,
          cached: false,
          capabilities: ['awq', 'instruct'],
          created_at: '2025-03-15T00:00:00.000Z',
          display_name: 'hermes3:8b',
          fit: 'fits-gpu',
          fits: true,
          id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
          recommended: true,
          served_model_name: 'hermes3:8b',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        },
        {
          active: false,
          added_by_you: false,
          cached: false,
          capabilities: ['awq'],
          display_name: 'qwen3:32b',
          fit: 'too-big',
          fits: false,
          id: 'Qwen/Qwen3-32B-AWQ',
          recommended: false,
          served_model_name: 'qwen3:32b',
          size_bytes: 0,
          size_label: '—'
        },
        {
          active: false,
          added_by_you: true,
          cached: true,
          capabilities: ['fp8'],
          display_name: 'cached-fp8',
          fit: 'unknown',
          fits: null,
          id: 'acme/mystery-fp8',
          recommended: false,
          served_model_name: 'mystery',
          size_bytes: 2 * 2 ** 30,
          size_label: '2.0 GB'
        }
      ]
    })
    renderPane()

    expect(await screen.findByText('Fits your GPU')).toBeTruthy()
    expect(screen.getByText('Released Mar 2025')).toBeTruthy()
    expect(screen.getByText('5.0 GB')).toBeTruthy()
    expect(screen.queryByText('Qwen/Qwen3-32B-AWQ')).toBeNull()
    expect(screen.getByRole('button', { name: /show models that don't fit/i })).toBeTruthy()
    expect(screen.getByText('Fit unknown')).toBeTruthy()
    expect(screen.getByText('Recommended')).toBeTruthy()
    expect(screen.getAllByText('AWQ').length).toBeGreaterThan(0)
    expect(screen.getByText('Instruct')).toBeTruthy()
    expect(screen.getByText('FP8')).toBeTruthy()

    const downloadFit = screen.getByRole('button', { name: /download · 5\.0 GB/i })
    expect((downloadFit as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(downloadFit)
    await waitFor(() => {
      expect(mocked.downloadVllmModel).toHaveBeenCalledWith('solidrust/Hermes-3-Llama-3.1-8B-AWQ')
    })

    fireEvent.click(screen.getByRole('button', { name: /show models that don't fit/i }))
    expect(screen.getByText('Qwen/Qwen3-32B-AWQ')).toBeTruthy()
    expect(screen.getByText('Too big for this machine')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^download$/i })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /^use$/i }))
    await waitFor(() => {
      expect(mocked.useVllm).toHaveBeenCalledWith('acme/mystery-fp8')
    })
    expect(mocked.setVllmModel).not.toHaveBeenCalled()
  })

  it('shows vLLM download bytes and percent on the bar', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      models: [{ id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: false,
          added_by_you: false,
          cached: false,
          capabilities: ['awq'],
          display_name: 'hermes3:8b',
          fit: 'fits-gpu',
          fits: true,
          id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
          recommended: true,
          served_model_name: 'hermes3:8b',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        }
      ]
    })
    const downloadJob = {
      detail: 'Downloading solidrust/Hermes-3-Llama-3.1-8B-AWQ',
      done_bytes: 1.2 * 2 ** 30,
      error: null,
      job_id: 'vd1',
      kind: 'model-download' as const,
      model_id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
      percent: 29,
      phase: 'downloading',
      status: 'running' as const,
      target: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
      total_bytes: 4.1 * 2 ** 30
    }
    mocked.getLocalModelsJobs.mockResolvedValue({ jobs: [downloadJob] })
    $localRuntimeJobs.set([downloadJob])
    renderPane()

    expect(await screen.findByText('Fits your GPU')).toBeTruthy()
    expect(screen.getByText(/1\.2 \/ 4\.1 GB/)).toBeTruthy()
    expect(screen.getByText(/29%/)).toBeTruthy()
    expect(screen.queryByText(/Downloading solidrust/)).toBeNull()
  })

  it('tags vLLM Hugging Face hits with honest fit and capabilities', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      models: [{ id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({ models: [] })
    mocked.searchVllmModels.mockResolvedValue({
      hits: [
        {
          capabilities: ['awq', 'instruct'],
          downloads: 9,
          fit: 'fits-gpu',
          gated: false,
          likes: 2,
          recommended: true,
          repo: 'Qwen/Qwen3-8B-AWQ',
          created_at: '2025-03-15T00:00:00.000Z',
          size_label: '4.4 GB',
          updated: '2026-01-01'
        },
        {
          capabilities: ['vision', 'moe', '32k', 'coding'],
          downloads: 1,
          fit: 'unknown',
          gated: false,
          likes: 0,
          repo: 'someone/mystery-weights',
          updated: ''
        },
        {
          cached: true,
          capabilities: ['awq'],
          downloads: 3,
          fit: 'too-big',
          gated: false,
          likes: 0,
          repo: 'Qwen/Qwen3-32B-AWQ',
          updated: ''
        }
      ]
    })

    renderPane()
    const box = await screen.findByPlaceholderText(/search models/i)
    fireEvent.change(box, { target: { value: 'qwen' } })
    await waitFor(() => {
      expect(mocked.searchVllmModels).toHaveBeenCalledWith('qwen')
    })

    expect(screen.getByText('Qwen/Qwen3-8B-AWQ')).toBeTruthy()
    expect(screen.getAllByText('Released Mar 2025')).toHaveLength(1)
    expect(screen.getByText('4.4 GB')).toBeTruthy()
    expect(screen.getByText('Fits your GPU')).toBeTruthy()
    expect(screen.getByText('Fit unknown')).toBeTruthy()
    expect(screen.getByText('Too big for this machine')).toBeTruthy()
    expect(screen.getByText('Recommended')).toBeTruthy()
    expect(screen.getByText('Instruct')).toBeTruthy()
    expect(screen.getByText('Sees images')).toBeTruthy()
    expect(screen.getByText('MoE')).toBeTruthy()
    expect(screen.getByText('32k')).toBeTruthy()
    expect(screen.getByText('Coding')).toBeTruthy()
    expect(screen.queryByText('someone/mystery-weights')).toBeTruthy()

    const downloads = screen.getAllByRole('button', { name: /^download$/i })
    expect(downloads.length).toBeGreaterThanOrEqual(2)
    expect(screen.queryByRole('button', { name: /^use$/i })).toBeNull()
  })

  it('greys out Download on a gated Hugging Face hit and does not start a pull', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      models: [],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({ models: [] })
    mocked.searchVllmModels.mockResolvedValue({
      hits: [
        {
          capabilities: ['instruct'],
          downloads: 9,
          fit: 'unknown',
          gated: true,
          likes: 2,
          repo: 'google/gemma-3-27b-it',
          updated: ''
        }
      ]
    })
    renderPane()
    fireEvent.change(await screen.findByPlaceholderText(/search models/i), { target: { value: 'gemma' } })
    await waitFor(() => {
      expect(mocked.searchVllmModels).toHaveBeenCalledWith('gemma')
    })

    expect(screen.getByText('google/gemma-3-27b-it')).toBeTruthy()
    expect(screen.getByText(/requires Hugging Face sign-in/i)).toBeTruthy()
    const gatedDownload = screen.getByRole('button', { name: /^download$/i })
    expect((gatedDownload as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(gatedDownload)
    expect(mocked.downloadVllmModel).not.toHaveBeenCalled()
  })

  it('hides Use on a cached vLLM row that is too big and still offers Download on an uncached too-big row', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      models: [],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: false,
          added_by_you: true,
          cached: true,
          capabilities: ['bf16'],
          display_name: 'hermes3:8b',
          fit: 'too-big',
          fits: false,
          id: 'NousResearch/Hermes-3-Llama-3.1-8B',
          recommended: false,
          served_model_name: 'hermes3:8b',
          size_bytes: 16 * 2 ** 30,
          size_label: '16.0 GB'
        },
        {
          active: false,
          added_by_you: false,
          cached: false,
          capabilities: ['awq'],
          display_name: 'qwen3:32b',
          fit: 'too-big',
          fits: false,
          id: 'Qwen/Qwen3-32B-AWQ',
          recommended: false,
          served_model_name: 'qwen3:32b',
          size_bytes: 0,
          size_label: '—'
        }
      ]
    })
    renderPane()

    expect(await screen.findAllByText('Too big for this machine')).not.toHaveLength(0)
    expect(screen.queryByRole('button', { name: /^use$/i })).toBeNull()
    expect(screen.queryByText('Qwen/Qwen3-32B-AWQ')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /show models that don't fit/i }))
    expect(screen.getByText('Qwen/Qwen3-32B-AWQ')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^download$/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /^use$/i })).toBeNull()
  })

  it('shows Finishing download after HF bytes hit 100%, not an install hang', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      models: [{ id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: false,
          added_by_you: false,
          cached: false,
          capabilities: ['awq'],
          display_name: 'hermes3:8b',
          fit: 'fits-gpu',
          fits: true,
          id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
          recommended: true,
          served_model_name: 'hermes3:8b',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        }
      ]
    })
    const downloadJob = {
      detail: 'Finishing download',
      done_bytes: 4.1 * 2 ** 30,
      error: null,
      job_id: 'vd-verify',
      kind: 'model-download' as const,
      model_id: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
      percent: 100,
      phase: 'verifying',
      status: 'running' as const,
      target: 'solidrust/Hermes-3-Llama-3.1-8B-AWQ',
      total_bytes: 4.1 * 2 ** 30
    }
    mocked.getLocalModelsJobs.mockResolvedValue({ jobs: [downloadJob] })
    $localRuntimeJobs.set([downloadJob])
    renderPane()

    expect(await screen.findByText('Finishing download')).toBeTruthy()
    expect(screen.queryByText(/installing/i)).toBeNull()
  })

  it('shows the llama.cpp engine-update icons for vLLM idle / available / current', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      models: [{ id: 'Qwen/Qwen3-8B-AWQ', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }],
      runtime_installed: true,
      tag: '0.28.0',
      configured_tag: '0.28.0',
      update_available: false,
      venv_ready: true
    })
    renderPane()

    expect(await screen.findByText('Engine up to date')).toBeTruthy()
    expect(screen.getByRole('button', { name: /check for update/i })).toBeTruthy()
    expect(screen.getByText('vLLM 0.28.0')).toBeTruthy()
    expect(screen.queryByText(/the latest release on PyPI/i)).toBeNull()
    expect(screen.queryByText(/installed at/i)).toBeNull()
    expect(screen.queryByText(/Download a model, then Use/i)).toBeNull()
    expect(screen.queryByText(/Hermes starts and manages the server for you/i)).toBeNull()
  })

  it('shows Engine update available with the same update action as llama.cpp', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      models: [{ id: 'Qwen/Qwen3-8B-AWQ', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }],
      runtime_installed: true,
      tag: '0.27.1',
      configured_tag: '0.28.0',
      update_available: true,
      venv_ready: true
    })
    renderPane()

    expect(await screen.findByText('Engine update available')).toBeTruthy()
    expect(screen.getByRole('button', { name: /update engine/i })).toBeTruthy()
  })

  it('deleting the last vLLM model shows first-time setup and does not start a download', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      models: [{ id: 'Qwen/Qwen3-8B-AWQ', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: true,
          added_by_you: false,
          cached: true,
          capabilities: ['awq'],
          display_name: 'qwen3:8b',
          fit: 'fits-gpu',
          fits: true,
          id: 'Qwen/Qwen3-8B-AWQ',
          recommended: true,
          served_model_name: 'qwen3:8b',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        }
      ]
    })
    mocked.deleteVllmModel.mockImplementation(async () => {
      mocked.getLocalModelsStatus.mockResolvedValue({
        ...VLLM_STATUS,
        models: [],
        runtime_installed: true,
        tag: '0.10.0',
        venv_ready: true,
        last_error: 'model was removed — Download to use again'
      })
      mocked.getVllmModels.mockResolvedValue({
        models: [
          {
            active: false,
            added_by_you: false,
            cached: false,
            capabilities: ['awq'],
            display_name: 'qwen3:8b',
            fit: 'fits-gpu',
            fits: true,
            id: 'Qwen/Qwen3-8B-AWQ',
            recommended: true,
            served_model_name: 'qwen3:8b',
            size_bytes: 5 * 2 ** 30,
            size_label: '5.0 GB'
          }
        ]
      })
      return { ok: true }
    })
    renderPane()

    const trash = await screen.findByRole('button', { name: /delete model/i })
    fireEvent.click(trash)
    await waitFor(() => {
      expect(mocked.deleteVllmModel).toHaveBeenCalledWith('Qwen/Qwen3-8B-AWQ')
    })
    expect(mocked.quickstartLocalModels).not.toHaveBeenCalled()
    expect(mocked.downloadVllmModel).not.toHaveBeenCalled()
    expect(await screen.findByRole('button', { name: /set up for me/i })).toBeTruthy()
    const downloadAgain = await screen.findByRole('button', { name: /download · 5\.0 GB/i })
    expect((downloadAgain as HTMLButtonElement).disabled).toBe(false)
    confirm.mockRestore()
  })

  it('Use switches back to a cached recommended model without running setup', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      model: 'acme/sideload-awq',
      served_model_name: 'sideload',
      active_model_id: 'sideload',
      server_running: true,
      server_base_url: 'http://127.0.0.1:18435/v1',
      models: [
        { id: 'Qwen/Qwen3-8B-AWQ', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' },
        { id: 'acme/sideload-awq', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }
      ],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: false,
          added_by_you: false,
          cached: true,
          capabilities: ['awq'],
          display_name: 'qwen3:8b',
          fit: 'fits-gpu',
          fits: true,
          id: 'Qwen/Qwen3-8B-AWQ',
          recommended: true,
          served_model_name: 'qwen3:8b',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        },
        {
          active: true,
          added_by_you: true,
          cached: true,
          capabilities: ['awq'],
          display_name: 'sideload',
          fit: 'fits-gpu',
          fits: true,
          id: 'acme/sideload-awq',
          recommended: false,
          served_model_name: 'sideload',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        }
      ]
    })
    renderPane()

    expect(await screen.findByRole('button', { name: /^use$/i })).toBeTruthy()
    expect(screen.getByText('In use')).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^in use$/i })).toBeNull()
    expect(screen.getAllByRole('button', { name: /^use$/i })).toHaveLength(1)
    expect(screen.queryByRole('button', { name: /set up for me/i })).toBeNull()
    expect(screen.getByRole('button', { name: /restore recommended setup/i })).toBeTruthy()
    expect(screen.getByText('Qwen/Qwen3-8B-AWQ')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /^use$/i }))
    await waitFor(() => {
      expect(mocked.useVllm).toHaveBeenCalledWith('Qwen/Qwen3-8B-AWQ')
    })
    expect(mocked.quickstartLocalModels).not.toHaveBeenCalled()
  })

  it('Restore recommended setup is a quiet failsafe, not the switch-back path', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      model: 'acme/sideload-awq',
      served_model_name: 'sideload',
      active_model_id: 'sideload',
      server_running: true,
      server_base_url: 'http://127.0.0.1:18435/v1',
      models: [{ id: 'acme/sideload-awq', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: false,
          added_by_you: false,
          cached: false,
          capabilities: ['awq'],
          display_name: 'qwen3:8b',
          fit: 'fits-gpu',
          fits: true,
          id: 'Qwen/Qwen3-8B-AWQ',
          recommended: true,
          served_model_name: 'qwen3:8b',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        },
        {
          active: true,
          added_by_you: true,
          cached: true,
          capabilities: ['awq'],
          display_name: 'sideload',
          fit: 'fits-gpu',
          fits: true,
          id: 'acme/sideload-awq',
          recommended: false,
          served_model_name: 'sideload',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        }
      ]
    })
    mocked.quickstartLocalModels.mockResolvedValue({
      display_name: 'qwen3:8b',
      download_bytes: 0,
      job_id: 'vq-reset',
      model_id: 'Qwen/Qwen3-8B-AWQ',
      needs_download: true,
      needs_runtime: false
    })
    renderPane()

    expect(await screen.findByRole('button', { name: /download · 5\.0 GB/i })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /set up for me/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /^use$/i })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /restore recommended setup/i }))
    await waitFor(() => {
      expect(mocked.quickstartLocalModels).toHaveBeenCalledWith()
    })
    expect(mocked.useVllm).not.toHaveBeenCalled()
    expect(mocked.deleteVllmModel).not.toHaveBeenCalled()
  })

  it('does not show In use until /v1/models is ready', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      model: 'NousResearch/Hermes-3-Llama-3.1-8B',
      served_model_name: null,
      active_model_id: null,
      server_running: false,
      start_phase: 'Capturing CUDA graphs',
      server_base_url: 'http://127.0.0.1:18435/v1',
      models: [
        { id: 'Qwen/Qwen3-8B-AWQ', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' },
        { id: 'NousResearch/Hermes-3-Llama-3.1-8B', size_bytes: 16 * 2 ** 30, size_label: '16.0 GB' }
      ],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: false,
          added_by_you: false,
          cached: true,
          capabilities: ['awq'],
          display_name: 'qwen3:8b',
          fit: 'fits-gpu',
          fits: true,
          id: 'Qwen/Qwen3-8B-AWQ',
          recommended: true,
          served_model_name: 'qwen3:8b',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        },
        {
          active: false,
          added_by_you: true,
          cached: true,
          capabilities: [],
          display_name: 'Hermes-3-Llama-3.1-8B',
          fit: 'too-big',
          fits: false,
          id: 'NousResearch/Hermes-3-Llama-3.1-8B',
          recommended: false,
          served_model_name: 'Hermes-3-Llama-3.1-8B',
          size_bytes: 16 * 2 ** 30,
          size_label: '16.0 GB'
        }
      ]
    })
    renderPane()

    expect(await screen.findByText('Qwen/Qwen3-8B-AWQ')).toBeTruthy()
    expect(screen.queryByText('In use')).toBeNull()
    expect(screen.getByRole('button', { name: /^use$/i })).toBeTruthy()
  })

  it('shows In use on the served cached model and Use on other cached rows', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      // Config still names the recommended row (written on Use click before
      // /v1/models). Live serve is the sideload — only that row is In use.
      model: 'Qwen/Qwen3-8B-AWQ',
      served_model_name: 'sideload',
      active_model_id: 'sideload',
      server_running: true,
      server_base_url: 'http://127.0.0.1:18435/v1',
      models: [
        { id: 'Qwen/Qwen3-8B-AWQ', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' },
        { id: 'acme/sideload-awq', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }
      ],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: true,
          added_by_you: false,
          cached: true,
          capabilities: ['awq'],
          display_name: 'qwen3:8b',
          fit: 'fits-gpu',
          fits: true,
          id: 'Qwen/Qwen3-8B-AWQ',
          recommended: true,
          served_model_name: 'qwen3:8b',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        },
        {
          active: false,
          added_by_you: true,
          cached: true,
          capabilities: ['awq'],
          display_name: 'sideload',
          fit: 'fits-gpu',
          fits: true,
          id: 'acme/sideload-awq',
          recommended: false,
          served_model_name: 'sideload',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        }
      ]
    })
    renderPane()

    const recommendedRow = (await screen.findByText('Qwen/Qwen3-8B-AWQ')).closest('[class*="@container"]')
    const servedRow = screen.getByText('acme/sideload-awq').closest('[class*="@container"]')
    expect(recommendedRow).toBeTruthy()
    expect(servedRow).toBeTruthy()
    expect(within(recommendedRow as HTMLElement).getByRole('button', { name: /^use$/i })).toBeTruthy()
    expect(within(recommendedRow as HTMLElement).queryByText('In use')).toBeNull()
    expect(within(servedRow as HTMLElement).getByText('In use')).toBeTruthy()
    expect(within(servedRow as HTMLElement).queryByRole('button', { name: /^use$/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /^in use$/i })).toBeNull()
    expect(screen.getAllByRole('button', { name: /^use$/i })).toHaveLength(1)
  })

  it('left-aligns Restore recommended setup as a tight ghost action', async () => {
    mocked.getLocalModelsStatus.mockResolvedValue({
      ...VLLM_STATUS,
      models: [{ id: 'acme/sideload-awq', size_bytes: 5 * 2 ** 30, size_label: '5.0 GB' }],
      runtime_installed: true,
      tag: '0.10.0',
      venv_ready: true
    })
    mocked.getVllmModels.mockResolvedValue({
      models: [
        {
          active: true,
          added_by_you: true,
          cached: true,
          capabilities: ['awq'],
          display_name: 'sideload',
          fit: 'fits-gpu',
          fits: true,
          id: 'acme/sideload-awq',
          recommended: false,
          served_model_name: 'sideload',
          size_bytes: 5 * 2 ** 30,
          size_label: '5.0 GB'
        }
      ]
    })
    renderPane()

    const restore = await screen.findByRole('button', { name: /restore recommended setup/i })
    const wrap = restore.closest('.justify-start')
    expect(wrap).toBeTruthy()
    expect(wrap!.className.split(/\s+/)).toEqual(expect.arrayContaining(['mt-1', 'flex', 'justify-start']))
    expect(wrap!.className.split(/\s+/)).not.toEqual(expect.arrayContaining(['py-2', 'mt-3']))
    expect(restore.className.split(/\s+/)).toEqual(expect.arrayContaining(['h-auto', 'px-0']))
  })
})
