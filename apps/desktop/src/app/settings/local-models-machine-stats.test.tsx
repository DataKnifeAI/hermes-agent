import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { LocalHardware, LocalModelsStatus } from '@/types/hermes'

import { LocalModelsMachineStats } from './local-models-machine-stats'

const HARDWARE: LocalHardware = {
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
  engine: 'vllm',
  models_dir: '/tmp/hermes-home/models',
  models_dir_display: '~/.hermes/models',
  models_storage_bytes: 17.6 * 2 ** 30,
  disk_free_bytes: 800 * 2 ** 30,
  disk_total_bytes: 2000 * 2 ** 30,
  runtime_dir: '/tmp/hermes-home/runtimes/vllm',
  runtime_dir_display: '~/.hermes/runtimes/vllm',
  occupancy_foreign: false,
  ctx_64k_feasible: true
}

const STATUS: LocalModelsStatus = {
  enabled: true,
  engine: 'vllm',
  tag: '0.14.1',
  configured_tag: '0.14.1',
  update_available: false,
  runtime_installed: true,
  runtime_backend: 'vllm',
  server_running: true,
  server_base_url: 'http://127.0.0.1:18435/v1',
  active_model_id: 'hermes3:8b',
  loaded_models: {},
  models: [],
  models_dir: '/tmp/hermes-home/models',
  vllm_version: '0.14.1',
  served_model_name: 'hermes3:8b'
}

function renderStats(overrides: { hardware?: Partial<LocalHardware>; status?: Partial<LocalModelsStatus> | null } = {}) {
  render(
    <I18nProvider>
      <LocalModelsMachineStats
        engine="vllm"
        hardware={{ ...HARDWARE, ...overrides.hardware }}
        status={overrides.status === null ? null : { ...STATUS, ...overrides.status }}
      />
    </I18nProvider>
  )
}

describe('LocalModelsMachineStats', () => {
  it('always shows GPU, VRAM, and RAM; hides the rest until Show more', () => {
    renderStats()

    expect(screen.getByText('NVIDIA GeForce RTX 5090')).toBeTruthy()
    expect(screen.getByText(/6\.0 GB \/ 32\.0 GB used/)).toBeTruthy()
    expect(screen.getByText(/56\.0 GB \/ 256\.0 GB used/)).toBeTruthy()
    expect(screen.queryByText(/driver 560/)).toBeNull()
    expect(screen.queryByText(/this engine/)).toBeNull()
    expect(screen.queryByText(/~\/\.hermes\/models/)).toBeNull()
    expect(screen.queryByText(/vLLM 0\.14\.1/)).toBeNull()
    expect(screen.queryByText('hermes3:8b')).toBeNull()
    expect(screen.queryByText(/64k context/)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /^show more$/i }))
    expect(screen.getByText(/driver 560/)).toBeTruthy()
    expect(screen.getByText(/this engine 5\.0 GB/)).toBeTruthy()
    expect(screen.getByText(/~\/\.hermes\/models/)).toBeTruthy()
    expect(screen.getByText(/vLLM 0\.14\.1/)).toBeTruthy()
    expect(screen.getByText('hermes3:8b')).toBeTruthy()
    expect(screen.getByText(/64k context fits this GPU/)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /^show less$/i }))
    expect(screen.queryByText(/vLLM 0\.14\.1/)).toBeNull()
    expect(screen.getByText('NVIDIA GeForce RTX 5090')).toBeTruthy()
  })

  it('uses hardware.vllm_version on the Engine line before status.tag', () => {
    renderStats({ hardware: { vllm_version: '1.2.3' }, status: { tag: '', vllm_version: null } })
    fireEvent.click(screen.getByRole('button', { name: /^show more$/i }))
    expect(screen.getByText(/vLLM 1\.2\.3/)).toBeTruthy()
  })

  it('falls back to status.tag when vllm_version is absent', () => {
    renderStats({ status: { tag: '0.9.0', vllm_version: null } })
    fireEvent.click(screen.getByRole('button', { name: /^show more$/i }))
    expect(screen.getByText(/vLLM 0\.9\.0/)).toBeTruthy()
  })
})
