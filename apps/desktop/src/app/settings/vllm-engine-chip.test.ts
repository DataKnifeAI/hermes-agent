import { describe, expect, it } from 'vitest'

import { vllmEngineChip, type VllmEngineChipCopy } from './vllm-engine-chip'

const copy: VllmEngineChipCopy = {
  engineFailed: 'Failed',
  engineInstalling: 'Installing',
  engineReady: 'Ready',
  engineStarting: 'Starting',
  engineStopped: 'Stopped',
  engineUpdating: 'Updating'
}

const idle = {
  active_model_id: null,
  engine_state: 'stopped' as const,
  last_error: null,
  runtime_installed: true,
  served_model_name: null,
  server_running: false,
  start_phase: null
}

describe('vllmEngineChip', () => {
  it('shows Ready only when the endpoint is healthy', () => {
    expect(
      vllmEngineChip({
        copy,
        status: {
          ...idle,
          engine_state: 'ready',
          served_model_name: 'qwen3:14b',
          server_running: true
        }
      })
    ).toEqual({ label: 'Ready', tone: 'success' })

    expect(
      vllmEngineChip({
        copy,
        status: {
          ...idle,
          engine_state: 'starting',
          start_phase: 'Capturing CUDA graphs'
        }
      })
    ).toEqual({ label: 'Starting', tone: 'warn' })

    expect(vllmEngineChip({ copy, status: idle })).toEqual({ label: 'Stopped', tone: 'muted' })
  })

  it('does not show Ready on spawn when engine_state is missing', () => {
    const spawned = {
      ...idle,
      engine_state: undefined,
      server_running: false,
      start_phase: 'Loading weights'
    }

    expect(vllmEngineChip({ copy, status: spawned })?.label).toBe('Starting')
    expect(vllmEngineChip({ copy, status: spawned })?.label).not.toMatch(/^Ready/)
  })

  it('labels error, install, and update without claiming Ready', () => {
    expect(
      vllmEngineChip({
        copy,
        status: { ...idle, engine_state: 'error', last_error: 'CUDA OOM' }
      })
    ).toEqual({ label: 'Failed', tone: 'destructive' })

    expect(
      vllmEngineChip({
        copy,
        jobs: [{ kind: 'vllm-install', status: 'running' }],
        status: { ...idle, engine_state: 'not_installed', runtime_installed: false }
      })
    ).toEqual({ label: 'Installing', tone: 'warn' })

    expect(
      vllmEngineChip({
        copy,
        jobs: [{ kind: 'vllm-update', status: 'running' }],
        status: idle
      })
    ).toEqual({ label: 'Updating', tone: 'warn' })
  })
})
