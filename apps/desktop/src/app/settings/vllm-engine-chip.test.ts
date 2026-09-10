import { describe, expect, it } from 'vitest'

import { vllmEngineChip, vllmEnginePower, type VllmEngineChipCopy } from './vllm-engine-chip'

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

  it('shows Starting when a stale Stopped arrives with a live pid or start job', () => {
    expect(
      vllmEngineChip({
        copy,
        status: { ...idle, engine_state: 'stopped', pid: 88 }
      })?.label
    ).toBe('Starting')

    expect(
      vllmEngineChip({
        copy,
        jobs: [{ kind: 'quickstart', phase: 'starting-server', status: 'running' }],
        status: idle
      })?.label
    ).toBe('Starting')

    expect(
      vllmEngineChip({
        copy,
        serverBusy: true,
        status: idle
      })?.label
    ).toBe('Starting')
  })

  it('does not treat leftover start_phase as Starting when the API says stopped', () => {
    expect(
      vllmEngineChip({
        copy,
        status: { ...idle, engine_state: 'stopped', start_phase: 'Capturing CUDA graphs' }
      })
    ).toEqual({ label: 'Stopped', tone: 'muted' })
  })

  it('follows engine_state for the Engine-row power button', () => {
    const powerCopy = { ...copy, startServer: 'Turn on', stopServer: 'Turn off' }

    expect(
      vllmEnginePower({
        copy: powerCopy,
        status: { ...idle, engine_state: 'starting', start_phase: 'Capturing CUDA graphs' }
      })
    ).toEqual({ action: null, disabled: true, label: 'Starting' })

    expect(
      vllmEnginePower({
        copy: powerCopy,
        status: { ...idle, engine_state: 'ready', server_running: true, served_model_name: 'qwen3:14b' }
      })
    ).toEqual({ action: 'stop', disabled: false, label: 'Turn off' })

    expect(vllmEnginePower({ copy: powerCopy, status: idle })).toEqual({
      action: 'start',
      disabled: false,
      label: 'Turn on'
    })

    expect(
      vllmEnginePower({
        copy: powerCopy,
        status: { ...idle, engine_state: 'error', last_error: 'CUDA OOM' }
      })
    ).toEqual({ action: 'start', disabled: false, label: 'Turn on' })
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
        jobs: [{ kind: 'vllm-install', phase: 'installing-venv', status: 'running' }],
        status: { ...idle, engine_state: 'not_installed', runtime_installed: false }
      })
    ).toEqual({ label: 'Installing', tone: 'warn' })

    expect(
      vllmEngineChip({
        copy,
        jobs: [{ kind: 'vllm-update', phase: 'updating-venv', status: 'running' }],
        status: idle
      })
    ).toEqual({ label: 'Updating', tone: 'warn' })
  })
})
