import { describe, expect, it } from 'vitest'

import {
  vllmDevicesRuntimeLine,
  vllmEngineChip,
  vllmEnginePower,
  vllmInstalledVersionsLine,
  type VllmEngineChipCopy
} from './vllm-engine-chip'

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

describe('vllmDevicesRuntimeLine', () => {
  const devices = { ...copy, deviceCpu: 'CPU', deviceGpu: 'GPU' }

  it('names both devices without a model id', () => {
    expect(
      vllmDevicesRuntimeLine(
        [
          { device: 'gpu', engine: 'vllm', engine_state: 'ready', server_running: true, runtime_installed: true },
          {
            device: 'cpu',
            engine: 'vllm',
            engine_state: 'stopped',
            server_running: false,
            runtime_installed: true,
            served_model_name: 'qwen3:4b'
          }
        ],
        devices
      )
    ).toBe('GPU Ready · CPU Stopped')
    expect(
      vllmDevicesRuntimeLine(
        [
          {
            device: 'gpu',
            engine: 'vllm',
            engine_state: 'ready',
            server_running: true,
            runtime_installed: true,
            served_model_name: 'qwen3:14b'
          }
        ],
        devices
      )
    ).not.toMatch(/qwen3/)
  })

  it('keeps a live CPU server visible when the GPU row is stopped', () => {
    const line = vllmDevicesRuntimeLine(
      [
        { engine: 'vllm', engine_state: 'stopped', server_running: false, runtime_installed: true },
        {
          engine: 'vllm-cpu',
          engine_state: 'ready',
          server_running: true,
          runtime_installed: true
        }
      ],
      devices
    )

    expect(line).toBe('GPU Stopped · CPU Ready')
    expect(line).not.toMatch(/CPU Stopped/)
  })
})

describe('vllmInstalledVersionsLine', () => {
  const devices = { deviceCpu: 'CPU', deviceGpu: 'GPU' }

  it('names each venv version and omits a device that is not installed', () => {
    expect(
      vllmInstalledVersionsLine(
        [
          { device: 'gpu', engine: 'vllm', vllm_version: '0.10.0' },
          { device: 'cpu', engine: 'vllm-cpu', vllm_version: '0.11.2' }
        ],
        devices
      )
    ).toBe('GPU 0.10.0 · CPU 0.11.2')

    expect(
      vllmInstalledVersionsLine(
        [
          { device: 'gpu', engine: 'vllm', vllm_version: '0.10.0' },
          { device: 'cpu', engine: 'vllm-cpu', vllm_version: null }
        ],
        devices,
        { device: 'cpu', version: '0.10.0' }
      )
    ).toBe('GPU 0.10.0')
  })

  it('uses the selected-device version only when rows did not report one', () => {
    expect(vllmInstalledVersionsLine(undefined, devices, { device: 'gpu', version: '0.9.0' })).toBe('GPU 0.9.0')
    expect(vllmInstalledVersionsLine(undefined, devices, { device: 'cpu', version: '0.9.0' })).toBe('CPU 0.9.0')
    expect(vllmInstalledVersionsLine([], devices, { device: 'gpu', version: '' })).toBeNull()
  })
})
