import type { LocalModelsStatus, LocalRuntimeJob, ManagedLocalEngine } from '@/types/hermes'

export type VllmEngineChipTone = 'destructive' | 'muted' | 'primary' | 'success' | 'warn'

export interface VllmEngineChipCopy {
  engineFailed: string
  engineInstalling: string
  engineReady: string
  engineStarting: string
  engineStopped: string
  engineUpdating: string
}

export interface VllmEngineChip {
  label: string
  tone: VllmEngineChipTone
}

const START_JOB_PHASES = new Set(['setting-default', 'starting', 'starting-server'])

function startJobRunning(jobs: readonly Pick<LocalRuntimeJob, 'kind' | 'phase' | 'status'>[]): boolean {
  return jobs.some(
    j => j.status === 'running' && (j.kind === 'quickstart' ? START_JOB_PHASES.has(j.phase) : false)
  )
}

export function vllmEngineState(
  status: Pick<
    LocalModelsStatus,
    'engine_state' | 'last_error' | 'pid' | 'runtime_installed' | 'server_running' | 'start_phase'
  >,
  extra: { serverBusy?: boolean; startJob?: boolean } = {}
): LocalModelsStatus['engine_state'] {
  if (status.server_running || status.engine_state === 'ready') {
    return 'ready'
  }

  // Starting beats a stale Stopped from an older backend that mapped
  // "not healthy yet" → stopped. Pid / start job / in-flight Use are live.
  // Leftover log needles (start_phase) are not a live serve — only used
  // when the backend omitted engine_state entirely.
  if (
    status.engine_state === 'starting' ||
    extra.serverBusy ||
    extra.startJob ||
    status.pid != null ||
    (!status.engine_state && Boolean(status.start_phase))
  ) {
    return 'starting'
  }

  if (status.engine_state === 'error' || status.last_error) {
    return 'error'
  }

  if (status.engine_state === 'stopped') {
    return 'stopped'
  }

  if (status.engine_state === 'not_installed') {
    return 'not_installed'
  }

  return status.runtime_installed ? 'stopped' : 'not_installed'
}

export type VllmEnginePowerAction = 'start' | 'stop' | null

export interface VllmEnginePower {
  action: VllmEnginePowerAction
  disabled: boolean
  label: string
}

export function vllmEnginePower({
  copy,
  jobs = [],
  serverBusy = false,
  status
}: {
  copy: Pick<VllmEngineChipCopy, 'engineStarting'> & { startServer: string; stopServer: string }
  jobs?: readonly Pick<LocalRuntimeJob, 'kind' | 'phase' | 'status'>[]
  serverBusy?: boolean
  status: Pick<
    LocalModelsStatus,
    'engine_state' | 'last_error' | 'pid' | 'runtime_installed' | 'server_running' | 'start_phase'
  >
}): VllmEnginePower {
  const state = vllmEngineState(status, { serverBusy, startJob: startJobRunning(jobs) })

  if (state === 'ready') {
    return { action: 'stop', disabled: serverBusy, label: copy.stopServer }
  }

  if (state === 'starting') {
    // Previous serve still healthy → Turn off stays. Never offer Turn on.
    if (status.server_running) {
      return { action: 'stop', disabled: serverBusy, label: copy.stopServer }
    }

    return { action: null, disabled: true, label: copy.engineStarting }
  }

  if (state === 'not_installed') {
    return { action: null, disabled: true, label: copy.startServer }
  }

  return { action: 'start', disabled: serverBusy, label: copy.startServer }
}

export function vllmEngineChip({
  copy,
  jobs = [],
  serverBusy = false,
  status
}: {
  copy: VllmEngineChipCopy
  jobs?: readonly Pick<LocalRuntimeJob, 'kind' | 'phase' | 'status'>[]
  serverBusy?: boolean
  status: Pick<
    LocalModelsStatus,
    'engine_state' | 'last_error' | 'pid' | 'runtime_installed' | 'server_running' | 'start_phase'
  >
}): null | VllmEngineChip {
  if (jobs.some(j => j.kind === 'vllm-install' && j.status === 'running')) {
    return { label: copy.engineInstalling, tone: 'warn' }
  }

  if (jobs.some(j => j.kind === 'vllm-update' && j.status === 'running')) {
    return { label: copy.engineUpdating, tone: 'warn' }
  }

  const state = vllmEngineState(status, { serverBusy, startJob: startJobRunning(jobs) })

  if (state === 'ready') {
    return { label: copy.engineReady, tone: 'success' }
  }

  if (state === 'starting') {
    return { label: copy.engineStarting, tone: 'warn' }
  }

  if (state === 'error') {
    return { label: copy.engineFailed, tone: 'destructive' }
  }

  if (state === 'stopped') {
    return { label: copy.engineStopped, tone: 'muted' }
  }

  return null
}

function chipStatus(row: ManagedLocalEngine): Parameters<typeof vllmEngineState>[0] {
  return {
    engine_state: row.engine_state,
    last_error: row.last_error,
    pid: row.pid,
    runtime_installed: Boolean(row.runtime_installed),
    server_running: Boolean(row.server_running),
    start_phase: row.start_phase
  }
}

function rowDevice(row: ManagedLocalEngine): 'cpu' | 'gpu' | null {
  if (row.device === 'cpu' || row.device === 'gpu') {
    return row.device
  }

  if (row.engine === 'vllm-cpu') {
    return 'cpu'
  }

  if (row.engine === 'vllm') {
    return 'gpu'
  }

  return null
}

export interface VllmDeviceRuntimePart {
  device: 'cpu' | 'gpu'
  label: string
  tone: VllmEngineChipTone
}

function deviceState(
  row: ManagedLocalEngine | undefined,
  copy: VllmEngineChipCopy
): { label: string; tone: VllmEngineChipTone } {
  if (!row) {
    return { label: copy.engineStopped, tone: 'muted' }
  }

  const state = vllmEngineState(chipStatus(row))

  if (state === 'ready') {
    return { label: copy.engineReady, tone: 'success' }
  }

  if (state === 'starting') {
    return { label: copy.engineStarting, tone: 'warn' }
  }

  if (state === 'error') {
    return { label: copy.engineFailed, tone: 'destructive' }
  }

  return { label: copy.engineStopped, tone: 'muted' }
}

/**
 * Runtime status for both servers, icon-first in the UI.
 * Labels are Ready / Stopped / … — no device word and no model name.
 */
export function vllmDevicesRuntimeParts(
  engines: readonly ManagedLocalEngine[] | undefined,
  copy: VllmEngineChipCopy
): null | VllmDeviceRuntimePart[] {
  if (!engines?.length) {
    return null
  }

  let gpu: ManagedLocalEngine | undefined
  let cpu: ManagedLocalEngine | undefined

  for (const row of engines) {
    const device = rowDevice(row)

    if (device === 'gpu') {
      gpu = row
    } else if (device === 'cpu') {
      cpu = row
    }
  }

  if (!gpu && !cpu) {
    return null
  }

  const gpuState = deviceState(gpu, copy)
  const cpuState = deviceState(cpu, copy)

  return [
    { device: 'gpu', label: gpuState.label, tone: gpuState.tone },
    { device: 'cpu', label: cpuState.label, tone: cpuState.tone }
  ]
}

/**
 * Installed package per venv: "GPU 0.10.0 · CPU 0.11.2".
 * A missing venv is omitted. The fallback fills only the selected device, and
 * only when rows did not report versions (older backend).
 */
export function vllmInstalledVersionsLine(
  engines: readonly ManagedLocalEngine[] | undefined,
  copy: { deviceCpu: string; deviceGpu: string },
  fallback?: { device?: 'cpu' | 'gpu' | null; version?: null | string }
): null | string {
  const found: Partial<Record<'cpu' | 'gpu', string>> = {}
  let reported = false

  for (const row of engines ?? []) {
    const device = rowDevice(row)

    if (!device) {
      continue
    }

    if (row.vllm_version !== undefined) {
      reported = true
    }

    const version = (row.vllm_version || '').trim()

    if (version) {
      found[device] = version
    }
  }

  const fallbackVersion = (fallback?.version || '').trim()

  if (!reported && fallback?.device && fallbackVersion) {
    found[fallback.device] = fallbackVersion
  }

  const parts = [found.gpu ? `${copy.deviceGpu} ${found.gpu}` : null, found.cpu ? `${copy.deviceCpu} ${found.cpu}` : null].filter(
    (part): part is string => Boolean(part)
  )

  return parts.length ? parts.join(' · ') : null
}
