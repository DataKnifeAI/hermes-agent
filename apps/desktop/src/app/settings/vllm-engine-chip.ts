import type { LocalModelsStatus, LocalRuntimeJob } from '@/types/hermes'

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
