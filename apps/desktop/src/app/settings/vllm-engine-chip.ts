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

function derivedState(
  status: Pick<LocalModelsStatus, 'engine_state' | 'last_error' | 'runtime_installed' | 'server_running' | 'start_phase'>
): LocalModelsStatus['engine_state'] {
  if (status.engine_state) {
    return status.engine_state
  }

  // Older backends: Ready only after /v1/models (server_running).
  if (status.server_running) {
    return 'ready'
  }

  if (status.start_phase) {
    return 'starting'
  }

  if (status.last_error) {
    return 'error'
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
  jobs?: readonly Pick<LocalRuntimeJob, 'kind' | 'status'>[]
  serverBusy?: boolean
  status: Pick<
    LocalModelsStatus,
    'engine_state' | 'last_error' | 'runtime_installed' | 'server_running' | 'start_phase'
  >
}): null | VllmEngineChip {
  if (jobs.some(j => j.kind === 'vllm-install' && j.status === 'running')) {
    return { label: copy.engineInstalling, tone: 'warn' }
  }

  if (jobs.some(j => j.kind === 'vllm-update' && j.status === 'running')) {
    return { label: copy.engineUpdating, tone: 'warn' }
  }

  const state =
    serverBusy && !status.server_running && derivedState(status) !== 'ready' ? 'starting' : derivedState(status)

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
