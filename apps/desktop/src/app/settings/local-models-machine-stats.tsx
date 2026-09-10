import { useState, type ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { AlertTriangle, Cpu, FolderOpen, Package, Zap } from '@/lib/icons'
import type { LocalEngine, LocalHardware, LocalModelsStatus } from '@/types/hermes'

import { Pill } from './primitives'

function gbLabel(bytes: number | null | undefined): string {
  if (bytes == null || bytes < 0) {
    return '—'
  }

  if (!bytes) {
    return '0.0 GB'
  }

  return `${(bytes / (1 << 30)).toFixed(1)} GB`
}

function vllmVersionOf(
  engine: LocalEngine,
  hardware: LocalHardware,
  status: LocalModelsStatus | null
): string {
  if (engine !== 'vllm') {
    return ''
  }

  // Isolated-venv only — hardware and status both come from installed_vllm_version().
  return (hardware.vllm_version || status?.vllm_version || status?.tag || '').trim()
}

function StatLine({
  icon,
  label,
  value
}: {
  icon?: ReactNode
  label: string
  value: ReactNode
}) {
  return (
    <div className="grid grid-cols-[7.5rem_minmax(0,1fr)] items-start gap-x-3 gap-y-0.5 py-0.5">
      <span className="inline-flex items-center gap-1.5 text-muted-foreground">
        {icon}
        {label}
      </span>
      <span className="min-w-0 break-all text-foreground">{value}</span>
    </div>
  )
}

export function LocalModelsMachineStats({
  engine,
  hardware,
  status
}: {
  engine: LocalEngine
  hardware: LocalHardware
  status: LocalModelsStatus | null
}) {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const [detailsOpen, setDetailsOpen] = useState(false)
  const cachePath = hardware.models_dir_display || hardware.models_dir
  const runtimePath = hardware.runtime_dir_display || hardware.runtime_dir || status?.venv_path
  const listenUrl = status?.server_base_url
  const served = status?.served_model_name || status?.active_model_id || status?.model
  const servingLabel = status?.server_running
    ? copy.servingReady
    : status?.start_phase
      ? copy.servingStarting
      : null
  const occupancy =
    hardware.occupancy_foreign || Boolean(status?.occupancy_message) || Boolean(status?.occupancy?.length)
  const used = hardware.vram_used_bytes
  const total = hardware.vram_total_bytes
  const vllmVersion = vllmVersionOf(engine, hardware, status)
  const engineValue = engine === 'vllm'
    ? vllmVersion
      ? `${copy.engineVllm} ${vllmVersion}`
      : copy.engineVllm
    : copy.engineLlama
  const ramUsed =
    hardware.ram_used_bytes ??
    (hardware.ram_total_bytes != null && hardware.ram_available_bytes != null
      ? Math.max(0, hardware.ram_total_bytes - hardware.ram_available_bytes)
      : null)
  const gpuExtras = [
    hardware.gpu_driver_version ? copy.gpuDriver(hardware.gpu_driver_version) : null,
    hardware.cuda_compute_capability ? copy.gpuCompute(hardware.cuda_compute_capability) : null
  ].filter(Boolean)

  return (
    <div className="grid gap-0.5 py-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height)">
      {hardware.gpu_name && (
        <StatLine
          icon={<Zap className="size-3.5" />}
          label="GPU"
          value={
            detailsOpen && gpuExtras.length > 0
              ? [hardware.gpu_name, ...gpuExtras].join(' · ')
              : hardware.gpu_name
          }
        />
      )}

      <StatLine
        icon={<Cpu className="size-3.5" />}
        label="VRAM"
        value={
          <span className="grid gap-0.5">
            <span>
              {used != null && total
                ? `${copy.vramUsed(gbLabel(used), gbLabel(total))}${
                    hardware.vram_free_bytes != null ? ` · ${copy.vramFree(gbLabel(hardware.vram_free_bytes))}` : ''
                  }`
                : copy.vram(gbLabel(total))}
            </span>
            {detailsOpen && (hardware.vram_engine_bytes != null || hardware.vram_other_bytes != null) && (
              <span className="text-muted-foreground">
                {[
                  hardware.vram_engine_bytes != null ? copy.vramEngine(gbLabel(hardware.vram_engine_bytes)) : null,
                  hardware.vram_other_bytes != null ? copy.vramOther(gbLabel(hardware.vram_other_bytes)) : null
                ]
                  .filter(Boolean)
                  .join(' · ')}
              </span>
            )}
          </span>
        }
      />

      <StatLine
        icon={<Package className="size-3.5" />}
        label="RAM"
        value={
          ramUsed != null && hardware.ram_total_bytes
            ? copy.ramUsed(gbLabel(ramUsed), gbLabel(hardware.ram_total_bytes))
            : copy.ram(gbLabel(hardware.ram_total_bytes))
        }
      />

      {detailsOpen && (
        <>
          {cachePath && (
            <StatLine
              icon={<FolderOpen className="size-3.5" />}
              label={copy.modelsPath}
              value={
                <span className="grid gap-0.5">
                  <span className="font-mono text-[0.72rem]">{cachePath}</span>
                  {hardware.models_storage_bytes != null && hardware.disk_free_bytes != null && (
                    <span className="text-muted-foreground">
                      {copy.storageUsed(gbLabel(hardware.models_storage_bytes), gbLabel(hardware.disk_free_bytes))}
                    </span>
                  )}
                </span>
              }
            />
          )}

          {engine === 'vllm' && runtimePath && (
            <StatLine
              icon={<FolderOpen className="size-3.5" />}
              label={copy.runtimePath}
              value={<span className="font-mono text-[0.72rem]">{runtimePath}</span>}
            />
          )}

          <StatLine
            label={copy.engineStat}
            value={
              <span>
                {engineValue}
                {listenUrl ? ` · ${listenUrl}` : ''}
              </span>
            }
          />

          {served && (
            <StatLine
              label={copy.servingStat}
              value={
                <span className="inline-flex flex-wrap items-center gap-1.5">
                  <span>{served}</span>
                  {servingLabel && <Pill tone={status?.server_running ? 'success' : 'warn'}>{servingLabel}</Pill>}
                </span>
              }
            />
          )}

          {hardware.ctx_64k_feasible != null && (
            <StatLine
              label="64k"
              value={hardware.ctx_64k_feasible ? copy.ctx64kFits : copy.ctx64kTight}
            />
          )}

          {hardware.uma && <Pill>{copy.unifiedMemory}</Pill>}

          {occupancy && (
            <p className="inline-flex items-center gap-1.5 pt-1 text-destructive">
              <AlertTriangle className="size-3.5" />
              {status?.occupancy_message || copy.occupancyWarn}
            </p>
          )}
        </>
      )}

      <Button
        aria-expanded={detailsOpen}
        className="mt-1 h-auto justify-self-start px-0 text-[length:var(--conversation-caption-font-size)] text-muted-foreground"
        onClick={() => setDetailsOpen(open => !open)}
        size="sm"
        variant="ghost"
      >
        {detailsOpen ? copy.showLess : copy.showMore}
      </Button>
    </div>
  )
}
