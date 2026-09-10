import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import {
  deleteVllmModel,
  downloadVllmModel,
  type HFSearchHit,
  type VllmInventoryModel,
  searchVllmModels,
  useVllm
} from '@/hermes'
import { useI18n } from '@/i18n'
import { Check, CheckCircle2, Cpu, Download, Eye, EyeOff, Loader2, Search, Trash2 } from '@/lib/icons'
import { $localRuntimeJobs, runningDownloadFor, watchLocalRuntimeJobs } from '@/store/local-runtime-jobs'
import { notify, notifyError, readableError } from '@/store/notifications'

import { ListRow, Pill, SettingsSection } from './primitives'

type FitKind = 'fits-gpu' | 'needs-ram' | 'too-big' | 'unknown'

const RELEASED_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] as const

function formatReleasedMonth(raw?: null | string): string | null {
  if (!raw) {
    return null
  }

  const match = raw.trim().match(/^(\d{4})-(\d{2})/)

  if (!match) {
    return null
  }

  const month = RELEASED_MONTHS[Number(match[2]) - 1]

  return month ? `${month} ${match[1]}` : null
}

function downloadBarLabel(
  job: { detail?: string; done_bytes?: number; percent?: number; phase?: string; total_bytes?: number },
  copy: {
    downloadBytesPct: (done: string, total: string, pct: number) => string
    downloadProgress: (done: string, total: string) => string
    downloadVerifying: string
  }
): string {
  if (job.phase === 'verifying') {
    return copy.downloadVerifying
  }

  const doneGb = job.done_bytes ? (job.done_bytes / (1 << 30)).toFixed(1) : '—'
  const totalGb = job.total_bytes ? (job.total_bytes / (1 << 30)).toFixed(1) : '—'

  if (job.total_bytes || job.done_bytes) {
    if (typeof job.percent === 'number') {
      return copy.downloadBytesPct?.(doneGb, `${totalGb} GB`, job.percent) ?? `${doneGb} / ${totalGb} GB · ${job.percent}%`
    }

    return copy.downloadProgress(`${doneGb} GB`, `${totalGb} GB`)
  }

  return job.detail || copy.downloadProgress(`${doneGb} GB`, `${totalGb} GB`)
}

function fitOf(model: { fit?: FitKind; fits?: boolean | null }): FitKind {
  if (model.fit) {
    return model.fit
  }

  if (model.fits === true) {
    return 'fits-gpu'
  }

  if (model.fits === false) {
    return 'too-big'
  }

  return 'unknown'
}

function fitRank(model: VllmInventoryModel): number {
  const fit = fitOf(model)

  if (fit === 'fits-gpu') {
    return 0
  }

  if (fit === 'unknown' || fit === 'needs-ram') {
    return 1
  }

  return 2
}

function hideByDefault(model: VllmInventoryModel): boolean {
  if (model.added_by_you) {
    return false
  }

  if (typeof model.hide_by_default === 'boolean') {
    return model.hide_by_default
  }

  return fitOf(model) === 'too-big'
}

function isServedModel(
  model: { id: string; served_model_name?: string },
  activeModelId?: null | string,
  servedModelName?: null | string
): boolean {
  const keys = [activeModelId, servedModelName].filter((k): k is string => Boolean(k))

  return keys.some(k => k === model.id || k === model.served_model_name)
}

function InUseState({ label }: { label: string }) {
  return (
    <Pill tone="primary">
      <Check className="mr-1 size-3" />
      {label}
    </Pill>
  )
}

function downloadLabel(copy: { downloadAction: (size: string) => string; downloadBare: string }, size?: string) {
  if (size && size !== '—') {
    return copy.downloadAction(size)
  }

  return copy.downloadBare
}

function capabilityLabel(
  cap: string,
  copy: { pillInstruct: string; pillTools: string; pillVision: string }
): string {
  if (cap === 'instruct') {
    return copy.pillInstruct
  }

  if (cap === 'tools') {
    return copy.pillTools
  }

  if (cap === 'vision') {
    return copy.pillVision
  }

  if (cap === 'omni') {
    return 'Omni'
  }

  if (cap === 'moe') {
    return 'MoE'
  }

  if (cap === 'coding') {
    return 'Coding'
  }

  if (cap === '32k' || cap === '64k' || cap === '128k') {
    return cap
  }

  return cap.toUpperCase()
}

function ModelMetaLines({
  createdAt,
  released,
  sizeLabel
}: {
  createdAt?: string
  released: (date: string) => string
  sizeLabel?: string
}) {
  const releasedText = formatReleasedMonth(createdAt)

  return (
    <>
      {releasedText && <span className="mt-1 block text-muted-foreground">{released(releasedText)}</span>}
      {sizeLabel && sizeLabel !== '—' && <span className="mt-1 block text-muted-foreground">{sizeLabel}</span>}
    </>
  )
}

function VllmModelTags({
  cached,
  capabilities,
  copy,
  fit,
  fitDetail,
  recommended
}: {
  cached?: boolean
  capabilities?: string[]
  copy: {
    browseFitUnknown: string
    browseFitUnknownHint: string
    downloaded: string
    pillFitsGpu: string
    pillInstruct: string
    pillTooBig: string
    pillTools: string
    pillUsesRam: string
    pillVision: string
    recommended: string
    vllmCachedPill: string
  }
  fit: FitKind
  fitDetail?: string
  recommended?: boolean
}) {
  return (
    <span className="mt-1.5 flex flex-wrap items-center gap-1.5">
      {fit === 'too-big' ? (
        <Tip label={fitDetail || copy.pillTooBig}>
          <Pill tone="destructive">
            <Cpu className="mr-1 size-3" />
            {copy.pillTooBig}
          </Pill>
        </Tip>
      ) : fit === 'needs-ram' ? (
        <Tip label={fitDetail || copy.pillUsesRam}>
          <Pill tone="warn">
            <Cpu className="mr-1 size-3" />
            {copy.pillUsesRam}
          </Pill>
        </Tip>
      ) : fit === 'fits-gpu' ? (
        <Tip label={fitDetail || copy.pillFitsGpu}>
          <Pill tone="success">
            <Cpu className="mr-1 size-3" />
            {copy.pillFitsGpu}
          </Pill>
        </Tip>
      ) : (
        <Tip label={fitDetail || copy.browseFitUnknownHint}>
          <Pill>
            <Cpu className="mr-1 size-3" />
            {copy.browseFitUnknown}
          </Pill>
        </Tip>
      )}

      {recommended && <Pill tone="primary">{copy.recommended}</Pill>}
      {cached && <Pill>{copy.downloaded}</Pill>}

      {(capabilities ?? []).map(cap => (
        <Pill key={cap}>{capabilityLabel(cap, copy)}</Pill>
      ))}
    </span>
  )
}

export function VllmModelsPane({
  activeModelId,
  models,
  onChanged,
  servedModelName
}: {
  activeModelId?: null | string
  models: VllmInventoryModel[]
  onChanged: () => void
  servedModelName?: null | string
}) {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const jobs = useStore($localRuntimeJobs)
  const [deleting, setDeleting] = useState<null | string>(null)
  const [setting, setSetting] = useState<null | string>(null)
  const [showUnfitting, setShowUnfitting] = useState(false)
  const anyDownloadRunning = jobs.some(j => j.kind === 'model-download' && j.status === 'running')
  const hiddenCount = models.filter(hideByDefault).length
  const visible = models.filter(model => showUnfitting || !hideByDefault(model))

  async function handleUse(model: VllmInventoryModel) {
    setSetting(model.id)

    try {
      const res = await useVllm(model.id)

      if (res.job_id) {
        watchLocalRuntimeJobs()

        return
      }

      notify({
        durationMs: 2_500,
        kind: 'success',
        message: copy.activateDoneToast(model.display_name || model.id),
        title: copy.title
      })
      onChanged()
    } catch (err) {
      notifyError(err, copy.vllmSetFailed)
    } finally {
      setSetting(null)
    }
  }

  async function handleDownload(model: { id: string; display_name?: string }) {
    try {
      const res = await downloadVllmModel(model.id)

      if (res.already_downloaded || !res.job_id) {
        onChanged()

        return
      }

      watchLocalRuntimeJobs()
    } catch (err) {
      notifyError(err, copy.downloadFailed(model.display_name || model.id))
    }
  }

  async function handleDelete(model: VllmInventoryModel) {
    if (!window.confirm(copy.deleteConfirm(model.id))) {
      return
    }

    setDeleting(model.id)

    try {
      await deleteVllmModel(model.id)
      notify({ durationMs: 2_500, kind: 'success', message: copy.deleted(model.id), title: copy.title })
      onChanged()
    } catch (err) {
      notifyError(err, copy.deleteFailed)
    } finally {
      setDeleting(null)
    }
  }

  const sorted = [...visible].sort((a, b) => fitRank(a) - fitRank(b))

  return (
    <>
      {/* Same mark as llama.cpp Local. Search lives in Find more models. */}
      <SettingsSection icon={Download} meta={`${visible.length}`} title={copy.modelsTitle}>
        <div className="grid gap-1">
          {sorted.map(model => {
            const busy = setting === model.id
            const dJob = runningDownloadFor(jobs, model.id)
            const fit = fitOf(model)
            const tooBig = fit === 'too-big'
            const officialTooBig = hideByDefault(model)

            return (
              <ListRow
                className={officialTooBig ? 'opacity-45' : undefined}
                action={
                  <div className="flex items-center justify-end gap-2">
                    {!model.cached ? (
                      dJob ? undefined : officialTooBig ? undefined : (
                        <Button
                          disabled={anyDownloadRunning}
                          onClick={() => void handleDownload(model)}
                          size="sm"
                          variant="outline"
                        >
                          <Download />
                          {downloadLabel(copy, model.size_label)}
                        </Button>
                      )
                    ) : isServedModel(model, activeModelId, servedModelName) ? (
                      <InUseState label={copy.inUsePill} />
                    ) : tooBig ? undefined : (
                      <Button className={busy ? '[&_svg]:animate-spin' : undefined} disabled={Boolean(setting)} onClick={() => void handleUse(model)} size="sm">
                        {busy ? <Loader2 /> : <Check />}
                        {copy.useAction}
                      </Button>
                    )}
                    {model.cached && (
                      <Tip label={copy.deleteAction}>
                        <Button
                          aria-label={copy.deleteAction}
                          className={deleting === model.id ? '[&_svg]:animate-spin' : undefined}
                          onClick={() => void handleDelete(model)}
                          size="icon"
                          variant="ghost"
                        >
                          {deleting === model.id ? <Loader2 /> : <Trash2 />}
                        </Button>
                      </Tip>
                    )}
                  </div>
                }
                below={
                  dJob ? (
                    <div className="mt-2 grid gap-1">
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
                        <div
                          className="h-full rounded-full bg-primary transition-[width] duration-300"
                          style={{ width: `${Math.max(2, Math.min(100, dJob.percent ?? 2))}%` }}
                        />
                      </div>
                      <p className="text-[0.68rem] text-muted-foreground">{downloadBarLabel(dJob, copy)}</p>
                    </div>
                  ) : undefined
                }
                description={
                  <>
                    <span className="font-mono text-[0.72rem]">{model.id}</span>
                    {model.added_by_you && <span className="mt-1 block">{copy.addedByYou}</span>}
                    <ModelMetaLines createdAt={model.created_at} released={copy.released} sizeLabel={model.size_label} />
                    <VllmModelTags
                      cached={model.cached}
                      capabilities={model.capabilities}
                      copy={copy}
                      fit={fit}
                      fitDetail={model.fit_detail}
                    />
                  </>
                }
                key={model.id}
                title={
                  <span className="inline-flex items-center gap-2">
                    {model.display_name || model.served_model_name || model.id}
                    {model.recommended && <Pill tone="primary">{copy.recommended}</Pill>}
                  </span>
                }
              />
            )
          })}
        </div>
        {hiddenCount > 0 && (
          <Button className="mt-2" onClick={() => setShowUnfitting(v => !v)} size="sm" variant="ghost">
            {showUnfitting ? <EyeOff /> : <Eye />}
            {showUnfitting ? copy.hideUnfitting : copy.showUnfitting}
          </Button>
        )}
      </SettingsSection>
      <VllmBrowseSection activeModelId={activeModelId} onChanged={onChanged} servedModelName={servedModelName} />
    </>
  )
}

function VllmBrowseSection({
  activeModelId,
  onChanged,
  servedModelName
}: {
  activeModelId?: null | string
  onChanged: () => void
  servedModelName?: null | string
}) {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const jobs = useStore($localRuntimeJobs)
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<HFSearchHit[]>([])
  const [searching, setSearching] = useState(false)
  const [setting, setSetting] = useState<null | string>(null)
  const [error, setError] = useState<null | string>(null)
  const searchSeq = useRef(0)
  const anyDownloadRunning = jobs.some(j => j.kind === 'model-download' && j.status === 'running')

  useEffect(() => {
    const q = query.trim()

    if (q.length < 2) {
      setHits([])
      setSearching(false)

      return
    }

    const seq = ++searchSeq.current
    setSearching(true)

    const handle = setTimeout(() => {
      searchVllmModels(q)
        .then(r => {
          if (searchSeq.current === seq) {
            setHits(r.hits)
            setError(null)
          }
        })
        .catch((e: Error) => {
          if (searchSeq.current === seq) {
            setError(readableError(e, copy.browseTitle).message)
          }
        })
        .finally(() => {
          if (searchSeq.current === seq) {
            setSearching(false)
          }
        })
    }, 350)

    return () => clearTimeout(handle)
  }, [query])

  const adopt = useCallback(
    (repo: string) => {
      setSetting(repo)
      useVllm(repo)
        .then(res => {
          if (res.job_id) {
            watchLocalRuntimeJobs()

            return
          }

          notify({
            durationMs: 3_000,
            kind: 'success',
            message: copy.activateDoneToast(repo),
            title: copy.browseTitle
          })
          onChanged()
        })
        .catch((e: Error) => notifyError(e, copy.vllmSetFailed))
        .finally(() => setSetting(null))
    },
    [copy.activateDoneToast, copy.browseTitle, copy.vllmSetFailed, onChanged]
  )

  const startDownload = useCallback(
    (repo: string) => {
      downloadVllmModel(repo)
        .then(r => {
          if (r.already_downloaded) {
            notify({ durationMs: 3_000, kind: 'info', message: copy.browseAlreadyDownloaded, title: copy.browseTitle })
            onChanged()

            return
          }

          watchLocalRuntimeJobs()
          notify({
            durationMs: 3_000,
            kind: 'info',
            message: copy.browseDownloadStarted.replace('{name}', repo),
            title: copy.browseTitle
          })
        })
        .catch((e: Error) => notifyError(e, copy.browseTitle))
    },
    [copy.browseAlreadyDownloaded, copy.browseDownloadStarted, copy.browseTitle, onChanged]
  )

  return (
    <SettingsSection icon={Search} title={copy.browseTitle}>
      <p className="text-[0.75rem] text-muted-foreground">{copy.vllmBrowseHint}</p>

      <div className="relative">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
        <input
          className="w-full rounded-md border border-(--ui-border) bg-transparent py-1.5 pl-8 pr-3 text-[0.8rem] outline-none placeholder:text-muted-foreground focus:border-primary"
          onChange={e => setQuery(e.target.value)}
          placeholder={copy.browsePlaceholder}
          value={query}
        />
      </div>

      {searching && (
        <p className="flex items-center gap-2 text-[0.75rem] text-muted-foreground">
          <Loader2 className="size-3 animate-spin" />
          {copy.browseSearching}
        </p>
      )}

      {error && <p className="text-[0.75rem] text-destructive">{error}</p>}

      {!searching && query.trim().length >= 2 && hits.length === 0 && !error && (
        <p className="text-[0.75rem] text-muted-foreground">{copy.vllmBrowseNoHits}</p>
      )}

      <div className="grid gap-1">
        {hits.map(hit => {
          const dJob = runningDownloadFor(jobs, hit.repo)
          const fit = hit.fit ?? 'unknown'
          const tooBig = fit === 'too-big'
          const gated = Boolean(hit.gated)
          const inUse = isServedModel({ id: hit.repo, served_model_name: hit.repo }, activeModelId, servedModelName)

          return (
            <ListRow
              action={
                <div className="flex items-center gap-2">
                  {hit.cached ? (
                    tooBig ? undefined : inUse ? (
                      <InUseState label={copy.inUsePill} />
                    ) : (
                      <Button
                        disabled={Boolean(setting) || jobs.some(j => j.status === 'running')}
                        onClick={() => adopt(hit.repo)}
                        size="sm"
                      >
                        {setting === hit.repo ? <Loader2 className="animate-spin" /> : <CheckCircle2 />}
                        {copy.useAction}
                      </Button>
                    )
                  ) : dJob ? undefined : gated ? (
                    <Tip label={copy.browseGatedHint}>
                      <span className="inline-flex">
                        <Button disabled size="sm" variant="outline">
                          <Download />
                          {copy.downloadBare}
                        </Button>
                      </span>
                    </Tip>
                  ) : (
                    <Button
                      disabled={anyDownloadRunning}
                      onClick={() => startDownload(hit.repo)}
                      size="sm"
                      variant="outline"
                    >
                      <Download />
                      {copy.downloadBare}
                    </Button>
                  )}
                </div>
              }
              below={
                dJob ? (
                  <div className="mt-2 grid gap-1">
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
                      <div
                        className="h-full rounded-full bg-primary transition-[width] duration-300"
                        style={{ width: `${Math.max(2, Math.min(100, dJob.percent ?? 2))}%` }}
                      />
                    </div>
                    <p className="text-[0.68rem] text-muted-foreground">{downloadBarLabel(dJob, copy)}</p>
                  </div>
                ) : undefined
              }
              description={
                <>
                  <span>
                    {Intl.NumberFormat().format(hit.downloads)} {copy.browseDownloads}
                    {' · '}
                    {Intl.NumberFormat().format(hit.likes)} {copy.browseLikes}
                    {hit.gated ? ` · ${copy.browseGated}` : ''}
                  </span>
                  <ModelMetaLines createdAt={hit.created_at} released={copy.released} sizeLabel={hit.size_label} />
                  <VllmModelTags
                    cached={hit.cached}
                    capabilities={hit.capabilities}
                    copy={copy}
                    fit={fit}
                    fitDetail={hit.fit_detail}
                    recommended={hit.recommended}
                  />
                </>
              }
              key={hit.repo}
              title={<span className="font-mono text-[0.8rem]">{hit.repo}</span>}
            />
          )
        })}
      </div>
    </SettingsSection>
  )
}
