import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import {
  deleteVllmModel,
  type HFSearchHit,
  searchVllmModels,
  setVllmModel,
  type VllmInventoryModel
} from '@/hermes'
import { useI18n } from '@/i18n'
import { Check, CheckCircle2, Loader2, Search, Trash2 } from '@/lib/icons'
import { $localRuntimeJobs } from '@/store/local-runtime-jobs'
import { notify, notifyError } from '@/store/notifications'

import { ListRow, Pill, SettingsSection } from './primitives'

export function VllmModelsPane({
  models,
  onChanged
}: {
  models: VllmInventoryModel[]
  onChanged: () => void
}) {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const [deleting, setDeleting] = useState<null | string>(null)
  const [setting, setSetting] = useState<null | string>(null)

  async function handleUse(model: VllmInventoryModel) {
    setSetting(model.id)

    try {
      await setVllmModel(model.id)
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

  return (
    <>
      <SettingsSection icon={Search} meta={`${models.length}`} title={copy.modelsTitle}>
        <div className="grid gap-1">
          {models.map(model => {
            const busy = setting === model.id

            return (
              <ListRow
                action={
                  <div className="flex items-center justify-end gap-2">
                    {model.cached && <Pill>{copy.vllmCachedPill}</Pill>}
                    {model.active ? (
                      <Tip label={copy.activeDetail}>
                        <Pill tone="primary">
                          <Check className="mr-1 size-3" />
                          {copy.activePill}
                        </Pill>
                      </Tip>
                    ) : (
                      <Button className={busy ? '[&_svg]:animate-spin' : undefined} disabled={Boolean(setting)} onClick={() => void handleUse(model)} size="sm">
                        {busy ? <Loader2 /> : <Check />}
                        {copy.useAction}
                      </Button>
                    )}
                    {model.cached && (
                      <Tip label={copy.deleteAction}>
                        <Button
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
                description={
                  <>
                    <span className="font-mono text-[0.72rem]">{model.id}</span>
                    {model.added_by_you && <span className="mt-1 block">{copy.addedByYou}</span>}
                    {model.size_label && model.size_label !== '—' && (
                      <span className="mt-1 block text-muted-foreground">{model.size_label}</span>
                    )}
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
      </SettingsSection>
      <VllmBrowseSection onChanged={onChanged} />
    </>
  )
}

function VllmBrowseSection({ onChanged }: { onChanged: () => void }) {
  const { t } = useI18n()
  const copy = t.settings.localModels
  const jobs = useStore($localRuntimeJobs)
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<HFSearchHit[]>([])
  const [searching, setSearching] = useState(false)
  const [setting, setSetting] = useState<null | string>(null)
  const [error, setError] = useState<null | string>(null)
  const searchSeq = useRef(0)

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
            setError(e.message)
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
      setVllmModel(repo)
        .then(() => {
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
        {hits.map(hit => (
          <ListRow
            action={
              <div className="flex items-center gap-2">
                {hit.cached && <Pill>{copy.vllmCachedPill}</Pill>}
                <Button
                  disabled={Boolean(setting) || jobs.some(j => j.status === 'running')}
                  onClick={() => adopt(hit.repo)}
                  size="sm"
                >
                  {setting === hit.repo ? <Loader2 className="animate-spin" /> : <CheckCircle2 />}
                  {copy.useAction}
                </Button>
              </div>
            }
            description={
              <span>
                {Intl.NumberFormat().format(hit.downloads)} {copy.browseDownloads}
                {' · '}
                {Intl.NumberFormat().format(hit.likes)} {copy.browseLikes}
                {hit.gated ? ` · ${copy.browseGated}` : ''}
              </span>
            }
            key={hit.repo}
            title={<span className="font-mono text-[0.8rem]">{hit.repo}</span>}
          />
        ))}
      </div>
    </SettingsSection>
  )
}
