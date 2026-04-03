import { useState } from 'react'
import { useSession } from './state/session'
import { useLocalImages } from './state/localImages'
import type { ImageDescriptor } from './api/client'
import { submitObservation } from './api/client'
import { HomeScreen } from './screens/HomeScreen'
import { ObservationWorkspace } from './screens/ObservationWorkspace'
import { LookupWorkspace } from './screens/LookupWorkspace'
import { MetadataSheet, type MetadataDraft } from './components/MetadataSheet'

type Mode = 'home' | 'observation' | 'lookup'

const defaultDraft = (): MetadataDraft => ({
  targetType: 'query',
  targetMode: 'create',
  targetId: '',
  encounterDate: new Date().toISOString().slice(0, 10),
  encounterSuffix: '',
  values: {},
  ready: false,
})

export function App() {
  const { error, loading } = useSession()
  const [mode, setMode] = useState<Mode>('home')
  const { files, previews, addFiles, removeAt, selectedIndex, selectedPreview, select } = useLocalImages()
  const [archiveImages, setArchiveImages] = useState<ImageDescriptor[]>([])
  const [selectedArchiveIndex, setSelectedArchiveIndex] = useState(0)
  const [metadataOpen, setMetadataOpen] = useState(false)
  const [metadataDraft, setMetadataDraft] = useState<MetadataDraft>(defaultDraft())
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [submitMessage, setSubmitMessage] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  function setArchiveSelection(image: ImageDescriptor, loadedItems?: ImageDescriptor[]) {
    if (loadedItems && loadedItems.length > 0) {
      setArchiveImages(loadedItems)
      const idx = loadedItems.findIndex((item) => item.image_id === image.image_id)
      setSelectedArchiveIndex(idx >= 0 ? idx : 0)
      return
    }
    setArchiveImages((prev) => {
      const existingIndex = prev.findIndex((item) => item.image_id === image.image_id)
      if (existingIndex >= 0) {
        setSelectedArchiveIndex(existingIndex)
        return prev
      }
      const next = [...prev, image]
      setSelectedArchiveIndex(next.length - 1)
      return next
    })
  }

  const archiveImage = archiveImages[selectedArchiveIndex] ?? null
  const inferredTargetType: 'query' | 'gallery' = archiveImage ? 'gallery' : 'query'
  const inferredTargetId = metadataDraft.targetId || (archiveImage ? 'anchovy' : '')

  function handleReady(nextDraft: MetadataDraft) {
    setMetadataDraft(nextDraft)
    setSubmitError(null)
    setSubmitMessage(null)
  }

  async function handleSubmit() {
    if (!metadataDraft.ready || files.length === 0) return
    setSubmitting(true)
    setSubmitError(null)
    setSubmitMessage(null)
    try {
      const result = await submitObservation(
        {
          target_type: metadataDraft.targetType,
          target_mode: metadataDraft.targetMode,
          target_id: metadataDraft.targetId,
          encounter_date: metadataDraft.encounterDate,
          encounter_suffix: metadataDraft.encounterSuffix,
          metadata: metadataDraft.values,
        },
        files,
      )
      setSubmitMessage(`${result.message}: ${result.entity_type}/${result.entity_id} ${result.encounter_folder}`)
    } catch (err) {
      setSubmitError(String(err))
    } finally {
      setSubmitting(false)
    }
  }

  const computedSubmitDisabled = submitting || !metadataDraft.ready || files.length === 0
  const metadataSummary = metadataDraft.ready
    ? `Ready for ${metadataDraft.targetType} / ${metadataDraft.targetId || 'missing target'} on ${metadataDraft.encounterDate}`
    : 'Open metadata, fill the required targeting info, then tap Ready.'
  const submitLabel = submitting ? 'Submitting…' : 'Submit to archive'

  if (loading) return <div style={{ padding: 12 }}>Loading…</div>
  if (error) return <div style={{ padding: 12, color: 'crimson' }}>{error}</div>

  return (
    <div style={{ maxWidth: 390, margin: '0 auto', padding: 8, fontFamily: 'sans-serif', display: 'grid', gap: 10, background: '#fafafa', minHeight: '100vh' }}>
      {mode === 'home' && <HomeScreen localCount={files.length} onNewObservation={() => setMode('observation')} onLookup={() => setMode('lookup')} />}
      {mode === 'observation' && (
        <ObservationWorkspace
          localPreviews={previews}
          selectedLocalIndex={selectedIndex}
          selectLocalIndex={select}
          removeLocalAt={removeAt}
          addFiles={addFiles}
          archiveImages={archiveImages}
          selectedArchiveIndex={selectedArchiveIndex}
          selectArchiveIndex={setSelectedArchiveIndex}
          metadataReady={metadataDraft.ready}
          metadataSummary={metadataSummary}
          submitDisabled={computedSubmitDisabled}
          submitLabel={submitLabel}
          submitError={submitError}
          submitMessage={submitMessage}
          onSubmit={() => void handleSubmit()}
          onBack={() => setMode('home')}
          onOpenMetadata={() => setMetadataOpen(true)}
          onOpenLookup={() => setMode('lookup')}
        />
      )}
      {mode === 'lookup' && (
        <LookupWorkspace
          selectedArchiveImage={archiveImage}
          onSelectArchiveImage={setArchiveSelection}
          onBack={() => setMode('home')}
          onOpenMetadata={() => setMetadataOpen(true)}
          canCompare={!!selectedPreview}
        />
      )}

      <MetadataSheet
        open={metadataOpen}
        initialDraft={{
          ...metadataDraft,
          targetType: metadataDraft.targetType || inferredTargetType,
          targetId: metadataDraft.targetId || inferredTargetId,
        }}
        onReady={handleReady}
        onClose={() => setMetadataOpen(false)}
      />
    </div>
  )
}
