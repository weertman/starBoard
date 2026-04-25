import { useEffect, useRef, useState } from 'react'
import { useSession } from './state/session'
import { useLocalImages } from './state/localImages'
import type { ImageDescriptor, MegaStarCapabilityInfo, MegaStarLookupCandidate, MegaStarLookupResponse } from './api/client'
import { lookupMegaStar, submitObservation } from './api/client'
import { initialDraftForMetadataSheet } from './metadataDraftDefaults'
import { HomeScreen } from './screens/HomeScreen'
import { ObservationWorkspace } from './screens/ObservationWorkspace'
import { LookupWorkspace } from './screens/LookupWorkspace'
import { MetadataSheet, type MetadataDraft } from './components/MetadataSheet'

type Mode = 'home' | 'observation' | 'lookup'
type LookupOrigin = 'home' | 'observation'
type LookupRequest = {
  entityType: 'gallery' | 'query'
  entityId: string
  encounter: string
  preferredImageId?: string | null
  nonce: number
}
type MegaStarState = {
  sourceKey: string | null
  loading: boolean
  result: MegaStarLookupResponse | null
  error: string | null
}

const EMPTY_MEGASTAR_STATE: MegaStarState = {
  sourceKey: null,
  loading: false,
  result: null,
  error: null,
}

const defaultDraft = (): MetadataDraft => ({
  targetType: 'query',
  targetMode: 'create',
  targetId: '',
  encounterDate: new Date().toISOString().slice(0, 10),
  encounterSuffix: '',
  values: {},
  ready: false,
})

function getLocalImageKey(file?: File | null) {
  if (!file) return ''
  return [file.name, file.size, file.lastModified, file.type].join(':')
}

export function App() {
  const { error, loading, session } = useSession()
  const { files, previews, addFiles, removeAt, selectedIndex, selectedPreview, select } = useLocalImages()
  const [mode, setMode] = useState<Mode>('home')
  const [lookupOrigin, setLookupOrigin] = useState<LookupOrigin>('home')
  const [lookupRequest, setLookupRequest] = useState<LookupRequest | null>(null)
  const [archiveImages, setArchiveImages] = useState<ImageDescriptor[]>([])
  const [selectedArchiveIndex, setSelectedArchiveIndex] = useState(0)
  const [megastar, setMegaStar] = useState<MegaStarState>(EMPTY_MEGASTAR_STATE)
  const [metadataOpen, setMetadataOpen] = useState(false)
  const [metadataDraft, setMetadataDraft] = useState<MetadataDraft>(defaultDraft())
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [submitMessage, setSubmitMessage] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const megastarRequestIdRef = useRef(0)

  const selectedLocalKey = getLocalImageKey(selectedPreview?.file)
  const megastarInfo: MegaStarCapabilityInfo = session?.megastar_lookup ?? { enabled: false, state: 'disabled', reason: 'session_not_loaded' }
  const megastarEnabled = megastarInfo.enabled

  useEffect(() => {
    if (megastar.sourceKey && megastar.sourceKey !== selectedLocalKey) {
      megastarRequestIdRef.current += 1
      setMegaStar(EMPTY_MEGASTAR_STATE)
    }
  }, [megastar.sourceKey, selectedLocalKey])

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

  function compareMegaStarCandidate(candidate: MegaStarLookupCandidate) {
    const loadedItems = megastar.result?.candidates.map((item) => item.best_match_image) ?? [candidate.best_match_image]
    setArchiveSelection(candidate.best_match_image, loadedItems)
    setMode('observation')
  }

  function openMegaStarCandidateInLookup(candidate: MegaStarLookupCandidate) {
    compareMegaStarCandidate(candidate)
    setLookupOrigin('observation')
    setLookupRequest({
      entityType: candidate.entity_type,
      entityId: candidate.entity_id,
      encounter: candidate.encounter ?? '',
      preferredImageId: candidate.best_match_image.image_id,
      nonce: Date.now(),
    })
    setMode('lookup')
  }

  async function handleMegaStarLookup(maxCandidates = 5) {
    const localFile = selectedPreview?.file
    if (!localFile || !megastarEnabled) return
    const sourceKey = getLocalImageKey(localFile)
    const requestId = megastarRequestIdRef.current + 1
    megastarRequestIdRef.current = requestId
    setMegaStar({ sourceKey, loading: true, result: null, error: null })
    try {
      const result = await lookupMegaStar(localFile, maxCandidates)
      if (megastarRequestIdRef.current !== requestId) return
      setMegaStar({ sourceKey, loading: false, result, error: null })
    } catch (err) {
      if (megastarRequestIdRef.current !== requestId) return
      setMegaStar({ sourceKey, loading: false, result: null, error: String(err) })
    }
  }

  function clearMegaStar() {
    megastarRequestIdRef.current += 1
    setMegaStar(EMPTY_MEGASTAR_STATE)
  }

  const archiveImage = archiveImages[selectedArchiveIndex] ?? null
  const inferredTargetType: 'query' | 'gallery' = archiveImage ? 'gallery' : 'query'
  const inferredTargetId = metadataDraft.targetId || ''

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
      {mode === 'home' && (
        <HomeScreen
          localCount={files.length}
          onNewObservation={() => setMode('observation')}
          onLookup={() => {
            setLookupOrigin('home')
            setMode('lookup')
          }}
        />
      )}
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
          megastarEnabled={megastarEnabled}
          megastarInfo={megastarInfo}
          megastarLoading={megastar.loading}
          megastarResult={megastar.result}
          megastarError={megastar.error}
          onMegaStarLookup={() => void handleMegaStarLookup(5)}
          onClearMegaStar={clearMegaStar}
          onRetryMegaStar={() => void handleMegaStarLookup(5)}
          onLoadMoreMegaStar={(n) => void handleMegaStarLookup(n)}
          onCompareMegaStarCandidate={compareMegaStarCandidate}
          onOpenMegaStarCandidate={openMegaStarCandidateInLookup}
          onSubmit={() => void handleSubmit()}
          onBack={() => setMode('home')}
          onOpenMetadata={() => setMetadataOpen(true)}
          onOpenLookup={() => {
            setLookupOrigin('observation')
            setMode('lookup')
          }}
        />
      )}
      {mode === 'lookup' && (
        <LookupWorkspace
          selectedArchiveImage={archiveImage}
          onSelectArchiveImage={setArchiveSelection}
          onBack={() => setMode(lookupOrigin === 'observation' ? 'observation' : 'home')}
          onOpenMetadata={() => setMetadataOpen(true)}
          canCompare={!!selectedPreview}
          initialRequest={lookupRequest}
        />
      )}

      <MetadataSheet
        open={metadataOpen}
        initialDraft={initialDraftForMetadataSheet(metadataDraft, inferredTargetType, inferredTargetId)}
        onReady={handleReady}
        onClose={() => setMetadataOpen(false)}
      />
    </div>
  )
}
