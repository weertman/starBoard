import { useState } from 'react'
import { useSession } from './state/session'
import { useLocalImages } from './state/localImages'
import type { ImageDescriptor } from './api/client'
import { HomeScreen } from './screens/HomeScreen'
import { ObservationWorkspace } from './screens/ObservationWorkspace'
import { LookupWorkspace } from './screens/LookupWorkspace'
import { MetadataSheet } from './components/MetadataSheet'

type Mode = 'home' | 'observation' | 'lookup'

export function App() {
  const { session, error, loading } = useSession()
  const [mode, setMode] = useState<Mode>('home')
  const { files, previews, addFiles, removeAt, selectedIndex, selectedPreview, select } = useLocalImages()
  const [archiveImages, setArchiveImages] = useState<ImageDescriptor[]>([])
  const [selectedArchiveIndex, setSelectedArchiveIndex] = useState(0)
  const [metadataOpen, setMetadataOpen] = useState(false)

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
  const inferredTargetId = archiveImage ? 'anchovy' : ''

  if (loading) return <div style={{ padding: 12 }}>Loading…</div>
  if (error) return <div style={{ padding: 12, color: 'crimson' }}>{error}</div>

  return (
    <div style={{ maxWidth: 430, margin: '0 auto', padding: 10, fontFamily: 'sans-serif', display: 'grid', gap: 12, background: '#fafafa', minHeight: '100vh' }}>
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
        files={files}
        initialTargetType={inferredTargetType}
        initialTargetId={inferredTargetId}
        onClose={() => setMetadataOpen(false)}
      />
    </div>
  )
}
