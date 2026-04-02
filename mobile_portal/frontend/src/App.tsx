import { useState } from 'react'
import { useSession } from './state/session'
import { useLocalImages } from './state/localImages'
import { CaptureScreen } from './screens/CaptureScreen'
import { MetadataScreen } from './screens/MetadataScreen'
import { LookupScreen } from './screens/LookupScreen'
import { CompareScreen } from './screens/CompareScreen'
import type { ImageDescriptor } from './api/client'

const tabs = ['capture', 'metadata', 'lookup', 'compare'] as const

type Tab = typeof tabs[number]

export function App() {
  const { session, error, loading } = useSession()
  const [tab, setTab] = useState<Tab>('capture')
  const { files, previews, addFiles, removeAt, selectedIndex, selectedPreview, select } = useLocalImages()
  const [archiveImages, setArchiveImages] = useState<ImageDescriptor[]>([])
  const [selectedArchiveIndex, setSelectedArchiveIndex] = useState(0)

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

  if (loading) return <div style={{ padding: 16 }}>Loading…</div>
  if (error) return <div style={{ padding: 16, color: 'crimson' }}>{error}</div>

  return (
    <div style={{ maxWidth: 720, margin: '0 auto', padding: 16, fontFamily: 'sans-serif', display: 'grid', gap: 16, background: '#fafafa', minHeight: '100vh' }}>
      <header>
        <h1 style={{ marginBottom: 4 }}>starBoard Mobile Portal</h1>
        <div style={{ color: '#555', fontSize: 14 }}>{session?.authenticated_email}</div>
        <div style={{ color: '#666', fontSize: 13, marginTop: 6 }}>{files.length} local image{files.length === 1 ? '' : 's'} ready • {archiveImage ? `archive image selected: ${archiveImage.label}` : 'no archive image selected'}</div>
      </header>
      <nav style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
        {tabs.map((name) => (
          <button key={name} onClick={() => setTab(name)} style={{ padding: 10, background: tab === name ? '#e6f0ff' : 'white', border: '1px solid #ccd6eb', borderRadius: 8, textTransform: 'capitalize' }}>{name}</button>
        ))}
      </nav>
      {tab === 'capture' && <CaptureScreen previews={previews} addFiles={addFiles} removeAt={removeAt} select={select} selectedIndex={selectedIndex} />}
      {tab === 'metadata' && <MetadataScreen files={files} />}
      {tab === 'lookup' && <LookupScreen selectedArchiveImage={archiveImage} onSelectArchiveImage={(img, loadedItems) => { setArchiveSelection(img, loadedItems); setTab('compare') }} />}
      {tab === 'compare' && <CompareScreen localPreviews={previews} selectedLocalIndex={selectedIndex} selectLocalIndex={select} removeLocalAt={removeAt} archiveImages={archiveImages} selectedArchiveIndex={selectedArchiveIndex} selectArchiveIndex={setSelectedArchiveIndex} />}
      {tab !== 'compare' && selectedPreview && archiveImage && <button onClick={() => setTab('compare')} style={{ position: 'sticky', bottom: 12, padding: 12, borderRadius: 10, border: '1px solid #2f6fed', background: '#2f6fed', color: 'white' }}>Open current comparison</button>}
    </div>
  )
}
