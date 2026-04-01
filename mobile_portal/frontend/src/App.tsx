import { useMemo, useState } from 'react'
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
  const { files, previews, addFiles, removeAt } = useLocalImages()
  const [archiveImage, setArchiveImage] = useState<ImageDescriptor | null>(null)
  const localSrc = useMemo(() => previews[0]?.url, [previews])

  if (loading) return <div style={{ padding: 16 }}>Loading…</div>
  if (error) return <div style={{ padding: 16, color: 'crimson' }}>{error}</div>

  return (
    <div style={{ maxWidth: 720, margin: '0 auto', padding: 16, fontFamily: 'sans-serif', display: 'grid', gap: 16 }}>
      <header>
        <h1 style={{ marginBottom: 4 }}>starBoard Mobile Portal</h1>
        <div style={{ color: '#555', fontSize: 14 }}>{session?.authenticated_email}</div>
      </header>
      <nav style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
        {tabs.map((name) => (
          <button key={name} onClick={() => setTab(name)} style={{ padding: 10, background: tab === name ? '#e6f0ff' : 'white' }}>{name}</button>
        ))}
      </nav>
      {tab === 'capture' && <CaptureScreen previews={previews} addFiles={addFiles} removeAt={removeAt} />}
      {tab === 'metadata' && <MetadataScreen files={files} />}
      {tab === 'lookup' && <LookupScreen onSelectArchiveImage={(img) => { setArchiveImage(img); setTab('compare') }} />}
      {tab === 'compare' && <CompareScreen localSrc={localSrc} archiveImage={archiveImage} />}
    </div>
  )
}
