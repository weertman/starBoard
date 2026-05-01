import { useState } from 'react'

import { BatchUploadPage } from './pages/BatchUploadPage'
import { FirstOrderPage } from './pages/FirstOrderPage'
import { GalleryPage } from './pages/GalleryPage'
import { SingleEntryPage } from './pages/SingleEntryPage'

type Tab = 'single-entry' | 'batch' | 'gallery' | 'first-order'

const tabButton = (active: boolean): React.CSSProperties => ({
  padding: '10px 14px',
  borderRadius: 8,
  border: active ? '2px solid #2563eb' : '1px solid #c8d0dd',
  background: active ? '#eff6ff' : '#fff',
  cursor: 'pointer',
  fontWeight: 600,
})

function initialTabFromPath(): Tab {
  if (window.location.pathname === '/batch-upload') return 'batch'
  return 'single-entry'
}

export default function App() {
  const [tab, setTab] = useState<Tab>(initialTabFromPath)

  return (
    <div style={{ background: '#f7f9fc', minHeight: '100vh' }}>
      <div style={{ maxWidth: 1180, margin: '0 auto', padding: '16px 18px 0 18px', fontFamily: 'system-ui, sans-serif' }}>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <button style={tabButton(tab === 'single-entry')} onClick={() => setTab('single-entry')}>Single Entry</button>
          <button style={tabButton(tab === 'batch')} onClick={() => setTab('batch')}>Batch Upload</button>
          <button style={tabButton(tab === 'gallery')} onClick={() => setTab('gallery')}>ID Review</button>
          <button style={tabButton(tab === 'first-order')} onClick={() => setTab('first-order')}>Query Matcher</button>
        </div>
      </div>
      {tab === 'single-entry' && <SingleEntryPage />}
      {tab === 'batch' && <BatchUploadPage />}
      {tab === 'gallery' && <GalleryPage />}
      {tab === 'first-order' && <FirstOrderPage />}
    </div>
  )
}
