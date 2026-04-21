import { useState } from 'react'

import { BatchUploadPage } from './pages/BatchUploadPage'
import { FirstOrderPage } from './pages/FirstOrderPage'
import { GalleryPage } from './pages/GalleryPage'

type Tab = 'batch' | 'gallery' | 'first-order'

const tabButton = (active: boolean): React.CSSProperties => ({
  padding: '10px 14px',
  borderRadius: 8,
  border: active ? '2px solid #2563eb' : '1px solid #c8d0dd',
  background: active ? '#eff6ff' : '#fff',
  cursor: 'pointer',
  fontWeight: 600,
})

export default function App() {
  const [tab, setTab] = useState<Tab>('batch')

  return (
    <div style={{ background: '#f7f9fc', minHeight: '100vh' }}>
      <div style={{ maxWidth: 1180, margin: '0 auto', padding: '16px 18px 0 18px', fontFamily: 'system-ui, sans-serif' }}>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <button style={tabButton(tab === 'batch')} onClick={() => setTab('batch')}>Batch Upload</button>
          <button style={tabButton(tab === 'gallery')} onClick={() => setTab('gallery')}>Gallery Review</button>
          <button style={tabButton(tab === 'first-order')} onClick={() => setTab('first-order')}>First-order Search</button>
        </div>
      </div>
      {tab === 'batch' && <BatchUploadPage />}
      {tab === 'gallery' && <GalleryPage />}
      {tab === 'first-order' && <FirstOrderPage />}
    </div>
  )
}
