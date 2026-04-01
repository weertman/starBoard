import { useState } from 'react'
import { lookupEntity, getEntityImages, type ArchiveEntityResponse, type ImageDescriptor } from '../api/client'
import { ArchiveImageStrip } from '../components/ArchiveImageStrip'

export function LookupScreen({ onSelectArchiveImage }: { onSelectArchiveImage: (image: ImageDescriptor) => void }) {
  const [entityId, setEntityId] = useState('')
  const [result, setResult] = useState<ArchiveEntityResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function doLookup() {
    setLoading(true)
    setError(null)
    try {
      const data = await lookupEntity(entityId, 'gallery')
      setResult(data)
      if (data.image_window.items[0]) onSelectArchiveImage(data.image_window.items[0])
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  async function loadMore() {
    if (!result?.image_window.next_offset) return
    const next = await getEntityImages(result.entity_id, 'gallery', result.image_window.next_offset, 4)
    setResult({
      ...result,
      image_window: {
        ...next,
        items: [...result.image_window.items, ...next.items],
      },
    })
  }

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <h2>Archive Lookup</h2>
      <div style={{ display: 'flex', gap: 8 }}>
        <input value={entityId} onChange={(e) => setEntityId(e.target.value)} placeholder="Enter gallery ID, e.g. anchovy" style={{ flex: 1 }} />
        <button onClick={doLookup} disabled={!entityId || loading}>{loading ? 'Loading…' : 'Look up'}</button>
      </div>
      {error && <div style={{ color: 'crimson' }}>{error}</div>}
      {result && (
        <>
          <div style={{ fontSize: 14 }}>Opened <strong>{result.entity_id}</strong></div>
          <ArchiveImageStrip items={result.image_window.items} onSelect={onSelectArchiveImage} />
          {result.image_window.next_offset != null && <button onClick={loadMore}>Load more</button>}
        </>
      )}
    </div>
  )
}
