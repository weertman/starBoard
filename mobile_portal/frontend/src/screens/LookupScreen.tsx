import { useMemo, useState } from 'react'
import { lookupEntity, getEntityImages, type ArchiveEntityResponse, type ImageDescriptor } from '../api/client'
import { ArchiveImageStrip } from '../components/ArchiveImageStrip'

export function LookupScreen({ selectedArchiveImage, onSelectArchiveImage }: { selectedArchiveImage?: ImageDescriptor | null; onSelectArchiveImage: (image: ImageDescriptor) => void }) {
  const [entityId, setEntityId] = useState('')
  const [result, setResult] = useState<ArchiveEntityResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const metadataRows = useMemo(() => Object.entries(result?.metadata_summary ?? {}).filter(([, value]) => String(value ?? '').trim() !== ''), [result])

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
    if (result?.image_window.next_offset == null) return
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
      <div style={{ color: '#555', fontSize: 14 }}>Pull up a known individual without uploading anything. Images load a few at a time, and full resolution is available when you compare.</div>
      <div style={{ display: 'flex', gap: 8 }}>
        <input value={entityId} onChange={(e) => setEntityId(e.target.value)} placeholder="Enter gallery ID, e.g. anchovy" style={{ flex: 1 }} />
        <button onClick={doLookup} disabled={!entityId || loading}>{loading ? 'Loading…' : 'Look up'}</button>
      </div>
      {error && <div style={{ color: 'crimson', whiteSpace: 'pre-wrap' }}>{error}</div>}
      {result && (
        <>
          <div style={{ padding: 12, border: '1px solid #ddd', borderRadius: 10, background: 'white', display: 'grid', gap: 6 }}>
            <div style={{ fontSize: 14 }}>Opened <strong>{result.entity_id}</strong></div>
            <div style={{ fontSize: 13, color: '#555' }}>{result.image_window.total} archive image{result.image_window.total === 1 ? '' : 's'} found</div>
            {metadataRows.length > 0 && (
              <details>
                <summary>Metadata summary</summary>
                <div style={{ display: 'grid', gap: 4, marginTop: 8, fontSize: 13 }}>
                  {metadataRows.slice(0, 12).map(([key, value]) => <div key={key}><strong>{key}</strong>: {String(value)}</div>)}
                </div>
              </details>
            )}
          </div>
          <ArchiveImageStrip items={result.image_window.items} onSelect={onSelectArchiveImage} selectedImageId={selectedArchiveImage?.image_id ?? null} />
          {result.image_window.next_offset != null && <button onClick={loadMore}>Load more archive images</button>}
        </>
      )}
    </div>
  )
}
