import { useEffect, useMemo, useState } from 'react'
import { lookupEntity, getEntityImages, suggestEntities, type ArchiveEntityResponse, type ImageDescriptor } from '../api/client'
import { ArchiveImageStrip } from '../components/ArchiveImageStrip'
import { ZoomableImagePane } from '../components/ZoomableImagePane'

export function LookupWorkspace({
  selectedArchiveImage,
  onSelectArchiveImage,
  onBack,
  onOpenMetadata,
  canCompare,
}: {
  selectedArchiveImage?: ImageDescriptor | null
  onSelectArchiveImage: (image: ImageDescriptor, loadedItems?: ImageDescriptor[]) => void
  onBack: () => void
  onOpenMetadata: () => void
  canCompare: boolean
}) {
  const [entityId, setEntityId] = useState('')
  const [result, setResult] = useState<ArchiveEntityResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])

  const metadataRows = useMemo(() => Object.entries(result?.metadata_summary ?? {}).filter(([, value]) => String(value ?? '').trim() !== ''), [result])
  const activeArchiveImage = selectedArchiveImage ?? result?.image_window.items[0] ?? null

  useEffect(() => {
    const q = entityId.trim()
    if (!q) {
      setSuggestions([])
      return
    }
    const handle = window.setTimeout(() => {
      suggestEntities('gallery', q, 8).then((data) => setSuggestions(data.items)).catch(() => setSuggestions([]))
    }, 150)
    return () => window.clearTimeout(handle)
  }, [entityId])

  async function doLookup(targetId?: string) {
    const id = (targetId ?? entityId).trim()
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const data = await lookupEntity(id, 'gallery')
      setEntityId(id)
      setResult(data)
      if (data.image_window.items[0]) onSelectArchiveImage(data.image_window.items[0], data.image_window.items)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  async function loadMore() {
    if (result?.image_window.next_offset == null) return
    const next = await getEntityImages(result.entity_id, 'gallery', result.image_window.next_offset, 4)
    const mergedItems = [...result.image_window.items, ...next.items]
    setResult({ ...result, image_window: { ...next, items: mergedItems } })
    if (activeArchiveImage) onSelectArchiveImage(activeArchiveImage, mergedItems)
  }

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
          <button onClick={onBack} style={{ border: '1px solid #ccd6eb', borderRadius: 10, padding: '8px 10px', background: 'white' }}>Home</button>
          <div style={{ fontWeight: 700 }}>Look up star</div>
          <button onClick={onOpenMetadata} style={{ border: '1px solid #ccd6eb', borderRadius: 10, padding: '8px 10px', background: canCompare ? '#eef4ff' : 'white' }}>Metadata</button>
        </div>
        <div style={{ color: '#667085', fontSize: 13 }}>Open a known star first. If you already have local photos loaded, you can compare from the observation workspace.</div>
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <input value={entityId} onChange={(e) => setEntityId(e.target.value)} placeholder="Enter gallery ID, e.g. anchovy" style={{ flex: 1, padding: 10, borderRadius: 10, border: '1px solid #ccd6eb' }} />
        <button onClick={() => void doLookup()} disabled={!entityId || loading}>{loading ? 'Loading…' : 'Open'}</button>
      </div>
      {suggestions.length > 0 && <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>{suggestions.map((item) => <button key={item} onClick={() => void doLookup(item)} style={{ border: '1px solid #ccd6eb', background: 'white', borderRadius: 999, padding: '6px 10px', fontSize: 13 }}>{item}</button>)}</div>}
      {error && <div style={{ color: 'crimson', whiteSpace: 'pre-wrap' }}>{error}</div>}

      {result && (
        <>
          <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 6 }}>
            <div style={{ fontSize: 14 }}>Opened <strong>{result.entity_id}</strong></div>
            <div style={{ fontSize: 13, color: '#555' }}>{result.image_window.total} archive image{result.image_window.total === 1 ? '' : 's'} found • {result.image_window.items.length} loaded</div>
            {metadataRows.length > 0 && <details><summary>Metadata summary</summary><div style={{ display: 'grid', gap: 4, marginTop: 8, fontSize: 13 }}>{metadataRows.slice(0, 12).map(([key, value]) => <div key={key}><strong>{key}</strong>: {String(value)}</div>)}</div></details>}
          </div>
          <ZoomableImagePane title="Archive workspace" subtitle={activeArchiveImage?.label} src={activeArchiveImage?.fullres_url ?? activeArchiveImage?.preview_url} />
          <ArchiveImageStrip items={result.image_window.items} onSelect={(img) => onSelectArchiveImage(img, result.image_window.items)} selectedImageId={activeArchiveImage?.image_id ?? null} />
          {result.image_window.next_offset != null && <button onClick={loadMore}>Load more archive images</button>}
        </>
      )}
    </div>
  )
}
