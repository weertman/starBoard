import { useMemo, useState } from 'react'

import { getGalleryEntity, type GalleryEntityResponse } from '../api/client'

const card: React.CSSProperties = {
  background: '#fff',
  border: '1px solid #d7deea',
  borderRadius: 10,
  padding: 12,
  boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
}

const input: React.CSSProperties = {
  width: '100%',
  padding: '8px 10px',
  borderRadius: 8,
  border: '1px solid #c8d0dd',
  boxSizing: 'border-box',
}

export function GalleryPage() {
  const [entityId, setEntityId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<GalleryEntityResponse | null>(null)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [encounterFilter, setEncounterFilter] = useState('__all__')

  const filteredImages = useMemo(() => {
    if (!result) return []
    if (encounterFilter === '__all__') return result.images
    return result.images.filter((image) => image.encounter === encounterFilter)
  }, [result, encounterFilter])

  const selectedImage = filteredImages[selectedIndex] ?? null

  async function handleLoad() {
    if (!entityId.trim()) return
    setBusy(true)
    setError(null)
    try {
      const next = await getGalleryEntity(entityId.trim())
      setResult(next)
      setEncounterFilter('__all__')
      setSelectedIndex(0)
    } catch (err) {
      setError(String(err))
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  function onChangeEncounterFilter(value: string) {
    setEncounterFilter(value)
    setSelectedIndex(0)
  }

  return (
    <main style={{ maxWidth: 1180, margin: '0 auto', padding: 18, fontFamily: 'system-ui, sans-serif', color: '#152033', background: '#f7f9fc', minHeight: '100vh' }}>
      <div style={{ display: 'grid', gap: 16 }}>
        <section style={card}>
          <h1 style={{ marginTop: 0 }}>Gallery Review</h1>
          <p style={{ marginTop: 0, color: '#516070' }}>Inspect one gallery ID, filter by encounter, and browse images in a stronger review layout.</p>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <input value={entityId} onChange={(e) => setEntityId(e.target.value)} placeholder="Enter gallery ID" style={input} />
            <button onClick={() => void handleLoad()} disabled={busy || !entityId.trim()} style={{ padding: '8px 12px' }}>
              {busy ? 'Loading…' : 'Load'}
            </button>
          </div>
        </section>

        {error && <section style={{ ...card, borderColor: '#e29a9a', background: '#fff5f5', color: '#7a1c1c' }}><b>Error:</b> {error}</section>}

        {result && (
          <>
            <section style={card}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'start', flexWrap: 'wrap' }}>
                <div>
                  <h2 style={{ marginTop: 0, marginBottom: 6 }}>{result.entity_id}</h2>
                  <div style={{ color: '#516070' }}>{result.images.length} images across {result.encounters.length} encounters</div>
                </div>
                <label style={{ minWidth: 260 }}>
                  <div style={{ marginBottom: 6 }}>Encounter filter</div>
                  <select value={encounterFilter} onChange={(e) => onChangeEncounterFilter(e.target.value)} style={input}>
                    <option value="__all__">All encounters</option>
                    {result.encounters.map((enc) => (
                      <option key={enc.encounter} value={enc.encounter}>{enc.label}</option>
                    ))}
                  </select>
                </label>
              </div>
            </section>

            <section style={card}>
              <h2 style={{ marginTop: 0 }}>Metadata summary</h2>
              <div style={{ display: 'grid', gap: 6, gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
                {Object.entries(result.metadata_summary).length === 0 ? (
                  <div style={{ color: '#516070' }}>No metadata summary available.</div>
                ) : Object.entries(result.metadata_summary).map(([k, v]) => (
                  <div key={k}><b>{k}:</b> {v}</div>
                ))}
              </div>
            </section>

            <section style={card}>
              <h2 style={{ marginTop: 0 }}>Images</h2>
              {selectedImage ? (
                <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'minmax(320px, 2fr) minmax(280px, 1fr)' }}>
                  <div>
                    <img src={selectedImage.preview_url} alt={selectedImage.label} style={{ width: '100%', maxHeight: 560, borderRadius: 10, border: '1px solid #d7deea', objectFit: 'contain', background: '#f7f9fc' }} />
                    <div style={{ marginTop: 8 }}><b>{selectedImage.label}</b>{selectedImage.encounter ? ` — ${selectedImage.encounter}` : ''}</div>
                    <div style={{ marginTop: 4 }}><a href={selectedImage.fullres_url} target="_blank" rel="noreferrer">Open full image</a></div>
                  </div>
                  <div style={{ display: 'grid', gap: 8, maxHeight: 560, overflowY: 'auto' }}>
                    {filteredImages.map((image, idx) => (
                      <button
                        key={image.image_id}
                        onClick={() => setSelectedIndex(idx)}
                        style={{
                          textAlign: 'left',
                          border: idx === selectedIndex ? '2px solid #2563eb' : '1px solid #d7deea',
                          background: idx === selectedIndex ? '#eff6ff' : '#fff',
                          borderRadius: 8,
                          padding: 8,
                          cursor: 'pointer',
                        }}
                      >
                        <div style={{ fontWeight: 600 }}>{image.label}</div>
                        <div style={{ color: '#516070', fontSize: 13 }}>{image.encounter ?? 'no encounter'}</div>
                        <div style={{ color: '#8091a7', fontSize: 12 }}><code>{image.image_id}</code></div>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div style={{ color: '#516070' }}>No images match the selected encounter filter.</div>
              )}
            </section>
          </>
        )}
      </div>
    </main>
  )
}
