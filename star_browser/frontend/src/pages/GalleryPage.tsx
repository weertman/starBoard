import { useMemo, useState } from 'react'

import { getIdReviewEntity, type GalleryEntityResponse } from '../api/client'

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
  const [archiveType, setArchiveType] = useState<'query' | 'gallery'>('query')
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
      const next = await getIdReviewEntity(archiveType, entityId.trim())
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
          <h1 style={{ marginTop: 0 }}>ID Review</h1>
          <p style={{ marginTop: 0, color: '#516070' }}>Inspect one query or gallery ID, filter by encounter, and browse images in a stronger review layout.</p>
          <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'minmax(180px, 220px) minmax(260px, 1fr) auto', alignItems: 'end' }}>
            <label>
              <div style={{ marginBottom: 6 }}>Review ID type</div>
              <select aria-label="Review ID type" value={archiveType} onChange={(e) => { setArchiveType(e.target.value as 'query' | 'gallery'); setResult(null); setError(null) }} style={input}>
                <option value="query">Query</option>
                <option value="gallery">Gallery</option>
              </select>
            </label>
            <label>
              <div style={{ marginBottom: 6 }}>ID</div>
              <input value={entityId} onChange={(e) => setEntityId(e.target.value)} placeholder="Enter query or gallery ID" style={input} />
            </label>
            <button onClick={() => void handleLoad()} disabled={busy || !entityId.trim()} style={{ padding: '8px 12px' }}>
              {busy ? 'Loading…' : 'Load ID'}
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
                  <select aria-label="Encounter filter" value={encounterFilter} onChange={(e) => onChangeEncounterFilter(e.target.value)} style={input}>
                    <option value="__all__">All encounters</option>
                    {result.encounters.map((enc) => (
                      <option key={enc.encounter} value={enc.encounter}>{enc.label}</option>
                    ))}
                  </select>
                </label>
              </div>
            </section>

            <section style={card}>
              <h2 style={{ marginTop: 0 }}>Metadata</h2>
              <h3 style={{ margin: '0 0 8px', fontSize: 16 }}>Latest metadata</h3>
              <div style={{ display: 'grid', gap: 6, gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
                {Object.entries(result.metadata_summary).length === 0 ? (
                  <div style={{ color: '#516070' }}>No metadata summary available.</div>
                ) : Object.entries(result.metadata_summary).map(([k, v]) => (
                  <div key={k}><b>{k}:</b> {v}</div>
                ))}
              </div>
              <h3 style={{ margin: '16px 0 8px', fontSize: 16 }}>All metadata rows</h3>
              <div style={{ display: 'grid', gap: 10 }}>
                {(result.metadata_rows ?? []).length === 0 ? (
                  <div style={{ color: '#516070' }}>No metadata rows available.</div>
                ) : (result.metadata_rows ?? []).map((row) => (
                  <div key={`${row.source}-${row.row_index}`} style={{ border: '1px solid #e1e7f0', borderRadius: 8, padding: 10 }}>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>Row {row.row_index} · {row.source}</div>
                    <div style={{ display: 'grid', gap: 4, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
                      {Object.entries(row.values).map(([k, v]) => <div key={k}><b>{k}:</b> {v}</div>)}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section style={card}>
              <h2 style={{ marginTop: 0 }}>Timeline</h2>
              <div style={{ display: 'grid', gap: 10 }}>
                {(result.timeline ?? []).length === 0 ? (
                  <div style={{ color: '#516070' }}>No timeline events available.</div>
                ) : (result.timeline ?? []).map((event) => (
                  <div key={event.encounter || event.label} style={{ borderLeft: '4px solid #2563eb', paddingLeft: 10 }}>
                    <div style={{ fontWeight: 700 }}>{event.date || 'Unknown date'}</div>
                    <div>{event.label}</div>
                    <div style={{ color: '#516070' }}>{event.image_count} {event.image_count === 1 ? 'image' : 'images'}</div>
                    {event.image_labels.length > 0 && <div style={{ color: '#516070', fontSize: 13 }}>{event.image_labels.join(', ')}</div>}
                  </div>
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
