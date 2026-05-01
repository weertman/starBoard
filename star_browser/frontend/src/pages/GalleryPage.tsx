import { useEffect, useMemo, useRef, useState } from 'react'

import { getIdReviewEntity, getIdReviewOptions, type GalleryEntityResponse, type IdReviewOption, type ImageDescriptor } from '../api/client'

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

type ImageViewState = {
  scale: number
  x: number
  y: number
  rotation: number
}

function InteractiveImageViewer({ image }: { image: ImageDescriptor }) {
  const [view, setView] = useState<ImageViewState>({ scale: 1, x: 0, y: 0, rotation: 0 })
  const rotateKeyDown = useRef(false)
  const dragStart = useRef<{ mode: 'pan' | 'rotate'; x: number; y: number; view: ImageViewState } | null>(null)
  const viewerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    setView({ scale: 1, x: 0, y: 0, rotation: 0 })
  }, [image.image_id])

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key.toLowerCase() === 'r') rotateKeyDown.current = true
    }
    function onKeyUp(e: KeyboardEvent) {
      if (e.key.toLowerCase() === 'r') rotateKeyDown.current = false
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
    }
  }, [])

  function applyWheelZoom(e: WheelEvent) {
    e.preventDefault()
    e.stopPropagation()
    const delta = Math.min(0.75, Math.abs(e.deltaY) / 1000)
    setView((current) => ({
      ...current,
      scale: Number(Math.max(0.2, Math.min(8, current.scale + (e.deltaY < 0 ? delta : -delta))).toFixed(2)),
    }))
  }

  useEffect(() => {
    function onWindowWheel(e: WheelEvent) {
      const viewer = viewerRef.current
      if (!viewer) return
      const rect = viewer.getBoundingClientRect()
      const isInsideViewer = e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom
      if (isInsideViewer || viewer.contains(e.target as Node | null)) applyWheelZoom(e)
    }
    window.addEventListener('wheel', onWindowWheel, { capture: true, passive: false })
    return () => window.removeEventListener('wheel', onWindowWheel, { capture: true })
  }, [])

  function onMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    if (e.button !== 0) return
    e.preventDefault()
    dragStart.current = {
      mode: rotateKeyDown.current ? 'rotate' : 'pan',
      x: e.clientX,
      y: e.clientY,
      view,
    }
    window.addEventListener('mousemove', onWindowMouseMove)
    window.addEventListener('mouseup', onWindowMouseUp)
  }

  function onWindowMouseMove(e: MouseEvent) {
    const start = dragStart.current
    if (!start) return
    const dx = e.clientX - start.x
    const dy = e.clientY - start.y
    if (start.mode === 'rotate') {
      setView({ ...start.view, rotation: Number((start.view.rotation + dx * 0.3).toFixed(1)) })
    } else {
      setView({ ...start.view, x: start.view.x + dx, y: start.view.y + dy })
    }
  }

  function onWindowMouseUp() {
    dragStart.current = null
    window.removeEventListener('mousemove', onWindowMouseMove)
    window.removeEventListener('mouseup', onWindowMouseUp)
  }

  const transform = `translate(${view.x}px, ${view.y}px) rotate(${view.rotation}deg) scale(${view.scale})`

  return (
    <div>
      <div style={{ marginBottom: 8, color: '#516070', fontSize: 13 }}>Wheel to zoom. Drag to pan. Hold R and drag to rotate.</div>
      <div
        ref={viewerRef}
        aria-label="Interactive image viewer"
        onMouseDown={onMouseDown}
        style={{ height: 560, overflow: 'hidden', borderRadius: 10, border: '1px solid #d7deea', background: '#f7f9fc', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: rotateKeyDown.current ? 'crosshair' : 'grab', userSelect: 'none' }}
      >
        <img
          src={image.preview_url}
          alt={image.label}
          draggable={false}
          style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', transform, transformOrigin: 'center center' }}
        />
      </div>
      <div style={{ marginTop: 8 }}><b>{image.label}</b>{image.encounter ? ` — ${image.encounter}` : ''}</div>
      <div style={{ marginTop: 4, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <button type="button" onClick={() => setView({ scale: 1, x: 0, y: 0, rotation: 0 })}>Reset image view</button>
        <a href={image.fullres_url} target="_blank" rel="noreferrer">Open full image</a>
      </div>
    </div>
  )
}

export function GalleryPage() {
  const [archiveType, setArchiveType] = useState<'query' | 'gallery'>('query')
  const [entityId, setEntityId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<GalleryEntityResponse | null>(null)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [encounterFilter, setEncounterFilter] = useState('__all__')
  const [idOptions, setIdOptions] = useState<IdReviewOption[]>([])
  const [optionsBusy, setOptionsBusy] = useState(false)
  const [optionError, setOptionError] = useState<string | null>(null)
  const [idSearch, setIdSearch] = useState('')
  const [locationFilter, setLocationFilter] = useState('__all__')
  const [observedFrom, setObservedFrom] = useState('')
  const [observedTo, setObservedTo] = useState('')

  useEffect(() => {
    let cancelled = false
    setOptionsBusy(true)
    setOptionError(null)
    void getIdReviewOptions(archiveType)
      .then((response) => {
        if (!cancelled) setIdOptions(response.options)
      })
      .catch((err) => {
        if (!cancelled) {
          setOptionError(String(err))
          setIdOptions([])
        }
      })
      .finally(() => {
        if (!cancelled) setOptionsBusy(false)
      })
    return () => { cancelled = true }
  }, [archiveType])

  const locations = useMemo(() => {
    return Array.from(new Set(idOptions.map((option) => option.location).filter(Boolean))).sort()
  }, [idOptions])

  const visibleOptions = useMemo(() => {
    const q = idSearch.trim().toLowerCase()
    return idOptions.filter((option) => {
      if (locationFilter !== '__all__' && option.location !== locationFilter) return false
      if (observedFrom && (!option.last_observation_date || option.last_observation_date < observedFrom)) return false
      if (observedTo && (!option.last_observation_date || option.last_observation_date > observedTo)) return false
      if (!q) return true
      const haystack = [
        option.entity_id,
        option.label,
        option.location,
        option.last_observation_date,
        ...Object.values(option.metadata ?? {}),
      ].join(' ').toLowerCase()
      return haystack.includes(q)
    })
  }, [idOptions, idSearch, locationFilter, observedFrom, observedTo])

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
              <select aria-label="Review ID type" value={archiveType} onChange={(e) => { setArchiveType(e.target.value as 'query' | 'gallery'); setEntityId(''); setResult(null); setError(null); setIdSearch(''); setLocationFilter('__all__'); setObservedFrom(''); setObservedTo('') }} style={input}>
                <option value="query">Query</option>
                <option value="gallery">Gallery</option>
              </select>
            </label>
            <div>
              <div style={{ marginBottom: 6 }}>Selected ID</div>
              <div style={{ padding: '8px 10px', borderRadius: 8, border: '1px solid #c8d0dd', background: '#f8fafc', minHeight: 18 }}>
                {entityId || 'Choose an ID from Available IDs below'}
              </div>
            </div>
            <button onClick={() => void handleLoad()} disabled={busy || !entityId.trim()} style={{ padding: '8px 12px' }}>
              {busy ? 'Loading…' : 'Load ID'}
            </button>
          </div>
          <div style={{ marginTop: 14, display: 'grid', gap: 10 }}>
            <h2 style={{ margin: 0, fontSize: 18 }}>Available IDs</h2>
            <div style={{ display: 'grid', gap: 10, gridTemplateColumns: 'minmax(220px, 1fr) minmax(180px, 220px) minmax(150px, 180px) minmax(150px, 180px)' }}>
              <label>
                <div style={{ marginBottom: 6 }}>Search IDs</div>
                <input aria-label="Search IDs" value={idSearch} onChange={(e) => setIdSearch(e.target.value)} placeholder="Search ID, location, metadata" style={input} />
              </label>
              <label>
                <div style={{ marginBottom: 6 }}>Location filter</div>
                <select aria-label="Location filter" value={locationFilter} onChange={(e) => setLocationFilter(e.target.value)} style={input}>
                  <option value="__all__">All locations</option>
                  {locations.map((location) => <option key={location} value={location}>{location}</option>)}
                </select>
              </label>
              <label>
                <div style={{ marginBottom: 6 }}>Observed from</div>
                <input aria-label="Observed from" type="date" value={observedFrom} onChange={(e) => setObservedFrom(e.target.value)} style={input} />
              </label>
              <label>
                <div style={{ marginBottom: 6 }}>Observed to</div>
                <input aria-label="Observed to" type="date" value={observedTo} onChange={(e) => setObservedTo(e.target.value)} style={input} />
              </label>
            </div>
            <div style={{ color: '#516070', fontSize: 13 }}>
              {optionsBusy ? 'Loading available IDs…' : `${visibleOptions.length} of ${idOptions.length} IDs shown.`}
              {optionError ? ` Could not load available IDs: ${optionError}` : ''}
            </div>
            <div role="listbox" aria-label="Available IDs" style={{ display: 'grid', gap: 6, maxHeight: 260, overflowY: 'auto', border: '1px solid #d7deea', borderRadius: 8, padding: 8, background: '#f8fafc' }}>
              {visibleOptions.length === 0 ? (
                <div style={{ color: '#516070' }}>No IDs match the current filters.</div>
              ) : visibleOptions.map((option) => (
                <button
                  key={option.entity_id}
                  role="option"
                  aria-label={option.label}
                  aria-selected={entityId === option.entity_id}
                  onClick={() => setEntityId(option.entity_id)}
                  style={{ textAlign: 'left', border: entityId === option.entity_id ? '2px solid #2563eb' : '1px solid #d7deea', borderRadius: 8, background: entityId === option.entity_id ? '#eff6ff' : '#fff', padding: 8, cursor: 'pointer' }}
                >
                  <div style={{ fontWeight: 700 }}>{option.label}</div>
                  <div style={{ color: '#516070', fontSize: 13 }}>{option.location || 'No location'}{option.last_observation_date ? ` · ${option.last_observation_date}` : ''}</div>
                </button>
              ))}
            </div>
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
              <h2 style={{ marginTop: 0 }}>Images</h2>
              {selectedImage ? (
                <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'minmax(320px, 2fr) minmax(280px, 1fr)' }}>
                  <div>
                    <InteractiveImageViewer image={selectedImage} />
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
          </>
        )}
      </div>
    </main>
  )
}
