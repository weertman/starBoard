import { useEffect, useMemo, useRef, useState } from 'react'

import {
  getFirstOrderMedia,
  getFirstOrderQueries,
  runFirstOrderSearch,
  type FirstOrderMediaImage,
  type FirstOrderMediaResponse,
  type FirstOrderPreset,
  type FirstOrderQueryOption,
  type FirstOrderSearchResponse,
} from '../api/client'

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

const stateLabels: Record<FirstOrderQueryOption['state'], string> = {
  not_attempted: 'Not attempted',
  pinned: 'Pinned',
  attempted: 'Attempted',
  matched: 'Matched',
}

const stateColors: Record<FirstOrderQueryOption['state'], React.CSSProperties> = {
  not_attempted: { background: '#eef3fb', color: '#344154' },
  pinned: { background: '#ffe0b2', color: '#6b3d00' },
  attempted: { background: '#fff1a8', color: '#5c4a00' },
  matched: { background: '#cfeecf', color: '#215621' },
}

const qualitySymbols = [
  ['madreporite_visibility', '●'],
  ['anus_visibility', '○'],
  ['postural_visibility', '★'],
] as const

const stateOrder: FirstOrderQueryOption['state'][] = ['not_attempted', 'pinned', 'attempted', 'matched']

type StateFilter = Record<FirstOrderQueryOption['state'], boolean>
type RankOrder = 'date_time' | 'existing_easy_match'
type ImageViewState = {
  scale: number
  x: number
  y: number
  rotation: number
}

function defaultStateFilter(): StateFilter {
  return { not_attempted: false, pinned: false, attempted: false, matched: false }
}

function hasAnyQuality(option: FirstOrderQueryOption): boolean {
  return qualitySymbols.some(([field]) => option.quality[field] != null)
}

function optionDateValue(option: FirstOrderQueryOption): string {
  return option.last_observation_date ?? ''
}

function inDateRange(option: FirstOrderQueryOption, from: string, through: string): boolean {
  const value = optionDateValue(option)
  if (!value) return !from && !through
  if (from && value < from) return false
  if (through && value > through) return false
  return true
}

function sortOptions(options: FirstOrderQueryOption[], rankOrder: RankOrder): FirstOrderQueryOption[] {
  return [...options].sort((a, b) => {
    if (rankOrder === 'existing_easy_match') {
      const byEasyMatch = (b.easy_match_score ?? 0) - (a.easy_match_score ?? 0)
      if (byEasyMatch !== 0) return byEasyMatch
    }
    const byDate = optionDateValue(b).localeCompare(optionDateValue(a))
    if (byDate !== 0) return byDate
    return a.query_id.localeCompare(b.query_id)
  })
}

function qualityColor(value: number | null | undefined): string {
  if (value == null) return '#9aa5b1'
  if (value < 0.34) return '#c94141'
  if (value < 0.67) return '#c49419'
  return '#2f8a3b'
}

function stateBadge(state: FirstOrderQueryOption['state']) {
  return (
    <span style={{ ...stateColors[state], borderRadius: 999, padding: '2px 8px', fontSize: 12, whiteSpace: 'nowrap' }}>
      {stateLabels[state]}
    </span>
  )
}

function optionLabel(option: FirstOrderQueryOption): string {
  const date = option.last_observation_date ? ` — ${option.last_observation_date}` : ''
  return `${option.query_id}${date} — ${stateLabels[option.state]}`
}

function QueryMatcherImagePane({ image, alt, ariaLabel }: { image: FirstOrderMediaImage; alt: string; ariaLabel: string }) {
  const [view, setView] = useState<ImageViewState>({ scale: 1, x: 0, y: 0, rotation: 0 })
  const rotateKeyDown = useRef(false)
  const viewerRef = useRef<HTMLDivElement | null>(null)
  const dragStart = useRef<{ mode: 'pan' | 'rotate'; x: number; y: number; view: ImageViewState } | null>(null)

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
      const targetIsInsideViewer = e.target instanceof Node && viewer.contains(e.target)
      if (isInsideViewer || targetIsInsideViewer) applyWheelZoom(e)
    }
    window.addEventListener('wheel', onWindowWheel, { capture: true, passive: false })
    return () => window.removeEventListener('wheel', onWindowWheel, { capture: true })
  }, [])

  function onMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    if (e.button !== 0) return
    e.preventDefault()
    dragStart.current = { mode: rotateKeyDown.current ? 'rotate' : 'pan', x: e.clientX, y: e.clientY, view }
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
        aria-label={ariaLabel}
        onMouseDown={onMouseDown}
        style={{ height: 640, overflow: 'hidden', borderRadius: 10, border: '1px solid #d7deea', background: '#eef2f7', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: rotateKeyDown.current ? 'crosshair' : 'grab', userSelect: 'none' }}
      >
        <img
          src={image.preview_url}
          alt={alt}
          draggable={false}
          style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', transform, transformOrigin: 'center center' }}
        />
      </div>
      <div style={{ marginTop: 8, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <button type="button" onClick={() => setView({ scale: 1, x: 0, y: 0, rotation: 0 })}>Reset image view</button>
      </div>
    </div>
  )
}

export function FirstOrderPage() {
  const [queryId, setQueryId] = useState('')
  const [queryOptions, setQueryOptions] = useState<FirstOrderQueryOption[]>([])
  const [queryFilter, setQueryFilter] = useState('')
  const [stateFilter, setStateFilter] = useState<StateFilter>(() => defaultStateFilter())
  const [withQualityOnly, setWithQualityOnly] = useState(false)
  const [observedFrom, setObservedFrom] = useState('')
  const [observedThrough, setObservedThrough] = useState('')
  const [locationFilter, setLocationFilter] = useState('')
  const [rankOrder, setRankOrder] = useState<RankOrder>('date_time')
  const [loadingQueries, setLoadingQueries] = useState(false)
  const [topK, setTopK] = useState(10)
  const [preset, setPreset] = useState<FirstOrderPreset>('all')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<FirstOrderSearchResponse | null>(null)
  const [queryMedia, setQueryMedia] = useState<FirstOrderMediaResponse | null>(null)
  const [candidateMedia, setCandidateMedia] = useState<Record<string, FirstOrderMediaResponse>>({})
  const [activeCandidateIndex, setActiveCandidateIndex] = useState(0)
  const [activeQueryImageIndex, setActiveQueryImageIndex] = useState(0)
  const [activeCandidateImageIndexes, setActiveCandidateImageIndexes] = useState<Record<string, number>>({})

  async function refreshQueries(preferredQueryId = queryId) {
    setLoadingQueries(true)
    setError(null)
    try {
      const next = await getFirstOrderQueries()
      setQueryOptions(next.queries)
      if (preferredQueryId && next.queries.some((option) => option.query_id === preferredQueryId)) {
        setQueryId(preferredQueryId)
        setQueryFilter(preferredQueryId)
      } else if (!queryId && next.queries.length > 0) {
        setQueryId(next.queries[0].query_id)
        setQueryFilter(next.queries[0].query_id)
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setLoadingQueries(false)
    }
  }

  useEffect(() => {
    void refreshQueries('')
    // Initial load only. refreshQueries intentionally reads current state for manual refreshes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const id = queryId.trim()
    if (!id) {
      setQueryMedia(null)
      setActiveQueryImageIndex(0)
      return
    }
    let cancelled = false
    getFirstOrderMedia('query', id)
      .then((media) => {
        if (!cancelled) {
          setQueryMedia(media)
          setActiveQueryImageIndex(0)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setQueryMedia(null)
          setActiveQueryImageIndex(0)
        }
      })
    return () => {
      cancelled = true
    }
  }, [queryId])

  const locationOptions = useMemo(() => {
    return Array.from(new Set(queryOptions.map((option) => option.last_location?.trim()).filter(Boolean) as string[])).sort((a, b) => a.localeCompare(b))
  }, [queryOptions])

  const filteredOptions = useMemo(() => {
    const needle = queryFilter.trim().toLowerCase()
    const exactKnownQuery = queryOptions.some((option) => option.query_id === queryFilter.trim())
    const textFiltered = !needle || exactKnownQuery
      ? queryOptions
      : queryOptions.filter((option) => option.query_id.toLowerCase().includes(needle))
    const activeStateFilter = stateOrder.some((state) => stateFilter[state])
    return sortOptions(
      textFiltered.filter((option) => (
        (!activeStateFilter || stateFilter[option.state])
        && (!withQualityOnly || hasAnyQuality(option))
        && inDateRange(option, observedFrom, observedThrough)
        && (!locationFilter || option.last_location === locationFilter)
      )),
      rankOrder,
    )
  }, [queryOptions, queryFilter, stateFilter, withQualityOnly, observedFrom, observedThrough, locationFilter, rankOrder])

  const selectedOption = filteredOptions.find((option) => option.query_id === queryId) ?? null

  function selectQuery(nextQueryId: string) {
    setQueryId(nextQueryId)
    setQueryFilter(nextQueryId)
  }

  function stepQuery(delta: number) {
    if (filteredOptions.length === 0) return
    const currentIndex = filteredOptions.findIndex((option) => option.query_id === queryId)
    const idx = currentIndex >= 0 ? currentIndex : 0
    const nextIdx = Math.min(filteredOptions.length - 1, Math.max(0, idx + delta))
    selectQuery(filteredOptions[nextIdx].query_id)
  }

  function handleQueryInput(value: string) {
    setQueryFilter(value)
    if (queryOptions.some((option) => option.query_id === value)) {
      setQueryId(value)
    } else {
      setQueryId(value.trim())
    }
  }

  function toggleStateFilter(state: FirstOrderQueryOption['state']) {
    setStateFilter((current) => ({ ...current, [state]: !current[state] }))
  }

  function clearQueryFilters() {
    setQueryFilter(queryId)
    setStateFilter(defaultStateFilter())
    setWithQualityOnly(false)
    setObservedFrom('')
    setObservedThrough('')
    setLocationFilter('')
  }

  async function handleSearch() {
    const searchQueryId = queryOptions.some((option) => option.query_id === queryId)
      ? queryId
      : filteredOptions.length === 1
        ? filteredOptions[0].query_id
        : queryId
    if (!searchQueryId.trim()) return
    setBusy(true)
    setError(null)
    try {
      const request = { query_id: searchQueryId.trim(), top_k: topK, preset, ...(preset === 'megastar' && activeQueryImage ? { query_image_id: activeQueryImage.image_id } : {}) }
      const next = await runFirstOrderSearch(request)
      setResult(next)
      setActiveCandidateIndex(0)
      setActiveCandidateImageIndexes({})
      setCandidateMedia({})
      void Promise.all(next.candidates.map(async (candidate) => {
        try {
          const media = await getFirstOrderMedia('gallery', candidate.entity_id)
          setCandidateMedia((current) => ({ ...current, [candidate.entity_id]: media }))
          if (candidate.preferred_image_id) {
            const preferredIndex = media.images.findIndex((image) => image.image_id === candidate.preferred_image_id)
            if (preferredIndex >= 0) {
              setActiveCandidateImageIndexes((current) => ({ ...current, [candidate.entity_id]: preferredIndex }))
            }
          }
        } catch {
          setCandidateMedia((current) => ({ ...current, [candidate.entity_id]: { target_type: 'gallery', entity_id: candidate.entity_id, images: [] } }))
        }
      }))
    } catch (err) {
      setError(String(err))
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  const activeCandidate = result?.candidates[activeCandidateIndex] ?? null
  const activeCandidateMedia = activeCandidate ? candidateMedia[activeCandidate.entity_id] ?? null : null
  const activeCandidateImageIndex = activeCandidate ? activeCandidateImageIndexes[activeCandidate.entity_id] ?? 0 : 0
  const activeCandidateImage = activeCandidateMedia?.images[activeCandidateImageIndex] ?? null
  const activeQueryImage = queryMedia?.images[activeQueryImageIndex] ?? null

  function stepProposal(delta: number) {
    if (!result?.candidates.length) return
    setActiveCandidateIndex((current) => Math.min(result.candidates.length - 1, Math.max(0, current + delta)))
  }

  function stepQueryImage(delta: number) {
    if (!queryMedia?.images.length) return
    setActiveQueryImageIndex((current) => Math.min(queryMedia.images.length - 1, Math.max(0, current + delta)))
  }

  function stepCandidateImage(delta: number) {
    if (!activeCandidate || !activeCandidateMedia?.images.length) return
    setActiveCandidateImageIndexes((current) => {
      const currentIndex = current[activeCandidate.entity_id] ?? 0
      return {
        ...current,
        [activeCandidate.entity_id]: Math.min(activeCandidateMedia.images.length - 1, Math.max(0, currentIndex + delta)),
      }
    })
  }

  return (
    <main style={{ maxWidth: 1180, margin: '0 auto', padding: 18, fontFamily: 'system-ui, sans-serif', color: '#152033', background: '#f7f9fc', minHeight: '100vh' }}>
      <div style={{ display: 'grid', gap: 16 }}>
        <section style={card}>
          <h1 style={{ marginTop: 0 }}>Query Matcher</h1>
          <p style={{ marginTop: 0, color: '#516070' }}>Run query-to-gallery matching with desktop-style query selection: type-to-search, workflow state, quality indicators, refresh, and previous/next navigation.</p>
          <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'minmax(280px, 1fr) 32px 32px 140px 180px auto auto', alignItems: 'start' }}>
            <div>
              <label style={{ display: 'block', fontWeight: 600, marginBottom: 4 }} htmlFor="first-order-query">Query</label>
              <input
                id="first-order-query"
                aria-label="Query"
                value={queryFilter}
                onChange={(e) => handleQueryInput(e.target.value)}
                placeholder={loadingQueries ? 'Loading queries…' : 'Type to search query IDs'}
                list="first-order-query-options"
                style={input}
              />
              <datalist id="first-order-query-options">
                {filteredOptions.map((option) => (
                  <option key={option.query_id} value={option.query_id} label={optionLabel(option)} />
                ))}
              </datalist>
              <div style={{ marginTop: 8, display: 'grid', gap: 8 }}>
                <fieldset style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: '6px 8px' }}>
                  <legend style={{ color: '#516070', fontSize: 12, padding: '0 4px' }}>Workflow state filters</legend>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {stateOrder.map((state) => (
                      <label key={state} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 13 }}>
                        <input type="checkbox" checked={stateFilter[state]} onChange={() => toggleStateFilter(state)} />
                        {stateLabels[state]}
                      </label>
                    ))}
                  </div>
                </fieldset>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8 }}>
                  <label style={{ display: 'grid', gap: 3, fontSize: 13 }}>
                    Observed from
                    <input aria-label="Observed from" type="date" value={observedFrom} onChange={(event) => setObservedFrom(event.target.value)} style={input} />
                  </label>
                  <label style={{ display: 'grid', gap: 3, fontSize: 13 }}>
                    Observed through
                    <input aria-label="Observed through" type="date" value={observedThrough} onChange={(event) => setObservedThrough(event.target.value)} style={input} />
                  </label>
                  <label style={{ display: 'grid', gap: 3, fontSize: 13 }}>
                    Last location
                    <select aria-label="Last location" value={locationFilter} onChange={(event) => setLocationFilter(event.target.value)} style={input}>
                      <option value="">All locations</option>
                      {locationOptions.map((location) => <option key={location} value={location}>{location}</option>)}
                    </select>
                  </label>
                  <label style={{ display: 'grid', gap: 3, fontSize: 13 }}>
                    Rank order
                    <select aria-label="Rank order" value={rankOrder} onChange={(event) => setRankOrder(event.target.value as RankOrder)} style={input}>
                      <option value="date_time">Date/time newest first</option>
                      <option value="existing_easy_match">Existing easy match first</option>
                    </select>
                  </label>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
                  <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 13 }}>
                    <input type="checkbox" checked={withQualityOnly} onChange={(event) => setWithQualityOnly(event.target.checked)} />
                    With any quality marker
                  </label>
                  <button type="button" onClick={clearQueryFilters} style={{ padding: '4px 8px' }}>Clear filters</button>
                </div>
              </div>
              <div style={{ marginTop: 6, maxHeight: 360, overflow: 'auto', border: '1px solid #e2e8f0', borderRadius: 8 }}>
                {filteredOptions.map((option) => (
                  <button
                    key={option.query_id}
                    type="button"
                    onClick={() => selectQuery(option.query_id)}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '6px 8px',
                      border: 0,
                      borderTop: '1px solid #eef2f7',
                      background: option.query_id === queryId ? '#e8f1ff' : '#fff',
                      cursor: 'pointer',
                      display: 'flex',
                      justifyContent: 'space-between',
                      gap: 8,
                    }}
                  >
                    <span>
                      <code>{option.query_id}</code>
                      {option.last_observation_date ? <span style={{ color: '#667085' }}> · {option.last_observation_date}</span> : null}
                      {option.last_location ? <span style={{ color: '#667085' }}> · {option.last_location}</span> : null}
                    </span>
                    <span style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
                      {qualitySymbols.map(([field, symbol]) => (
                        <span key={field} title={field} style={{ color: qualityColor(option.quality[field]), fontWeight: 700 }}>{symbol}</span>
                      ))}
                      {stateBadge(option.state)}
                    </span>
                  </button>
                ))}
                {filteredOptions.length === 0 && <div style={{ padding: 8, color: '#667085' }}>No matching queries.</div>}
              </div>
              <div style={{ color: '#667085', fontSize: 12, marginTop: 4 }}>
                {filteredOptions.length} of {queryOptions.length} queries shown. Matched queries are sorted to the bottom.
              </div>
            </div>
            <button type="button" onClick={() => stepQuery(-1)} disabled={queryOptions.length === 0} title="Previous query in list" style={{ marginTop: 24, padding: '8px 0' }}>◀</button>
            <button type="button" onClick={() => stepQuery(1)} disabled={queryOptions.length === 0} title="Next query in list" style={{ marginTop: 24, padding: '8px 0' }}>▶</button>
            <div>
              <label style={{ display: 'block', fontWeight: 600, marginBottom: 4 }}>Top-K</label>
              <input aria-label="Top-K" type="number" min={1} max={500} value={topK} onChange={(e) => setTopK(Number(e.target.value) || 10)} style={input} />
            </div>
            <div>
              <label style={{ display: 'block', fontWeight: 600, marginBottom: 4 }}>Preset</label>
              <select aria-label="Preset" value={preset} onChange={(e) => setPreset(e.target.value as typeof preset)} style={input}>
                <option value="all">All fields</option>
                <option value="colors">Colors</option>
                <option value="text">Text</option>
                <option value="arms_patterns">Arms + patterns</option>
                <option value="megastar">MegaStar</option>
              </select>
            </div>
            <button onClick={() => void refreshQueries(queryId)} disabled={loadingQueries} style={{ marginTop: 24, padding: '8px 12px' }}>
              {loadingQueries ? 'Refreshing…' : 'Refresh'}
            </button>
            <button onClick={() => void handleSearch()} disabled={busy || !queryId.trim()} style={{ marginTop: 24, padding: '8px 12px' }}>
              {busy ? 'Searching…' : 'Search'}
            </button>
          </div>
          {selectedOption && (
            <div style={{ marginTop: 12, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', color: '#516070' }}>
              <span>Selected: <code>{selectedOption.query_id}</code></span>
              {stateBadge(selectedOption.state)}
              {selectedOption.last_observation_date && <span>Last observed: {selectedOption.last_observation_date}</span>}
              {selectedOption.last_location && <span>Last location: {selectedOption.last_location}</span>}
              <span title="Quality indicators: madreporite, anus, posture">
                Quality: {qualitySymbols.map(([field, symbol]) => <span key={field} style={{ color: qualityColor(selectedOption.quality[field]), fontWeight: 700, marginLeft: 4 }}>{symbol}</span>)}
              </span>
            </div>
          )}
          <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: 'minmax(220px, 360px) 1fr', gap: 12, alignItems: 'start' }}>
            <div style={{ border: '1px solid #e2e8f0', borderRadius: 10, padding: 10, background: '#fbfdff' }}>
              <h2 style={{ marginTop: 0, marginBottom: 8, fontSize: 16 }}>Selected query image</h2>
              {activeQueryImage ? (
                <>
                  <a href={activeQueryImage.fullres_url} target="_blank" rel="noreferrer" style={{ color: 'inherit', textDecoration: 'none' }}>
                    <img
                      src={activeQueryImage.preview_url}
                      alt={`Selected query ${queryId} image ${activeQueryImage.label}`}
                      loading="lazy"
                      style={{ display: 'block', width: '100%', maxHeight: 260, objectFit: 'contain', borderRadius: 8, background: '#eef2f7' }}
                    />
                  </a>
                  <div style={{ marginTop: 8, display: 'flex', justifyContent: 'center', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                    <button type="button" onClick={() => stepQueryImage(-1)} disabled={activeQueryImageIndex === 0}>Previous selected query image</button>
                    <b>Selected query image {activeQueryImageIndex + 1} of {queryMedia?.images.length ?? 0}</b>
                    <button type="button" onClick={() => stepQueryImage(1)} disabled={activeQueryImageIndex >= (queryMedia?.images.length ?? 0) - 1}>Next selected query image</button>
                  </div>
                </>
              ) : (
                <div style={{ color: '#667085', padding: 16, background: '#eef2f7', borderRadius: 8 }}>No query image loaded.</div>
              )}
            </div>
          </div>
        </section>

        {error && <section style={{ ...card, borderColor: '#e29a9a', background: '#fff5f5', color: '#7a1c1c' }}><b>Error:</b> {error}</section>}

        {result && (
          <section style={card} aria-label="Query Matcher side-by-side comparison">
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
              <h2 style={{ marginTop: 0, marginBottom: 0 }}>Results for {result.query_id}</h2>
              <div style={{ color: '#516070' }}>Preset: <b>{result.preset}</b></div>
            </div>
            {result.candidates.length === 0 || !activeCandidate ? (
              <div style={{ color: '#516070', marginTop: 12 }}>No ranked candidates returned.</div>
            ) : (
              <div style={{ display: 'grid', gap: 12, marginTop: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'center', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                  <button type="button" onClick={() => stepProposal(-1)} disabled={activeCandidateIndex === 0} style={{ padding: '8px 12px' }}>Previous proposal</button>
                  <b>Proposal {activeCandidateIndex + 1} of {result.candidates.length}</b>
                  <button type="button" onClick={() => stepProposal(1)} disabled={activeCandidateIndex >= result.candidates.length - 1} style={{ padding: '8px 12px' }}>Next proposal</button>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(280px, 1fr))', gap: 14, alignItems: 'stretch' }}>
                  <section aria-label="Query image comparison panel" style={{ border: '1px solid #d7deea', borderRadius: 12, background: '#fbfdff', padding: 12, display: 'grid', gap: 10 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'baseline' }}>
                      <h3 style={{ margin: 0 }}>Query</h3>
                      <code>{result.query_id}</code>
                    </div>
                    {activeQueryImage ? (
                      <>
                        <QueryMatcherImagePane
                          image={activeQueryImage}
                          alt={`Selected query ${result.query_id} image ${activeQueryImage.label}`}
                          ariaLabel="Query matcher query image viewer"
                        />
                        <div style={{ display: 'flex', justifyContent: 'center', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                          <button type="button" onClick={() => stepQueryImage(-1)} disabled={activeQueryImageIndex === 0}>Previous query image</button>
                          <b>Query image {activeQueryImageIndex + 1} of {queryMedia?.images.length ?? 0}</b>
                          <button type="button" onClick={() => stepQueryImage(1)} disabled={activeQueryImageIndex >= (queryMedia?.images.length ?? 0) - 1}>Next query image</button>
                        </div>
                      </>
                    ) : (
                      <div style={{ color: '#667085', padding: 16, textAlign: 'center' }}>No query image loaded.</div>
                    )}
                  </section>
                  <section aria-label="Active proposal comparison panel" style={{ border: '1px solid #d7deea', borderRadius: 12, background: '#fff', padding: 12, display: 'grid', gap: 10 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'baseline' }}>
                      <h3 style={{ margin: 0 }}>Proposal</h3>
                      <span style={{ color: '#516070' }}>Score {activeCandidate.score.toFixed(4)}</span>
                    </div>
                    {activeCandidateImage ? (
                      <>
                        <QueryMatcherImagePane
                          image={activeCandidateImage}
                          alt={`Rank ${activeCandidateIndex + 1} ${activeCandidate.entity_id} image ${activeCandidateImage.label}`}
                          ariaLabel="Query matcher proposal image viewer"
                        />
                        <div style={{ display: 'flex', justifyContent: 'center', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                          <button type="button" onClick={() => stepCandidateImage(-1)} disabled={activeCandidateImageIndex === 0}>Previous proposal image</button>
                          <b>Proposal image {activeCandidateImageIndex + 1} of {activeCandidateMedia?.images.length ?? 0}</b>
                          <button type="button" onClick={() => stepCandidateImage(1)} disabled={activeCandidateImageIndex >= (activeCandidateMedia?.images.length ?? 0) - 1}>Next proposal image</button>
                        </div>
                      </>
                    ) : (
                      <div style={{ color: '#667085', padding: 16, textAlign: 'center' }}>Image loads when media is available.</div>
                    )}
                    <div style={{ display: 'grid', gap: 8 }}>
                      <code style={{ fontWeight: 700, fontSize: 16 }}>{activeCandidate.entity_id}</code>
                      {activeCandidateImage?.encounter && <span style={{ color: '#b26a00', fontSize: 12 }}>Encounter: {activeCandidateImage.encounter}</span>}
                      <div style={{ color: '#516070', fontSize: 13 }}>Contributing fields: {activeCandidate.k_contrib}</div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {Object.entries(activeCandidate.field_breakdown).length === 0 ? (
                          <span style={{ color: '#667085' }}>No field breakdown</span>
                        ) : Object.entries(activeCandidate.field_breakdown).map(([field, value]) => (
                          <span key={field} style={{ background: '#e8f1ff', border: '1px solid #c7d7ef', borderRadius: 999, padding: '2px 7px', fontSize: 12 }}>
                            {field}: {value.toFixed(3)}
                          </span>
                        ))}
                      </div>
                    </div>
                  </section>
                </div>
              </div>
            )}
          </section>
        )}
      </div>
    </main>
  )
}
