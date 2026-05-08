import { Fragment, useEffect, useMemo, useRef, useState } from 'react'

import { getIdReviewEntity, getIdReviewOptions, getLocationSites, getMetadataSchema, renameIdReviewEntity, setIdReviewFirstImage, updateIdReviewMetadata, type GalleryEntityResponse, type IdReviewOption, type ImageDescriptor, type LocationSite, type SchemaField } from '../api/client'
import { trackActivity } from '../activity'

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

function groupFields(fields: SchemaField[]) {
  const order: string[] = []
  const groups = new Map<string, { title: string; fields: SchemaField[] }>()
  for (const field of fields) {
    if (!groups.has(field.group)) {
      order.push(field.group)
      groups.set(field.group, { title: field.group_display_name, fields: [] })
    }
    groups.get(field.group)!.fields.push(field)
  }
  const priority: Record<string, number> = { location: 0, health: 1 }
  order.sort((a, b) => (priority[a] ?? 2) - (priority[b] ?? 2))
  return order.map((key) => ({ key, ...groups.get(key)! }))
}

const SHORT_ARM_SEVERITIES = ['very_tiny', 'tiny', 'small', 'short']

type ShortArmEntry = {
  position: number
  severity: string
}

type HealthCodeEntry = {
  code: string
  count?: number
  plus?: boolean
}

function healthOption(field: SchemaField, code: string) {
  return field.options.find((option) => String(option.value) === code)
}

function parseHealthCodes(value: string, field: SchemaField): HealthCodeEntry[] {
  if (!value.trim()) return []
  const knownCodes = field.options.map((option) => String(option.value)).sort((a, b) => b.length - a.length)
  return value.split(',').map((part) => {
    const compact = part.trim().toUpperCase().replace(/\s+/g, '')
    const code = knownCodes.find((candidate) => compact.startsWith(candidate))
    if (!code) return null
    const option = healthOption(field, code)
    const rest = compact.slice(code.length)
    const countMatch = /^\((\d+)\+?\)/.exec(rest)
    const count = countMatch && option?.requires_count ? Number(countMatch[1]) : undefined
    const plus = Boolean(option?.allows_plus && (rest.includes('+)') || rest.endsWith('+')))
    return { code, count, plus }
  }).filter((entry): entry is HealthCodeEntry => Boolean(entry))
}

function serializeHealthCodes(entries: HealthCodeEntry[], field: SchemaField): string {
  return entries.filter((entry) => entry.code).map((entry) => {
    const option = healthOption(field, entry.code)
    let out = entry.code
    if (option?.requires_count) out += `(${Math.max(1, entry.count ?? 1)})`
    if (option?.allows_plus && entry.plus) out += '+'
    return out
  }).join(', ')
}

function parseShortArmCode(value: string): ShortArmEntry[] {
  if (!value.trim()) return []
  return value.split(',').map((part) => {
    const trimmed = part.trim()
    const modern = /^(very_tiny|tiny|small|short)\((\d+)\)$/i.exec(trimmed)
    if (modern) return { severity: modern[1].toLowerCase(), position: Number(modern[2]) }
    const legacy = /^\(?(\d+)\)?(\*{0,3})(?:\(r\))?$/.exec(trimmed)
    if (!legacy) return null
    let severity = 'short'
    if (legacy[2].length >= 2) severity = 'tiny'
    if (legacy[2].length === 1) severity = 'small'
    if (trimmed.startsWith('(') && trimmed.endsWith(')')) severity = 'tiny'
    return { severity, position: Number(legacy[1]) }
  }).filter((entry): entry is ShortArmEntry => Boolean(entry))
}

function serializeShortArmCode(entries: ShortArmEntry[]): string {
  return [...entries].sort((a, b) => a.position - b.position).map((entry) => `${entry.severity}(${entry.position})`).join(', ')
}

function HealthCodeField({ field, value, onChange }: { field: SchemaField; value: string; onChange: (v: string) => void }) {
  const [entries, setEntries] = useState<HealthCodeEntry[]>(() => parseHealthCodes(value, field))

  useEffect(() => {
    setEntries(parseHealthCodes(value, field))
  }, [field, value])

  function update(next: HealthCodeEntry[]) {
    setEntries(next)
    onChange(serializeHealthCodes(next, field))
  }

  function updateEntry(index: number, patch: Partial<HealthCodeEntry>) {
    update(entries.map((entry, i) => i === index ? { ...entry, ...patch } : entry))
  }

  return (
    <div style={{ display: 'grid', gap: 8 }}>
      {entries.map((entry, index) => {
        const option = healthOption(field, entry.code)
        return (
          <div key={`id-review-health-code-${index}`} style={{ display: 'grid', gap: 8, gridTemplateColumns: 'minmax(220px, 1fr) auto auto auto', alignItems: 'end' }}>
            <label>
              <div>Health code</div>
              <select aria-label={`Health code ${index + 1}`} value={entry.code} onChange={(e) => updateEntry(index, { code: e.target.value, count: 1, plus: false })} style={input}>
                <option value="">Select health code…</option>
                {field.options.map((candidate) => (
                  <option key={`${field.name}-${candidate.value}`} value={String(candidate.value)}>{candidate.value} — {candidate.label}</option>
                ))}
              </select>
            </label>
            {option?.requires_count && (
              <label>
                <div>Count</div>
                <input aria-label={`Health code ${index + 1} count`} type="number" min={1} max={30} value={entry.count ?? 1} onChange={(e) => updateEntry(index, { count: Number(e.target.value || 1) })} style={{ ...input, width: 86 }} />
              </label>
            )}
            {option?.allows_plus && (
              <label style={{ display: 'flex', gap: 6, alignItems: 'center', paddingBottom: 8 }}>
                <input aria-label={`Health code ${index + 1} plus`} type="checkbox" checked={Boolean(entry.plus)} onChange={(e) => updateEntry(index, { plus: e.target.checked })} />
                +
              </label>
            )}
            <button type="button" aria-label={`Remove health code ${index + 1}`} onClick={() => update(entries.filter((_, i) => i !== index))} style={{ padding: '8px 10px' }}>✕</button>
            {option?.definition && <div style={{ gridColumn: '1 / -1', color: '#516070', fontSize: 13 }}>{option.definition}</div>}
          </div>
        )
      })}
      <div>
        <button type="button" onClick={() => update([...entries, { code: '', count: 1, plus: false }])} style={{ padding: '8px 12px' }}>+ Add health code</button>
      </div>
    </div>
  )
}

function renderMetadataField(field: SchemaField, value: string, onChange: (v: string) => void) {
  const commonProps = {
    id: `id-review-metadata-${field.name}`,
    'aria-label': field.display_name,
    title: field.tooltip,
    style: input,
    value,
    onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => onChange(e.target.value),
  }

  if (field.mobile_widget === 'short_arm_code') {
    const entries = parseShortArmCode(value)
    const updateEntries = (next: ShortArmEntry[]) => onChange(serializeShortArmCode(next))
    const updateEntry = (index: number, patch: Partial<ShortArmEntry>) => {
      updateEntries(entries.map((entry, i) => i === index ? { ...entry, ...patch } : entry))
    }
    return (
      <div style={{ display: 'grid', gap: 8 }}>
        {entries.map((entry, index) => (
          <div key={`id-review-short-arm-${index}`} style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
            <label>
              <div>Arm position</div>
              <input aria-label={`Short arm ${index + 1} position`} type="number" min={1} max={25} value={entry.position} onChange={(e) => updateEntry(index, { position: Number(e.target.value || 1) })} style={{ ...input, width: 90 }} />
            </label>
            <label>
              <div>Severity</div>
              <select aria-label={`Short arm ${index + 1} severity`} value={entry.severity} onChange={(e) => updateEntry(index, { severity: e.target.value })} style={{ ...input, width: 130 }}>
                {SHORT_ARM_SEVERITIES.map((severity) => <option key={severity} value={severity}>{severity}</option>)}
              </select>
            </label>
            <button type="button" aria-label={`Remove short arm ${index + 1}`} onClick={() => updateEntries(entries.filter((_, i) => i !== index))} style={{ padding: '8px 10px' }}>✕</button>
          </div>
        ))}
        <div>
          <button type="button" onClick={() => updateEntries([...entries, { position: 1, severity: 'very_tiny' }])} style={{ padding: '8px 12px' }}>+ Add short arm</button>
        </div>
      </div>
    )
  }
  if (field.mobile_widget === 'health_code') {
    return <HealthCodeField field={field} value={value} onChange={onChange} />
  }
  if (field.mobile_widget === 'textarea') {
    return <textarea {...commonProps} rows={3} />
  }
  if ((field.mobile_widget === 'select' || field.mobile_widget === 'color_select') && (field.options.length > 0 || field.vocabulary.length > 0)) {
    const options = field.options.length > 0
      ? field.options.map((option) => ({ label: option.label, value: String(option.value) }))
      : field.vocabulary.map((item) => ({ label: item, value: item }))
    return (
      <select {...commonProps}>
        <option value="">—</option>
        {options.map((option) => (
          <option key={`${field.name}-${option.value}`} value={option.value}>{option.label}</option>
        ))}
      </select>
    )
  }
  if (field.mobile_widget === 'location' && field.vocabulary.length > 0) {
    const listId = `id-review-${field.name}-options`
    return (
      <>
        <input {...commonProps} list={listId} />
        <datalist id={listId}>
          {field.vocabulary.map((item) => <option key={`${listId}-${item}`} value={item} />)}
        </datalist>
      </>
    )
  }
  if (field.mobile_widget === 'number') {
    return <input {...commonProps} type="number" min={field.min_value ?? undefined} max={field.max_value ?? undefined} />
  }
  return <input {...commonProps} />
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
        <a href={image.fullres_url} download={image.label}>Download image</a>
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
  const [knownSites, setKnownSites] = useState<LocationSite[]>([])
  const [observedFrom, setObservedFrom] = useState('')
  const [observedTo, setObservedTo] = useState('')
  const [renameDraft, setRenameDraft] = useState('')
  const [metadataDraft, setMetadataDraft] = useState<Record<string, string>>({})
  const [metadataSchema, setMetadataSchema] = useState<SchemaField[]>([])
  const [showNewLocationInput, setShowNewLocationInput] = useState(false)
  const [editBusy, setEditBusy] = useState(false)
  const [editMessage, setEditMessage] = useState<string | null>(null)
  const [bestImageBusy, setBestImageBusy] = useState(false)

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

  useEffect(() => {
    let cancelled = false
    void Promise.all([getLocationSites(), getMetadataSchema()])
      .then(([locationResponse, schemaResponse]) => {
        if (!cancelled) {
          setKnownSites(locationResponse.sites)
          setMetadataSchema(schemaResponse.fields)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setKnownSites([])
          setMetadataSchema([])
        }
      })
    return () => { cancelled = true }
  }, [])

  const locations = useMemo(() => {
    return knownSites.map((site) => site.name)
  }, [knownSites])

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
  const schemaFieldNames = useMemo(() => new Set(metadataSchema.map((field) => field.name)), [metadataSchema])
  const groupedMetadataSchema = useMemo(() => groupFields(metadataSchema), [metadataSchema])
  const editableMetadataFields = useMemo(() => {
    const keys = new Set<string>()
    if (metadataSchema.length > 0) {
      metadataSchema.forEach((field) => keys.add(field.name))
    }
    if (result) {
      Object.keys(result.metadata_summary ?? {}).forEach((key) => keys.add(key))
      ;(result.metadata_rows ?? []).forEach((row) => Object.keys(row.values ?? {}).forEach((key) => keys.add(key)))
    }
    return Array.from(keys).sort((a, b) => a.localeCompare(b))
  }, [metadataSchema, result])
  const extraMetadataFields = useMemo(() => {
    return editableMetadataFields.filter((key) => !schemaFieldNames.has(key))
  }, [editableMetadataFields, schemaFieldNames])

  function applyLoadedResult(next: GalleryEntityResponse) {
    setResult(next)
    setRenameDraft(next.entity_id)
    setMetadataDraft({ ...(next.metadata_summary ?? {}) })
    setEditMessage(null)
  }

  async function handleLoad() {
    if (!entityId.trim()) return
    setBusy(true)
    setError(null)
    const startedAt = Date.now()
    try {
      const next = await getIdReviewEntity(archiveType, entityId.trim())
      applyLoadedResult(next)
      setEncounterFilter('__all__')
      setSelectedIndex(0)
      trackActivity({ event_type: 'id_review.entity.loaded', workflow: 'id_review', entity_type: archiveType, entity_id: next.entity_id, success: true, duration_ms: Date.now() - startedAt, details: { image_count: next.images.length, metadata_rows: next.metadata_rows.length } })
    } catch (err) {
      trackActivity({ event_type: 'id_review.entity.loaded', workflow: 'id_review', entity_type: archiveType, entity_id: entityId.trim(), success: false, duration_ms: Date.now() - startedAt })
      setError(String(err))
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  function onChangeEncounterFilter(value: string) {
    trackActivity({ event_type: 'id_review.encounter_filter.changed', workflow: 'id_review', entity_type: archiveType, entity_id: result?.entity_id, details: { filter: value === '__all__' ? 'all' : 'encounter' } })
    setEncounterFilter(value)
    setSelectedIndex(0)
  }

  async function handleRename() {
    if (!result || !renameDraft.trim() || renameDraft.trim() === result.entity_id) return
    setEditBusy(true)
    setError(null)
    try {
      const next = await renameIdReviewEntity(archiveType, result.entity_id, renameDraft.trim())
      applyLoadedResult(next)
      setEntityId(next.entity_id)
      setEditMessage(`Renamed ID to ${next.entity_id}.`)
      trackActivity({ event_type: 'id_review.rename.completed', workflow: 'id_review', entity_type: archiveType, entity_id: next.entity_id, success: true, details: { previous_entity_id: result.entity_id } })
      void getIdReviewOptions(archiveType).then((response) => setIdOptions(response.options)).catch(() => undefined)
    } catch (err) {
      trackActivity({ event_type: 'id_review.rename.completed', workflow: 'id_review', entity_type: archiveType, entity_id: result.entity_id, success: false })
      setError(String(err))
    } finally {
      setEditBusy(false)
    }
  }

  async function handleMetadataSave() {
    if (!result) return
    setEditBusy(true)
    setError(null)
    try {
      const next = await updateIdReviewMetadata(archiveType, result.entity_id, metadataDraft)
      applyLoadedResult(next)
      setEditMessage('Metadata saved.')
      trackActivity({ event_type: 'id_review.metadata_save.completed', workflow: 'id_review', entity_type: archiveType, entity_id: result.entity_id, success: true, details: { field_count: Object.keys(metadataDraft).length } })
      void getIdReviewOptions(archiveType).then((response) => setIdOptions(response.options)).catch(() => undefined)
    } catch (err) {
      trackActivity({ event_type: 'id_review.metadata_save.completed', workflow: 'id_review', entity_type: archiveType, entity_id: result.entity_id, success: false, details: { field_count: Object.keys(metadataDraft).length } })
      setError(String(err))
    } finally {
      setEditBusy(false)
    }
  }

  async function handleSetFirstImage() {
    if (!result || !selectedImage) return
    setBestImageBusy(true)
    setError(null)
    try {
      await setIdReviewFirstImage(selectedImage.image_id)
      const next = await getIdReviewEntity(archiveType, result.entity_id)
      applyLoadedResult(next)
      setEncounterFilter('__all__')
      setSelectedIndex(0)
      setEditMessage(`Set first image to ${selectedImage.label}.`)
      trackActivity({ event_type: 'id_review.first_image_set', workflow: 'id_review', entity_type: archiveType, entity_id: result.entity_id, success: true, details: { image_label: selectedImage.label } })
    } catch (err) {
      trackActivity({ event_type: 'id_review.first_image_set', workflow: 'id_review', entity_type: archiveType, entity_id: result.entity_id, success: false })
      setError(String(err))
    } finally {
      setBestImageBusy(false)
    }
  }

  function updateMetadataDraft(key: string, value: string) {
    setMetadataDraft((current) => ({ ...current, [key]: value }))
  }

  function updateSavedLocation(value: string) {
    if (value === '__new__') {
      setShowNewLocationInput(true)
      setMetadataDraft((current) => ({ ...current, location: '' }))
      return
    }
    setShowNewLocationInput(false)
    setMetadataDraft((current) => ({ ...current, location: value }))
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
              <div style={{ marginTop: 14, display: 'grid', gap: 10, gridTemplateColumns: 'minmax(260px, 1fr) auto', alignItems: 'end' }}>
                <label>
                  <div style={{ marginBottom: 6 }}>Rename selected ID</div>
                  <input aria-label="Rename selected ID" value={renameDraft} onChange={(e) => setRenameDraft(e.target.value)} style={input} />
                </label>
                <button type="button" disabled={editBusy || !renameDraft.trim() || renameDraft.trim() === result.entity_id} onClick={() => void handleRename()} style={{ padding: '8px 12px' }}>
                  Save ID name
                </button>
              </div>
              {editMessage && <div style={{ marginTop: 8, color: '#0b6b2b' }}>{editMessage}</div>}
            </section>

            <section style={card}>
              <h2 style={{ marginTop: 0 }}>Images</h2>
              {selectedImage ? (
                <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'minmax(320px, 2fr) minmax(280px, 1fr)' }}>
                  <div>
                    <InteractiveImageViewer image={selectedImage} />
                    <div style={{ marginTop: 8, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                      <button type="button" onClick={() => void handleSetFirstImage()} disabled={bestImageBusy || selectedImage.image_id === result.images[0]?.image_id} style={{ padding: '8px 12px' }}>
                        {bestImageBusy ? 'Setting first image…' : 'Set first image'}
                      </button>
                      <span style={{ color: '#516070', fontSize: 13 }}>The first image is what users see first for this ID.</span>
                    </div>
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
              <h3 style={{ margin: '16px 0 8px', fontSize: 16 }}>Edit metadata</h3>
              {editableMetadataFields.length === 0 ? (
                <div style={{ color: '#516070' }}>No editable metadata fields available.</div>
              ) : (
                <div style={{ display: 'grid', gap: 12 }}>
                  {groupedMetadataSchema.map((group) => (
                    <section key={group.key} style={{ border: '1px solid #e1e7f0', borderRadius: 8, padding: 10, background: '#f8fafc' }}>
                      <h3 style={{ margin: '0 0 8px', fontSize: 16 }}>{group.title}</h3>
                      <div style={{ display: 'grid', gap: 10, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
                        {group.fields.map((field) => {
                          if (field.name === 'location') {
                            return (
                              <Fragment key="id-review-location-controls">
                                <label>
                                  <div style={{ marginBottom: 6 }}>Saved locations</div>
                                  <select
                                    aria-label="Saved locations"
                                    style={input}
                                    value={showNewLocationInput ? '__new__' : (metadataDraft.location ?? '')}
                                    onChange={(e) => updateSavedLocation(e.target.value)}
                                  >
                                    <option value="__new__">Add new…</option>
                                    <option value="">— choose —</option>
                                    {knownSites.map((site) => (
                                      <option key={`id-review-saved-location-${site.name}`} value={site.name}>{site.name}</option>
                                    ))}
                                  </select>
                                </label>
                                <div style={{ display: 'flex', alignItems: 'end' }}>
                                  <button type="button" onClick={() => setShowNewLocationInput(true)} style={{ padding: '8px 12px' }}>Add new location</button>
                                </div>
                                {showNewLocationInput && (
                                  <label style={{ gridColumn: '1 / -1' }}>
                                    <div style={{ marginBottom: 6 }}>{field.display_name}</div>
                                    {renderMetadataField(field, metadataDraft[field.name] ?? '', (value) => updateMetadataDraft(field.name, value))}
                                  </label>
                                )}
                              </Fragment>
                            )
                          }
                          if (field.mobile_widget === 'short_arm_code' || field.mobile_widget === 'health_code') {
                            return (
                              <div key={field.name}>
                                <div style={{ marginBottom: 6 }}>{field.display_name}</div>
                                {renderMetadataField(field, metadataDraft[field.name] ?? '', (value) => updateMetadataDraft(field.name, value))}
                              </div>
                            )
                          }
                          return (
                            <label key={field.name}>
                              <div style={{ marginBottom: 6 }}>{field.display_name}</div>
                              {renderMetadataField(field, metadataDraft[field.name] ?? '', (value) => updateMetadataDraft(field.name, value))}
                            </label>
                          )
                        })}
                      </div>
                    </section>
                  ))}
                  {extraMetadataFields.length > 0 && (
                    <section style={{ border: '1px solid #e1e7f0', borderRadius: 8, padding: 10, background: '#f8fafc' }}>
                      <h3 style={{ margin: '0 0 8px', fontSize: 16 }}>Other metadata</h3>
                      <div style={{ display: 'grid', gap: 10, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
                        {extraMetadataFields.map((key) => (
                          <label key={key}>
                            <div style={{ marginBottom: 6 }}>{key}</div>
                            <input aria-label={`Metadata field ${key}`} value={metadataDraft[key] ?? ''} onChange={(e) => updateMetadataDraft(key, e.target.value)} style={input} />
                          </label>
                        ))}
                      </div>
                    </section>
                  )}
                  {metadataSchema.length === 0 && extraMetadataFields.length === 0 && (
                    <div style={{ display: 'grid', gap: 10, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
                      {editableMetadataFields.map((key) => (
                        <label key={key}>
                          <div style={{ marginBottom: 6 }}>{key}</div>
                          <input aria-label={`Metadata field ${key}`} value={metadataDraft[key] ?? ''} onChange={(e) => updateMetadataDraft(key, e.target.value)} style={input} />
                        </label>
                      ))}
                    </div>
                  )}
                  <div>
                    <button type="button" disabled={editBusy} onClick={() => void handleMetadataSave()} style={{ padding: '8px 12px' }}>Save metadata</button>
                  </div>
                </div>
              )}
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
