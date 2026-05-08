import { Fragment, useEffect, useMemo, useState } from 'react'

import {
  getIdReviewOptions,
  getLocationSites,
  getMetadataSchema,
  submitEntry,
  type LocationSite,
  type SchemaField,
  type SubmissionResponse,
  type IdReviewOption,
} from '../api/client'
import { LocationSiteMap } from '../components/LocationSiteMap'
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
    const entry: HealthCodeEntry = { code, count, plus }
    return entry
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
    if (modern) {
      return { severity: modern[1].toLowerCase(), position: Number(modern[2]) }
    }
    const legacy = /^\(?(\d+)\)?(\*{0,3})(?:\(r\))?$/.exec(trimmed)
    if (legacy) {
      let severity = 'short'
      if (legacy[2].length >= 2) severity = 'tiny'
      if (legacy[2].length === 1) severity = 'small'
      if (trimmed.startsWith('(') && trimmed.endsWith(')')) severity = 'tiny'
      return { severity, position: Number(legacy[1]) }
    }
    return null
  }).filter((entry): entry is ShortArmEntry => Boolean(entry))
}

function serializeShortArmCode(entries: ShortArmEntry[]): string {
  return [...entries]
    .sort((a, b) => a.position - b.position)
    .map((entry) => `${entry.severity}(${entry.position})`)
    .join(', ')
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
          <div key={`health-code-${index}`} style={{ display: 'grid', gap: 8, gridTemplateColumns: 'minmax(220px, 1fr) auto auto auto', alignItems: 'end' }}>
            <label>
              <div>Health code</div>
              <select
                aria-label={`Health code ${index + 1}`}
                value={entry.code}
                onChange={(e) => updateEntry(index, { code: e.target.value, count: 1, plus: false })}
                style={input}
              >
                <option value="">Select health code…</option>
                {field.options.map((candidate) => (
                  <option key={`${field.name}-${candidate.value}`} value={String(candidate.value)}>{candidate.value} — {candidate.label}</option>
                ))}
              </select>
            </label>
            {option?.requires_count && (
              <label>
                <div>Count</div>
                <input
                  aria-label={`Health code ${index + 1} count`}
                  type="number"
                  min={1}
                  max={30}
                  value={entry.count ?? 1}
                  onChange={(e) => updateEntry(index, { count: Number(e.target.value || 1) })}
                  style={{ ...input, width: 86 }}
                />
              </label>
            )}
            {option?.allows_plus && (
              <label style={{ display: 'flex', gap: 6, alignItems: 'center', paddingBottom: 8 }}>
                <input
                  aria-label={`Health code ${index + 1} plus`}
                  type="checkbox"
                  checked={Boolean(entry.plus)}
                  onChange={(e) => updateEntry(index, { plus: e.target.checked })}
                />
                +
              </label>
            )}
            <button
              type="button"
              aria-label={`Remove health code ${index + 1}`}
              onClick={() => update(entries.filter((_, i) => i !== index))}
              style={{ padding: '8px 10px' }}
            >
              ✕
            </button>
            {option?.definition && <div style={{ gridColumn: '1 / -1', color: '#516070', fontSize: 13 }}>{option.definition}</div>}
          </div>
        )
      })}
      <div>
        <button type="button" onClick={() => update([...entries, { code: '', count: 1, plus: false }])} style={{ padding: '8px 12px' }}>
          + Add health code
        </button>
      </div>
    </div>
  )
}

export function SingleEntryPage() {
  const [schema, setSchema] = useState<SchemaField[]>([])
  const [knownSites, setKnownSites] = useState<LocationSite[]>([])
  const [metadata, setMetadata] = useState<Record<string, string>>({})
  const [showNewLocationInput, setShowNewLocationInput] = useState(false)
  const [pickingCoordinates, setPickingCoordinates] = useState(false)
  const [targetType, setTargetType] = useState<'gallery' | 'query'>('query')
  const [targetMode, setTargetMode] = useState<'create' | 'append'>('create')
  const [targetId, setTargetId] = useState('')
  const [targetOptions, setTargetOptions] = useState<IdReviewOption[]>([])
  const [targetOptionsBusy, setTargetOptionsBusy] = useState(false)
  const [targetPickerOpen, setTargetPickerOpen] = useState(false)
  const [encounterDate, setEncounterDate] = useState(new Date().toISOString().slice(0, 10))
  const [encounterSuffix, setEncounterSuffix] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<SubmissionResponse | null>(null)

  useEffect(() => {
    void (async () => {
      try {
        const [schemaResponse, locationSitesResponse] = await Promise.all([
          getMetadataSchema(),
          getLocationSites(),
        ])
        setSchema(schemaResponse.fields)
        setKnownSites(locationSitesResponse.sites)
      } catch (err) {
        setError(String(err))
      }
    })()
  }, [])

  useEffect(() => {
    if (targetType === 'gallery' && targetMode === 'create') {
      setTargetMode('append')
    }
  }, [targetMode, targetType])

  useEffect(() => {
    let cancelled = false
    if (targetMode !== 'append') {
      setTargetOptions([])
      setTargetPickerOpen(false)
      return () => { cancelled = true }
    }
    setTargetOptionsBusy(true)
    void getIdReviewOptions(targetType)
      .then((response) => {
        if (!cancelled) setTargetOptions(response.options)
      })
      .catch(() => {
        if (!cancelled) setTargetOptions([])
      })
      .finally(() => {
        if (!cancelled) setTargetOptionsBusy(false)
      })
    return () => { cancelled = true }
  }, [targetMode, targetType])

  const grouped = useMemo(() => groupFields(schema), [schema])
  const visibleTargetOptions = useMemo(() => {
    const q = targetId.trim().toLowerCase()
    return targetOptions.filter((option) => {
      if (!q) return true
      return [option.entity_id, option.label, option.location, option.last_observation_date, ...Object.values(option.metadata ?? {})].join(' ').toLowerCase().includes(q)
    }).slice(0, 30)
  }, [targetId, targetOptions])
  const hasLocation = Boolean(metadata.location?.trim())
  const canSubmit = Boolean(targetId.trim() && hasLocation && files.length > 0 && !busy)

  function updateMetadata(name: string, value: string) {
    setMetadata((current) => ({ ...current, [name]: value }))
  }

  function updateSavedLocation(value: string) {
    trackActivity({ event_type: 'single_entry.location.changed', workflow: 'single_entry', details: { mode: value === '__new__' ? 'new_location' : 'saved_location', has_value: Boolean(value && value !== '__new__') } })
    if (value === '__new__') {
      setShowNewLocationInput(true)
      setMetadata((current) => ({ ...current, location: '' }))
      return
    }
    setShowNewLocationInput(false)
    setMetadata((current) => ({ ...current, location: value }))
  }

  async function handleSubmit() {
    if (!canSubmit) return
    setBusy(true)
    setError(null)
    setResult(null)
    const startedAt = Date.now()
    trackActivity({ event_type: 'single_entry.submit.started', workflow: 'single_entry', entity_type: targetType, entity_id: targetId.trim(), details: { target_mode: targetMode, file_count: files.length, has_location: hasLocation } })
    try {
      const response = await submitEntry({
        target_type: targetType,
        target_mode: targetMode,
        target_id: targetId.trim(),
        encounter_date: encounterDate,
        encounter_suffix: encounterSuffix.trim(),
        metadata,
        files,
      })
      setResult(response)
      trackActivity({ event_type: 'single_entry.submit.completed', workflow: 'single_entry', entity_type: response.entity_type, entity_id: response.entity_id, success: true, duration_ms: Date.now() - startedAt, details: { accepted_images: response.accepted_images, skipped_images: response.skipped_images, target_mode: targetMode } })
    } catch (err) {
      trackActivity({ event_type: 'single_entry.submit.completed', workflow: 'single_entry', entity_type: targetType, entity_id: targetId.trim(), success: false, duration_ms: Date.now() - startedAt, details: { target_mode: targetMode, file_count: files.length } })
      setError(String(err))
    } finally {
      setBusy(false)
    }
  }

  function setShortArmEntries(fieldName: string, entries: ShortArmEntry[]) {
    updateMetadata(fieldName, serializeShortArmCode(entries))
  }

  function renderHealthCodeField(field: SchemaField) {
    return <HealthCodeField field={field} value={metadata[field.name] ?? ''} onChange={(value) => updateMetadata(field.name, value)} />
  }

  function renderShortArmField(field: SchemaField) {
    const entries = parseShortArmCode(metadata[field.name] ?? '')
    const updateEntry = (index: number, patch: Partial<ShortArmEntry>) => {
      const next = entries.map((entry, i) => i === index ? { ...entry, ...patch } : entry)
      setShortArmEntries(field.name, next)
    }
    return (
      <div style={{ display: 'grid', gap: 8 }}>
        {entries.map((entry, index) => (
          <div key={`short-arm-${index}`} style={{ display: 'flex', gap: 8, alignItems: 'end', flexWrap: 'wrap' }}>
            <label>
              <div>Arm position</div>
              <input
                aria-label={`Short arm ${index + 1} position`}
                type="number"
                min={1}
                max={25}
                value={entry.position}
                onChange={(e) => updateEntry(index, { position: Number(e.target.value || 1) })}
                style={{ ...input, width: 90 }}
              />
            </label>
            <label>
              <div>Severity</div>
              <select
                aria-label={`Short arm ${index + 1} severity`}
                value={entry.severity}
                onChange={(e) => updateEntry(index, { severity: e.target.value })}
                style={{ ...input, width: 130 }}
              >
                {SHORT_ARM_SEVERITIES.map((severity) => <option key={severity} value={severity}>{severity}</option>)}
              </select>
            </label>
            <button
              type="button"
              aria-label={`Remove short arm ${index + 1}`}
              onClick={() => setShortArmEntries(field.name, entries.filter((_, i) => i !== index))}
              style={{ padding: '8px 10px' }}
            >
              ✕
            </button>
          </div>
        ))}
        <div>
          <button type="button" onClick={() => setShortArmEntries(field.name, [...entries, { position: 1, severity: 'very_tiny' }])} style={{ padding: '8px 12px' }}>
            + Add short arm
          </button>
        </div>
      </div>
    )
  }

  function renderField(field: SchemaField) {
    const value = metadata[field.name] ?? ''
    const commonProps = {
      id: field.name,
      'aria-label': field.display_name,
      title: field.tooltip,
      style: input,
      value,
      onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => updateMetadata(field.name, e.target.value),
    }

    if (field.mobile_widget === 'short_arm_code') {
      return renderShortArmField(field)
    }
    if (field.mobile_widget === 'health_code') {
      return renderHealthCodeField(field)
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
      const listId = `${field.name}-options`
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

  function locationMapUrl() {
    const latitude = metadata.latitude?.trim() ?? ''
    const longitude = metadata.longitude?.trim() ?? ''
    const location = metadata.location?.trim() ?? ''
    if (latitude && longitude) {
      const lat = encodeURIComponent(latitude)
      const lon = encodeURIComponent(longitude)
      return `https://www.openstreetmap.org/export/embed.html?bbox=${lon}%2C${lat}%2C${lon}%2C${lat}&layer=mapnik&marker=${lat}%2C${lon}`
    }
    if (location) {
      return `https://www.openstreetmap.org/export/embed.html?query=${encodeURIComponent(location)}`
    }
    const salishLat = '48.8'
    const salishLon = '-123.0'
    return `https://www.openstreetmap.org/export/embed.html?bbox=-125.5%2C47.0%2C-121.0%2C50.0&layer=mapnik&marker=${salishLat}%2C${salishLon}`
  }

  function handleMapPick(latitude: number, longitude: number) {
    setMetadata((current) => ({
      ...current,
      latitude: latitude.toFixed(6),
      longitude: longitude.toFixed(6),
    }))
    setPickingCoordinates(false)
  }

  return (
    <main style={{ maxWidth: 1180, margin: '0 auto', padding: 18, fontFamily: 'system-ui, sans-serif', color: '#152033', background: '#f7f9fc', minHeight: '100vh' }}>
      <div style={{ display: 'grid', gap: 16 }}>
        <section style={card}>
          <h1 style={{ marginTop: 0 }}>Single Entry</h1>
          <p style={{ marginTop: 0, color: '#516070' }}>Schema-driven browser data entry for creating or appending archive observations.</p>
          <details style={{ margin: '10px 0 14px', padding: 12, borderRadius: 8, background: '#f8fafc', border: '1px solid #d7deea' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 700 }}>How to use Single Entry</summary>
            <div style={{ display: 'grid', gap: 12, marginTop: 12, color: '#405064' }}>
              <section>
                <h3 style={{ margin: '0 0 6px', fontSize: 15, color: '#152033' }}>1. Target and encounter</h3>
                <ol style={{ margin: '0 0 0 20px', padding: 0 }}>
                  <li>Choose Queries or Gallery, choose create/append mode, then enter the target ID.</li>
                  <li>Set the encounter date and optional encounter suffix before submitting.</li>
                </ol>
              </section>
              <section>
                <h3 style={{ margin: '0 0 6px', fontSize: 15, color: '#152033' }}>2. Location and metadata</h3>
                <ol style={{ margin: '0 0 0 20px', padding: 0 }}>
                  <li>Use a saved location or Add new location, then verify latitude/longitude on the map.</li>
                  <li>Fill in the observation metadata fields that apply to this entry.</li>
                </ol>
              </section>
              <section>
                <h3 style={{ margin: '0 0 6px', fontSize: 15, color: '#152033' }}>3. Images and review</h3>
                <ol style={{ margin: '0 0 0 20px', padding: 0 }}>
                  <li>Choose image files from this computer.</li>
                  <li>Review the selected filenames before submitting.</li>
                </ol>
              </section>
              <section>
                <h3 style={{ margin: '0 0 6px', fontSize: 15, color: '#152033' }}>4. Submit entry</h3>
                <ol style={{ margin: '0 0 0 20px', padding: 0 }}>
                  <li>Click Submit entry to archive only after the target, metadata, and selected images look correct.</li>
                </ol>
              </section>
            </div>
          </details>
          <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
            <label>
              <div>Archive</div>
              <select aria-label="Archive" value={targetType} onChange={(e) => setTargetType(e.target.value as typeof targetType)} style={input}>
                <option value="query">Queries</option>
                <option value="gallery">Gallery</option>
              </select>
            </label>
            <label>
              <div>Mode</div>
              <select aria-label="Mode" value={targetMode} onChange={(e) => setTargetMode(e.target.value as typeof targetMode)} style={input}>
                {targetType === 'query' && <option value="create">Create new query</option>}
                <option value="append">Append to existing {targetType}</option>
              </select>
            </label>
            <div>
              <div style={{ marginBottom: 6 }}>Target ID</div>
              <input
                aria-label="Target ID"
                value={targetId}
                onChange={(e) => { setTargetId(e.target.value); if (targetMode === 'append') setTargetPickerOpen(true) }}
                onFocus={() => { if (targetMode === 'append') setTargetPickerOpen(true) }}
                style={input}
              />
              {targetMode === 'append' && targetPickerOpen && (
                <div role="listbox" aria-label="Existing target IDs" style={{ marginTop: 6, maxHeight: 220, overflowY: 'auto', border: '1px solid #d7deea', borderRadius: 8, padding: 6, background: '#f8fafc', display: 'grid', gap: 4 }}>
                  {targetOptionsBusy ? (
                    <div style={{ color: '#516070', padding: 6 }}>Loading existing IDs…</div>
                  ) : visibleTargetOptions.length === 0 ? (
                    <div style={{ color: '#516070', padding: 6 }}>No existing IDs match.</div>
                  ) : visibleTargetOptions.map((option) => (
                    <button
                      key={option.entity_id}
                      type="button"
                      role="option"
                      aria-label={option.label}
                      aria-selected={targetId === option.entity_id}
                      onClick={() => { setTargetId(option.entity_id); setTargetPickerOpen(false) }}
                      style={{ textAlign: 'left', border: targetId === option.entity_id ? '2px solid #2563eb' : '1px solid #d7deea', borderRadius: 8, background: targetId === option.entity_id ? '#eff6ff' : '#fff', padding: 8, cursor: 'pointer' }}
                    >
                      <div style={{ fontWeight: 700 }}>{option.entity_id}</div>
                      <div style={{ color: '#516070', fontSize: 13 }}>{option.location || 'No location'}{option.last_observation_date ? ` · ${option.last_observation_date}` : ''}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
            <label>
              <div>Encounter date</div>
              <input aria-label="Encounter date" type="date" value={encounterDate} onChange={(e) => setEncounterDate(e.target.value)} style={input} />
            </label>
            <label>
              <div>Encounter suffix</div>
              <input aria-label="Encounter suffix" value={encounterSuffix} onChange={(e) => setEncounterSuffix(e.target.value)} style={input} />
            </label>
            <label>
              <div>Upload images from this computer</div>
              <input aria-label="Upload images from this computer" type="file" accept="image/*,.orf,.ORF" multiple onChange={(e) => setFiles(Array.from(e.target.files ?? []))} style={input} />
            </label>
          </div>
        </section>

        {grouped.map((group) => (
          <section key={group.key} style={card}>
            <h2 style={{ marginTop: 0 }}>{group.title}</h2>
            {group.key === 'location' ? (
              <div style={{ display: 'grid', gap: 12 }}>
                <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
                  {group.fields.map((field) => {
                    if (field.name === 'location') {
                      return (
                        <Fragment key="location-controls">
                          <label key="saved-locations">
                            <div>Saved locations</div>
                            <select
                              aria-label="Saved locations"
                              style={input}
                              value={showNewLocationInput ? '__new__' : (metadata.location ?? '')}
                              onChange={(e) => updateSavedLocation(e.target.value)}
                            >
                              <option value="__new__">Add new…</option>
                              <option value="">— choose —</option>
                              {knownSites.map((site) => (
                                <option key={`saved-location-${site.name}`} value={site.name}>{site.name}</option>
                              ))}
                            </select>
                          </label>
                          <div key="location-action" style={{ display: 'flex', alignItems: 'end' }}>
                            <button type="button" onClick={() => setShowNewLocationInput(true)} style={{ padding: '8px 12px' }}>Add new location</button>
                          </div>
                          {showNewLocationInput && (
                            <label key="new-location-input" style={{ gridColumn: '1 / -1' }}>
                              <div>Location</div>
                              {renderField(field)}
                            </label>
                          )}
                        </Fragment>
                      )
                    }
                    return (
                      <label key={field.name}>
                        <div>{field.display_name}</div>
                        {renderField(field)}
                      </label>
                    )
                  })}
                </div>
                <div style={{ position: 'relative' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <div style={{ color: '#516070', fontSize: 13 }}>
                      {pickingCoordinates ? 'Click the map to set coordinates.' : 'Pan/zoom freely. Known sites are shown on the map.'}
                    </div>
                    <button
                      type="button"
                      onClick={() => setPickingCoordinates((current) => !current)}
                      style={{ padding: '8px 12px' }}
                    >
                      {pickingCoordinates ? 'Cancel coordinate pick' : 'Pick coordinates on map'}
                    </button>
                  </div>
                  <LocationSiteMap
                    sites={knownSites}
                    selectedLatitude={metadata.latitude ? Number(metadata.latitude) : undefined}
                    selectedLongitude={metadata.longitude ? Number(metadata.longitude) : undefined}
                    picking={pickingCoordinates}
                    onPick={handleMapPick}
                  />
                </div>
              </div>
            ) : (
              <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
                {group.fields.map((field) => (
                  field.mobile_widget === 'short_arm_code' || field.mobile_widget === 'health_code' ? (
                    <div key={field.name}>
                      <div style={{ marginBottom: 6 }}>{field.display_name}</div>
                      {renderField(field)}
                    </div>
                  ) : (
                    <label key={field.name}>
                      <div>{field.display_name}</div>
                      {renderField(field)}
                    </label>
                  )
                ))}
              </div>
            )}
          </section>
        ))}

        {error && <section style={{ ...card, borderColor: '#e29a9a', background: '#fff5f5', color: '#7a1c1c' }}><b>Error:</b> {error}</section>}

        <section style={card}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <button onClick={() => void handleSubmit()} disabled={!canSubmit} style={{ padding: '8px 12px' }}>
              {busy ? 'Submitting…' : 'Submit entry to archive'}
            </button>
            <span style={{ color: '#516070' }}>{files.length} file(s) selected</span>
          </div>
          {!hasLocation && <div style={{ marginTop: 8, color: '#7a1c1c' }}>Location is required before upload.</div>}
          {files.length > 0 && (
            <div style={{ marginTop: 12, padding: 10, borderRadius: 8, background: '#f8fafc', border: '1px solid #d7deea' }}>
              <h2 style={{ margin: '0 0 6px', fontSize: 16 }}>Review selected image files</h2>
              <div style={{ color: '#516070', fontSize: 13 }}>{files.length} file(s) selected from this computer.</div>
              <ul style={{ margin: '8px 0 0 18px', padding: 0 }}>
                {files.map((file) => (
                  <li key={`${file.name}-${file.size}-${file.lastModified}`}>{file.name}</li>
                ))}
              </ul>
            </div>
          )}
        </section>

        {result && (
          <section style={card}>
            <h2 style={{ marginTop: 0 }}>Result</h2>
            <div style={{ display: 'grid', gap: 6 }}>
              <div><b>Status:</b> {result.status}</div>
              <div><b>Entity:</b> {result.entity_type} / <code>{result.entity_id}</code></div>
              <div><b>Encounter:</b> {result.encounter_folder}</div>
              <div><b>Accepted images:</b> {result.accepted_images}</div>
              <div>{result.message}</div>
            </div>
          </section>
        )}
      </div>
    </main>
  )
}
