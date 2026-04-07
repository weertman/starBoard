import { useEffect, useMemo, useRef, useState } from 'react'
import { getMetadataSchema, getLookupOptions, type SchemaField } from '../api/client'

export type MetadataDraft = {
  targetType: 'query' | 'gallery'
  targetMode: 'create' | 'append'
  targetId: string
  encounterDate: string
  encounterSuffix: string
  values: Record<string, string>
  ready: boolean
}

// ---- reusable inline styles ----
const inputStyle = { padding: 10, borderRadius: 10, border: '1px solid #ccd6eb', width: '100%', boxSizing: 'border-box' as const }
const pillStyle = { border: '1px solid #ccd6eb', background: 'white', borderRadius: 999, padding: '6px 10px', fontSize: 13, cursor: 'pointer' }
const dropdownWrapStyle: React.CSSProperties = { position: 'relative' }
const dropdownListStyle: React.CSSProperties = {
  position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 50,
  background: 'white', border: '1px solid #ccd6eb', borderRadius: 10,
  maxHeight: 200, overflowY: 'auto', boxShadow: '0 4px 12px rgba(0,0,0,.10)',
}
const dropdownItemStyle: React.CSSProperties = { padding: '8px 12px', cursor: 'pointer', borderBottom: '1px solid #f0f0f0' }

// ---- Short Arm Code editor ----
const SEVERITIES = ['very_tiny', 'tiny', 'small', 'short'] as const
type ShortArmEntry = { position: number; severity: string }

function parseShortArmCode(raw: string): ShortArmEntry[] {
  if (!raw.trim()) return []
  const re = /(very_tiny|tiny|small|short)\s*\(\s*(\d+)\s*\)/g
  const entries: ShortArmEntry[] = []
  let m: RegExpExecArray | null
  while ((m = re.exec(raw)) !== null) entries.push({ severity: m[1], position: parseInt(m[2]) })
  return entries
}
function serializeShortArmCode(entries: ShortArmEntry[]): string {
  return entries.filter(e => e.position > 0).sort((a, b) => a.position - b.position).map(e => `${e.severity}(${e.position})`).join(', ')
}

function ShortArmCodeEditor({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [entries, setEntries] = useState<ShortArmEntry[]>(() => parseShortArmCode(value))
  function update(next: ShortArmEntry[]) { setEntries(next); onChange(serializeShortArmCode(next)) }
  function addEntry() { update([...entries, { position: 1, severity: 'short' }]) }
  function removeEntry(i: number) { update(entries.filter((_, idx) => idx !== i)) }
  function setField(i: number, key: keyof ShortArmEntry, val: string | number) {
    const next = entries.map((e, idx) => idx === i ? { ...e, [key]: val } : e)
    update(next)
  }
  return (
    <div style={{ display: 'grid', gap: 6 }}>
      {entries.map((entry, i) => (
        <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <label style={{ fontSize: 12, whiteSpace: 'nowrap' }}>Arm</label>
          <input type="number" min={1} max={25} value={entry.position} onChange={e => setField(i, 'position', parseInt(e.target.value) || 1)} style={{ ...inputStyle, width: 60 }} />
          <select value={entry.severity} onChange={e => setField(i, 'severity', e.target.value)} style={{ ...inputStyle, width: 'auto' }}>
            {SEVERITIES.map(s => <option key={s} value={s}>{s.replace('_', ' ')}</option>)}
          </select>
          <button onClick={() => removeEntry(i)} style={{ border: 'none', background: 'none', color: 'crimson', cursor: 'pointer', fontSize: 18, padding: '0 4px' }}>×</button>
        </div>
      ))}
      <button onClick={addEntry} style={{ ...pillStyle, alignSelf: 'start', color: '#2f6fed' }}>+ Add short arm</button>
    </div>
  )
}

// ---- Searchable dropdown for colors / locations ----
function SearchableSelect({ value, onChange, options, placeholder }: {
  value: string; onChange: (v: string) => void; options: string[]; placeholder?: string
}) {
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const wrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClick(e: MouseEvent) { if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const filtered = useMemo(() => {
    const q = filter.toLowerCase()
    return q ? options.filter(o => o.toLowerCase().includes(q)) : options
  }, [options, filter])

  return (
    <div ref={wrapRef} style={dropdownWrapStyle}>
      <input
        value={open ? filter : value}
        onFocus={() => { setOpen(true); setFilter(value) }}
        onChange={e => { setFilter(e.target.value); onChange(e.target.value) }}
        placeholder={placeholder}
        style={inputStyle}
      />
      {open && filtered.length > 0 && (
        <div style={dropdownListStyle}>
          {filtered.map(opt => (
            <div key={opt} style={{ ...dropdownItemStyle, fontWeight: opt === value ? 600 : 400 }}
              onClick={() => { onChange(opt); setOpen(false) }}>{opt}</div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---- Target ID dropdown (shows full list, filters on type) ----
function TargetIdPicker({ value, onChange, entityType }: {
  value: string; onChange: (v: string) => void; entityType: 'gallery' | 'query'
}) {
  const [allIds, setAllIds] = useState<string[]>([])
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    getLookupOptions(entityType, '', 500).then(data => setAllIds(data.ids)).catch(() => setAllIds([]))
  }, [entityType])

  useEffect(() => {
    function handleClick(e: MouseEvent) { if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const filtered = useMemo(() => {
    const q = value.trim().toLowerCase()
    return q ? allIds.filter(id => id.toLowerCase().includes(q)) : allIds
  }, [allIds, value])

  return (
    <div ref={wrapRef} style={dropdownWrapStyle}>
      <input
        value={value}
        onFocus={() => setOpen(true)}
        onChange={e => { onChange(e.target.value); setOpen(true) }}
        placeholder={entityType === 'gallery' ? 'search gallery IDs…' : 'search query IDs…'}
        style={inputStyle}
      />
      {open && filtered.length > 0 && (
        <div style={dropdownListStyle}>
          {filtered.slice(0, 30).map(id => (
            <div key={id} style={{ ...dropdownItemStyle, fontWeight: id === value ? 600 : 400 }}
              onClick={() => { onChange(id); setOpen(false) }}>{id}</div>
          ))}
          {filtered.length > 30 && <div style={{ ...dropdownItemStyle, color: '#999', fontStyle: 'italic' }}>{filtered.length - 30} more…</div>}
        </div>
      )}
    </div>
  )
}

// ---- Field renderer ----
function renderFieldInput(field: SchemaField, value: string, setValue: (value: string) => void) {
  // Short arm code: structured editor
  if (field.mobile_widget === 'short_arm_code') {
    return <ShortArmCodeEditor value={value} onChange={setValue} />
  }
  // Color fields: searchable dropdown
  if (field.mobile_widget === 'color_select' && field.vocabulary.length > 0) {
    return <SearchableSelect value={value} onChange={setValue} options={field.vocabulary} placeholder="select or type a color…" />
  }
  // Location name: searchable dropdown
  if (field.mobile_widget === 'location' && field.vocabulary.length > 0) {
    return <SearchableSelect value={value} onChange={setValue} options={field.vocabulary} placeholder="select or type a location…" />
  }
  // Select with fixed options
  if (field.options.length > 0) {
    return (
      <select value={value} onChange={(e) => setValue(e.target.value)} style={inputStyle}>
        <option value="">--</option>
        {field.options.map((option) => (
          <option key={String(option.value)} value={String(option.value)}>{option.label}</option>
        ))}
      </select>
    )
  }
  // Textarea
  if (field.mobile_widget === 'textarea') {
    return <textarea value={value} onChange={(e) => setValue(e.target.value)} rows={3} style={inputStyle} />
  }
  // Number with min/max
  if (field.mobile_widget === 'number') {
    return (
      <input
        type="number"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        min={field.min_value ?? undefined}
        max={field.max_value ?? undefined}
        step={field.field_type.includes('float') ? 'any' : 1}
        style={inputStyle}
      />
    )
  }
  // Default text
  return <input type="text" value={value} onChange={(e) => setValue(e.target.value)} style={inputStyle} />
}

// ---- Location group: name + lat + lon ----
function LocationGroup({
  fields, values, setValues,
}: {
  fields: SchemaField[]
  values: Record<string, string>
  setValues: React.Dispatch<React.SetStateAction<Record<string, string>>>
}) {
  const locField = fields.find(f => f.name === 'location')
  const latField = fields.find(f => f.name === 'latitude')
  const lonField = fields.find(f => f.name === 'longitude')
  function set(name: string, val: string) { setValues(prev => ({ ...prev, [name]: val })) }

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      {locField && (
        <label style={{ display: 'grid', gap: 4 }}>
          <span>{locField.display_name}{locField.required ? ' *' : ''}</span>
          {renderFieldInput(locField, values[locField.name] ?? '', v => set(locField.name, v))}
        </label>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {latField && (
          <label style={{ display: 'grid', gap: 4 }}>
            <span>Latitude</span>
            <input type="number" step="any" min={-90} max={90}
              value={values['latitude'] ?? ''} onChange={e => set('latitude', e.target.value)}
              placeholder="e.g. 48.546" style={inputStyle} />
          </label>
        )}
        {lonField && (
          <label style={{ display: 'grid', gap: 4 }}>
            <span>Longitude</span>
            <input type="number" step="any" min={-180} max={180}
              value={values['longitude'] ?? ''} onChange={e => set('longitude', e.target.value)}
              placeholder="e.g. -123.013" style={inputStyle} />
          </label>
        )}
      </div>
    </div>
  )
}

// ---- Main sheet ----
export function MetadataSheet({
  open, initialDraft, onClose, onReady,
}: {
  open: boolean
  initialDraft: MetadataDraft
  onClose: () => void
  onReady: (draft: MetadataDraft) => void
}) {
  const [fields, setFields] = useState<SchemaField[]>([])
  const [targetType, setTargetType] = useState<'query' | 'gallery'>(initialDraft.targetType)
  const [targetMode, setTargetMode] = useState<'create' | 'append'>(initialDraft.targetMode)
  const [targetId, setTargetId] = useState(initialDraft.targetId)
  const [encounterDate, setEncounterDate] = useState(initialDraft.encounterDate)
  const [encounterSuffix, setEncounterSuffix] = useState(initialDraft.encounterSuffix)
  const [values, setValues] = useState<Record<string, string>>(initialDraft.values)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  useEffect(() => {
    if (!open) return
    getMetadataSchema().then(data => setFields(data.fields)).catch(err => setError(String(err)))
  }, [open])

  useEffect(() => {
    setTargetType(initialDraft.targetType)
    setTargetMode(initialDraft.targetMode)
    setTargetId(initialDraft.targetId)
    setEncounterDate(initialDraft.encounterDate)
    setEncounterSuffix(initialDraft.encounterSuffix)
    setValues(initialDraft.values)
  }, [initialDraft])

  useEffect(() => { if (targetType === 'gallery') setTargetMode('append') }, [targetType])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    return q ? fields.filter(f => f.display_name.toLowerCase().includes(q) || f.name.toLowerCase().includes(q) || f.group_display_name.toLowerCase().includes(q)) : fields
  }, [fields, search])

  const groupedFields = useMemo(() => {
    const groups = new Map<string, { displayName: string; fields: SchemaField[] }>()
    for (const field of filtered) {
      if (!groups.has(field.group)) groups.set(field.group, { displayName: field.group_display_name, fields: [] })
      groups.get(field.group)!.fields.push(field)
    }
    return Array.from(groups.entries())
  }, [filtered])

  function markReady() {
    if (!targetId) { setError('Choose a target ID before marking metadata ready.'); return }
    onReady({ targetType, targetMode, targetId, encounterDate, encounterSuffix, values, ready: true })
    onClose()
  }

  if (!open) return null

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(17,24,39,.22)', display: 'grid', alignItems: 'end', zIndex: 40 }}>
      <div style={{ background: 'white', borderTopLeftRadius: 20, borderTopRightRadius: 20, padding: '10px 12px 18px', maxHeight: '82vh', overflow: 'auto', boxShadow: '0 -8px 28px rgba(0,0,0,.14)' }}>
        <div style={{ width: 52, height: 5, borderRadius: 999, background: '#d1d6df', margin: '0 auto 10px' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, marginBottom: 10 }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>Metadata</div>
            <div style={{ color: '#667085', fontSize: 13 }}>Complete this sheet, then mark it ready.</div>
          </div>
          <button onClick={onClose} style={{ border: '1px solid #ccd6eb', background: 'white', borderRadius: 10, padding: '8px 10px' }}>Close</button>
        </div>

        {/* Target section */}
        <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid #ddd', borderRadius: 12, background: '#fbfcfe' }}>
          <label style={{ display: 'grid', gap: 4 }}>Target type
            <select value={targetType} onChange={e => setTargetType(e.target.value as 'query' | 'gallery')} style={inputStyle}>
              <option value="query">Query</option>
              <option value="gallery">Gallery</option>
            </select>
          </label>
          <label style={{ display: 'grid', gap: 4 }}>Target mode
            <select value={targetMode} onChange={e => setTargetMode(e.target.value as 'create' | 'append')} disabled={targetType === 'gallery'} style={inputStyle}>
              <option value="create">Create</option>
              <option value="append">Append</option>
            </select>
          </label>
          <label style={{ display: 'grid', gap: 4 }}>Target ID</label>
          {targetMode === 'create' && targetType === 'query' ? (
            <input value={targetId} onChange={e => setTargetId(e.target.value)} placeholder="new query ID" style={inputStyle} />
          ) : (
            <TargetIdPicker value={targetId} onChange={setTargetId} entityType={targetType} />
          )}
          <label style={{ display: 'grid', gap: 4 }}>Encounter date<input type="date" value={encounterDate} onChange={e => setEncounterDate(e.target.value)} style={inputStyle} /></label>
          <label style={{ display: 'grid', gap: 4 }}>Encounter suffix (optional)<input value={encounterSuffix} onChange={e => setEncounterSuffix(e.target.value)} style={inputStyle} /></label>
        </div>

        {/* Search */}
        <label style={{ display: 'grid', gap: 4, marginTop: 10 }}>Search metadata fields
          <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search by field or group" style={inputStyle} />
        </label>

        {/* Metadata fields */}
        <div style={{ display: 'grid', gap: 10, marginTop: 10 }}>
          {groupedFields.map(([groupName, group]) => {
            // Location group gets special compound rendering
            if (groupName === 'location') {
              return (
                <details key={groupName} open style={{ border: '1px solid #ddd', borderRadius: 12, padding: 10, background: 'white' }}>
                  <summary style={{ fontWeight: 600, cursor: 'pointer' }}>{group.displayName}</summary>
                  <div style={{ marginTop: 10 }}>
                    <LocationGroup fields={group.fields} values={values} setValues={setValues} />
                  </div>
                </details>
              )
            }
            return (
              <details key={groupName} open={groupName === 'numeric' || groupName === 'notes'} style={{ border: '1px solid #ddd', borderRadius: 12, padding: 10, background: 'white' }}>
                <summary style={{ fontWeight: 600, cursor: 'pointer' }}>{group.displayName} ({group.fields.length})</summary>
                <div style={{ display: 'grid', gap: 10, marginTop: 10 }}>
                  {group.fields.map(field => (
                    <label key={field.name} style={{ display: 'grid', gap: 4 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                        <span>{field.display_name}{field.required ? ' *' : ''}</span>
                        <span style={{ fontSize: 12, color: '#666' }}>{field.name}</span>
                      </div>
                      {renderFieldInput(field, values[field.name] ?? '', next => setValues(prev => ({ ...prev, [field.name]: next })))}
                      {field.tooltip && <div style={{ fontSize: 12, color: '#666' }}>{field.tooltip}</div>}
                    </label>
                  ))}
                </div>
              </details>
            )
          })}
        </div>

        <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
          <button onClick={markReady} style={{ padding: 12, borderRadius: 12, background: '#2f6fed', color: 'white', border: '1px solid #2f6fed', fontWeight: 600 }}>Ready</button>
          {error && <div style={{ color: 'crimson', whiteSpace: 'pre-wrap' }}>{error}</div>}
        </div>
      </div>
    </div>
  )
}
