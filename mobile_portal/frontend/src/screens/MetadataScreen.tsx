import { useEffect, useMemo, useState } from 'react'
import { getMetadataSchema, submitObservation, suggestEntities, type SchemaField } from '../api/client'

function renderFieldInput(field: SchemaField, value: string, setValue: (value: string) => void) {
  if (field.options.length > 0) {
    return (
      <select value={value} onChange={(e) => setValue(e.target.value)} style={{ padding: 10, borderRadius: 8, border: '1px solid #ccd6eb' }}>
        <option value="">--</option>
        {field.options.map((option) => <option key={String(option.value)} value={String(option.value)}>{option.label}</option>)}
      </select>
    )
  }
  if (field.vocabulary.length > 0) {
    const listId = `${field.name}-options`
    return (
      <>
        <input list={listId} value={value} onChange={(e) => setValue(e.target.value)} style={{ padding: 10, borderRadius: 8, border: '1px solid #ccd6eb' }} />
        <datalist id={listId}>
          {field.vocabulary.map((option) => <option key={option} value={option} />)}
        </datalist>
      </>
    )
  }
  if (field.mobile_widget === 'textarea') {
    return <textarea value={value} onChange={(e) => setValue(e.target.value)} rows={4} style={{ padding: 10, borderRadius: 8, border: '1px solid #ccd6eb' }} />
  }
  const inputType = field.field_type.includes('numeric') ? 'number' : 'text'
  return <input type={inputType} value={value} onChange={(e) => setValue(e.target.value)} style={{ padding: 10, borderRadius: 8, border: '1px solid #ccd6eb' }} />
}

export function MetadataScreen({ files }: { files: File[] }) {
  const [fields, setFields] = useState<SchemaField[]>([])
  const [values, setValues] = useState<Record<string, string>>({})
  const [targetType, setTargetType] = useState<'query' | 'gallery'>('query')
  const [targetMode, setTargetMode] = useState<'create' | 'append'>('create')
  const [targetId, setTargetId] = useState('')
  const [targetSuggestions, setTargetSuggestions] = useState<string[]>([])
  const [encounterDate, setEncounterDate] = useState(new Date().toISOString().slice(0, 10))
  const [encounterSuffix, setEncounterSuffix] = useState('')
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')

  useEffect(() => {
    getMetadataSchema().then((data) => setFields(data.fields)).catch((err) => setError(String(err)))
  }, [])

  useEffect(() => {
    if (targetType === 'gallery') setTargetMode('append')
  }, [targetType])

  useEffect(() => {
    const q = targetId.trim()
    if (!q || (targetType === 'query' && targetMode === 'create')) {
      setTargetSuggestions([])
      return
    }
    const handle = window.setTimeout(() => {
      suggestEntities(targetType, q, 8).then((data) => setTargetSuggestions(data.items)).catch(() => setTargetSuggestions([]))
    }, 150)
    return () => window.clearTimeout(handle)
  }, [targetId, targetType, targetMode])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    return q ? fields.filter((field) => field.display_name.toLowerCase().includes(q) || field.name.toLowerCase().includes(q) || field.group_display_name.toLowerCase().includes(q)) : fields
  }, [fields, search])

  const groupedFields = useMemo(() => {
    const groups = new Map<string, { displayName: string; fields: SchemaField[] }>()
    for (const field of filtered) {
      if (!groups.has(field.group)) groups.set(field.group, { displayName: field.group_display_name, fields: [] })
      groups.get(field.group)!.fields.push(field)
    }
    return Array.from(groups.entries())
  }, [filtered])

  async function onSubmit() {
    setError(null)
    setMessage(null)
    try {
      const result = await submitObservation({
        target_type: targetType,
        target_mode: targetMode,
        target_id: targetId,
        encounter_date: encounterDate,
        encounter_suffix: encounterSuffix,
        metadata: values,
      }, files)
      setMessage(`${result.message}: ${result.entity_type}/${result.entity_id} ${result.encounter_folder}`)
    } catch (err) {
      setError(String(err))
    }
  }

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <h2>Metadata + Submit</h2>
      <div style={{ color: '#555', fontSize: 14 }}>Fill the full starBoard metadata set, then submit the currently selected local photos into the archive.</div>
      <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid #ddd', borderRadius: 10, background: 'white' }}>
        <label style={{ display: 'grid', gap: 4 }}>Target type
          <select value={targetType} onChange={(e) => setTargetType(e.target.value as 'query' | 'gallery')} style={{ padding: 10, borderRadius: 8, border: '1px solid #ccd6eb' }}>
            <option value="query">Query</option>
            <option value="gallery">Gallery</option>
          </select>
        </label>
        <label style={{ display: 'grid', gap: 4 }}>Target mode
          <select value={targetMode} onChange={(e) => setTargetMode(e.target.value as 'create' | 'append')} disabled={targetType === 'gallery'} style={{ padding: 10, borderRadius: 8, border: '1px solid #ccd6eb' }}>
            <option value="create">Create</option>
            <option value="append">Append</option>
          </select>
        </label>
        <label style={{ display: 'grid', gap: 4 }}>Target ID
          <input value={targetId} onChange={(e) => setTargetId(e.target.value)} placeholder={targetType === 'gallery' ? 'known gallery ID (e.g. anchovy)' : 'new or existing query ID'} style={{ padding: 10, borderRadius: 8, border: '1px solid #ccd6eb' }} />
        </label>
        {targetSuggestions.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {targetSuggestions.map((item) => (
              <button key={item} onClick={() => setTargetId(item)} style={{ border: '1px solid #ccd6eb', background: 'white', borderRadius: 999, padding: '6px 10px', fontSize: 13 }}>{item}</button>
            ))}
          </div>
        )}
        <label style={{ display: 'grid', gap: 4 }}>Encounter date<input type="date" value={encounterDate} onChange={(e) => setEncounterDate(e.target.value)} style={{ padding: 10, borderRadius: 8, border: '1px solid #ccd6eb' }} /></label>
        <label style={{ display: 'grid', gap: 4 }}>Encounter suffix (optional)<input value={encounterSuffix} onChange={(e) => setEncounterSuffix(e.target.value)} placeholder="dock / sample / diverA" style={{ padding: 10, borderRadius: 8, border: '1px solid #ccd6eb' }} /></label>
        <div style={{ fontSize: 13, color: '#555' }}>Selected local files: {files.length}</div>
      </div>
      <label style={{ display: 'grid', gap: 4 }}>Search metadata fields
        <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search by field or group" style={{ padding: 10, borderRadius: 8, border: '1px solid #ccd6eb' }} />
      </label>
      <div style={{ display: 'grid', gap: 12 }}>
        {groupedFields.map(([groupName, group]) => (
          <details key={groupName} open={groupName === 'location' || groupName === 'numeric' || groupName === 'notes'} style={{ border: '1px solid #ddd', borderRadius: 10, padding: 10, background: 'white' }}>
            <summary style={{ fontWeight: 600, cursor: 'pointer' }}>{group.displayName} ({group.fields.length})</summary>
            <div style={{ display: 'grid', gap: 10, marginTop: 10 }}>
              {group.fields.map((field) => (
                <label key={field.name} style={{ display: 'grid', gap: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                    <span>{field.display_name}{field.required ? ' *' : ''}</span>
                    <span style={{ fontSize: 12, color: '#666' }}>{field.name}</span>
                  </div>
                  {renderFieldInput(field, values[field.name] ?? '', (next) => setValues((prev) => ({ ...prev, [field.name]: next })))}
                  {field.tooltip && <div style={{ fontSize: 12, color: '#666' }}>{field.tooltip}</div>}
                </label>
              ))}
            </div>
          </details>
        ))}
      </div>
      <button onClick={onSubmit} disabled={!targetId || files.length === 0} style={{ padding: 12, borderRadius: 10, background: '#2f6fed', color: 'white', border: '1px solid #2f6fed' }}>Submit</button>
      {message && <div style={{ color: 'green' }}>{message}</div>}
      {error && <div style={{ color: 'crimson', whiteSpace: 'pre-wrap' }}>{error}</div>}
    </div>
  )
}
