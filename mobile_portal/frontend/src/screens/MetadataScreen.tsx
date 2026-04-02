import { useEffect, useMemo, useState } from 'react'
import { getMetadataSchema, submitObservation, type SchemaField } from '../api/client'

function renderFieldInput(field: SchemaField, value: string, setValue: (value: string) => void) {
  if (field.options.length > 0) {
    return (
      <select value={value} onChange={(e) => setValue(e.target.value)}>
        <option value="">--</option>
        {field.options.map((option) => <option key={String(option.value)} value={String(option.value)}>{option.label}</option>)}
      </select>
    )
  }
  if (field.vocabulary.length > 0) {
    const listId = `${field.name}-options`
    return (
      <>
        <input list={listId} value={value} onChange={(e) => setValue(e.target.value)} />
        <datalist id={listId}>
          {field.vocabulary.map((option) => <option key={option} value={option} />)}
        </datalist>
      </>
    )
  }
  if (field.mobile_widget === 'textarea') {
    return <textarea value={value} onChange={(e) => setValue(e.target.value)} rows={4} />
  }
  const inputType = field.field_type.includes('numeric') ? 'number' : 'text'
  return <input type={inputType} value={value} onChange={(e) => setValue(e.target.value)} />
}

export function MetadataScreen({ files }: { files: File[] }) {
  const [fields, setFields] = useState<SchemaField[]>([])
  const [values, setValues] = useState<Record<string, string>>({})
  const [targetType, setTargetType] = useState<'query' | 'gallery'>('query')
  const [targetMode, setTargetMode] = useState<'create' | 'append'>('create')
  const [targetId, setTargetId] = useState('')
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
        <label>Target type
          <select value={targetType} onChange={(e) => setTargetType(e.target.value as 'query' | 'gallery')}>
            <option value="query">Query</option>
            <option value="gallery">Gallery</option>
          </select>
        </label>
        <label>Target mode
          <select value={targetMode} onChange={(e) => setTargetMode(e.target.value as 'create' | 'append')} disabled={targetType === 'gallery'}>
            <option value="create">Create</option>
            <option value="append">Append</option>
          </select>
        </label>
        <label>Target ID<input value={targetId} onChange={(e) => setTargetId(e.target.value)} placeholder={targetType === 'gallery' ? 'known gallery ID (e.g. anchovy)' : 'new or existing query ID'} /></label>
        <label>Encounter date<input type="date" value={encounterDate} onChange={(e) => setEncounterDate(e.target.value)} /></label>
        <label>Encounter suffix (optional)<input value={encounterSuffix} onChange={(e) => setEncounterSuffix(e.target.value)} placeholder="dock / sample / diverA" /></label>
        <div style={{ fontSize: 13, color: '#555' }}>Selected local files: {files.length}</div>
      </div>
      <label>Search metadata fields
        <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search by field or group" />
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
      <button onClick={onSubmit} disabled={!targetId || files.length === 0}>Submit</button>
      {message && <div style={{ color: 'green' }}>{message}</div>}
      {error && <div style={{ color: 'crimson', whiteSpace: 'pre-wrap' }}>{error}</div>}
    </div>
  )
}
