import { useEffect, useMemo, useState } from 'react'

import {
  getMetadataSchema,
  submitEntry,
  type SchemaField,
  type SubmissionResponse,
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
  return order.map((key) => ({ key, ...groups.get(key)! }))
}

export function SingleEntryPage() {
  const [schema, setSchema] = useState<SchemaField[]>([])
  const [metadata, setMetadata] = useState<Record<string, string>>({})
  const [targetType, setTargetType] = useState<'gallery' | 'query'>('query')
  const [targetMode, setTargetMode] = useState<'create' | 'append'>('create')
  const [targetId, setTargetId] = useState('')
  const [encounterDate, setEncounterDate] = useState(new Date().toISOString().slice(0, 10))
  const [encounterSuffix, setEncounterSuffix] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<SubmissionResponse | null>(null)

  useEffect(() => {
    void (async () => {
      try {
        const response = await getMetadataSchema()
        setSchema(response.fields)
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

  const grouped = useMemo(() => groupFields(schema), [schema])

  function updateMetadata(name: string, value: string) {
    setMetadata((current) => ({ ...current, [name]: value }))
  }

  async function handleSubmit() {
    if (!targetId.trim() || files.length === 0) return
    setBusy(true)
    setError(null)
    setResult(null)
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
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(false)
    }
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

    if (field.mobile_widget === 'textarea') {
      return <textarea {...commonProps} rows={3} />
    }
    if (field.mobile_widget === 'select' && field.options.length > 0) {
      return (
        <select {...commonProps}>
          <option value="">—</option>
          {field.options.map((option) => (
            <option key={`${field.name}-${option.value}`} value={String(option.value)}>{option.label}</option>
          ))}
        </select>
      )
    }
    if ((field.mobile_widget === 'color_select' || field.mobile_widget === 'location') && field.vocabulary.length > 0) {
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

  return (
    <main style={{ maxWidth: 1180, margin: '0 auto', padding: 18, fontFamily: 'system-ui, sans-serif', color: '#152033', background: '#f7f9fc', minHeight: '100vh' }}>
      <div style={{ display: 'grid', gap: 16 }}>
        <section style={card}>
          <h1 style={{ marginTop: 0 }}>Single Entry</h1>
          <p style={{ marginTop: 0, color: '#516070' }}>Schema-driven browser data entry for creating or appending archive observations.</p>
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
            <label>
              <div>Target ID</div>
              <input aria-label="Target ID" value={targetId} onChange={(e) => setTargetId(e.target.value)} style={input} />
            </label>
            <label>
              <div>Encounter date</div>
              <input aria-label="Encounter date" type="date" value={encounterDate} onChange={(e) => setEncounterDate(e.target.value)} style={input} />
            </label>
            <label>
              <div>Encounter suffix</div>
              <input aria-label="Encounter suffix" value={encounterSuffix} onChange={(e) => setEncounterSuffix(e.target.value)} style={input} />
            </label>
            <label>
              <div>Images</div>
              <input aria-label="Images" type="file" multiple onChange={(e) => setFiles(Array.from(e.target.files ?? []))} style={input} />
            </label>
          </div>
        </section>

        {grouped.map((group) => (
          <section key={group.key} style={card}>
            <h2 style={{ marginTop: 0 }}>{group.title}</h2>
            <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
              {group.fields.map((field) => (
                <label key={field.name}>
                  <div>{field.display_name}</div>
                  {renderField(field)}
                </label>
              ))}
            </div>
          </section>
        ))}

        {error && <section style={{ ...card, borderColor: '#e29a9a', background: '#fff5f5', color: '#7a1c1c' }}><b>Error:</b> {error}</section>}

        <section style={card}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <button onClick={() => void handleSubmit()} disabled={busy || !targetId.trim() || files.length === 0} style={{ padding: '8px 12px' }}>
              {busy ? 'Submitting…' : 'Submit entry'}
            </button>
            <span style={{ color: '#516070' }}>{files.length} file(s) selected</span>
          </div>
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
