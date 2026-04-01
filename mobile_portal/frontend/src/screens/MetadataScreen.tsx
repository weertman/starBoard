import { useEffect, useMemo, useState } from 'react'
import { getMetadataSchema, submitObservation, type SchemaField } from '../api/client'

export function MetadataScreen({ files }: { files: File[] }) {
  const [fields, setFields] = useState<SchemaField[]>([])
  const [values, setValues] = useState<Record<string, string>>({})
  const [targetType, setTargetType] = useState<'query' | 'gallery'>('query')
  const [targetMode, setTargetMode] = useState<'create' | 'append'>('create')
  const [targetId, setTargetId] = useState('')
  const [encounterDate, setEncounterDate] = useState(new Date().toISOString().slice(0, 10))
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getMetadataSchema().then((data) => setFields(data.fields)).catch((err) => setError(String(err)))
  }, [])

  const visibleFields = useMemo(() => fields.slice(0, 12), [fields])

  async function onSubmit() {
    setError(null)
    setMessage(null)
    try {
      const result = await submitObservation({
        target_type: targetType,
        target_mode: targetMode,
        target_id: targetId,
        encounter_date: encounterDate,
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
      <div style={{ display: 'grid', gap: 8 }}>
        <label>Target type
          <select value={targetType} onChange={(e) => setTargetType(e.target.value as 'query' | 'gallery')}>
            <option value="query">Query</option>
            <option value="gallery">Gallery</option>
          </select>
        </label>
        <label>Target mode
          <select value={targetMode} onChange={(e) => setTargetMode(e.target.value as 'create' | 'append')}>
            <option value="create">Create</option>
            <option value="append">Append</option>
          </select>
        </label>
        <label>Target ID<input value={targetId} onChange={(e) => setTargetId(e.target.value)} /></label>
        <label>Encounter date<input type="date" value={encounterDate} onChange={(e) => setEncounterDate(e.target.value)} /></label>
      </div>
      <div style={{ display: 'grid', gap: 8 }}>
        {visibleFields.map((field) => (
          <label key={field.name}>
            {field.display_name}
            {field.options.length > 0 ? (
              <select value={values[field.name] ?? ''} onChange={(e) => setValues((prev) => ({ ...prev, [field.name]: e.target.value }))}>
                <option value="">--</option>
                {field.options.map((option) => <option key={String(option.value)} value={String(option.value)}>{option.label}</option>)}
              </select>
            ) : (
              <input value={values[field.name] ?? ''} onChange={(e) => setValues((prev) => ({ ...prev, [field.name]: e.target.value }))} />
            )}
          </label>
        ))}
      </div>
      <div>Selected local files: {files.length}</div>
      <button onClick={onSubmit} disabled={!targetId || files.length === 0}>Submit</button>
      {message && <div style={{ color: 'green' }}>{message}</div>}
      {error && <div style={{ color: 'crimson' }}>{error}</div>}
    </div>
  )
}
