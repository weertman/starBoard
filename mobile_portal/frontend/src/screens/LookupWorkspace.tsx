import { useEffect, useMemo, useState } from 'react'
import { lookupEntity, getEntityImages, getEntityEncounters, getLookupOptions, type ArchiveEntityResponse, type ImageDescriptor, type EncounterOption } from '../api/client'
import { ArchiveImageStrip } from '../components/ArchiveImageStrip'
import { ZoomableImagePane } from '../components/ZoomableImagePane'

export function LookupWorkspace({
  selectedArchiveImage,
  onSelectArchiveImage,
  onBack,
  onOpenMetadata,
  canCompare,
}: {
  selectedArchiveImage?: ImageDescriptor | null
  onSelectArchiveImage: (image: ImageDescriptor, loadedItems?: ImageDescriptor[]) => void
  onBack: () => void
  onOpenMetadata: () => void
  canCompare: boolean
}) {
  const [entityType, setEntityType] = useState<'gallery' | 'query'>('gallery')
  const [entityId, setEntityId] = useState('')
  const [location, setLocation] = useState('')
  const [selectedEncounter, setSelectedEncounter] = useState('')
  const [locationOptions, setLocationOptions] = useState<string[]>([])
  const [idOptions, setIdOptions] = useState<string[]>([])
  const [encounterOptions, setEncounterOptions] = useState<EncounterOption[]>([])
  const [result, setResult] = useState<ArchiveEntityResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const metadataRows = useMemo(() => Object.entries(result?.metadata_summary ?? {}).filter(([, value]) => String(value ?? '').trim() !== ''), [result])
  const activeArchiveImage = selectedArchiveImage ?? result?.image_window.items[0] ?? null

  useEffect(() => {
    getLookupOptions(entityType, location, 300)
      .then((data) => {
        setLocationOptions(data.locations)
        setIdOptions(data.ids)
        if (entityId && !data.ids.includes(entityId)) {
          setEntityId('')
          setEncounterOptions([])
          setSelectedEncounter('')
        }
      })
      .catch((err) => setError(String(err)))
  }, [entityType, location])

  useEffect(() => {
    if (!entityId) {
      setEncounterOptions([])
      setSelectedEncounter('')
      return
    }
    getEntityEncounters(entityId, entityType)
      .then((data) => {
        setEncounterOptions(data.encounters)
        setSelectedEncounter('')
      })
      .catch((err) => setError(String(err)))
  }, [entityId, entityType])

  async function doLookup(targetId?: string, encounterOverride?: string) {
    const id = (targetId ?? entityId).trim()
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const useEncounter = encounterOverride ?? selectedEncounter
      const data = await lookupEntity(id, entityType, useEncounter)
      setEntityId(id)
      setResult(data)
      setEncounterOptions(data.encounters)
      setSelectedEncounter(data.selected_encounter || useEncounter || '')
      if (data.image_window.items[0]) onSelectArchiveImage(data.image_window.items[0], data.image_window.items)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  async function loadMore() {
    if (result?.image_window.next_offset == null) return
    const next = await getEntityImages(result.entity_id, entityType, result.image_window.next_offset, 4, selectedEncounter)
    const mergedItems = [...result.image_window.items, ...next.items]
    setResult({ ...result, image_window: { ...next, items: mergedItems } })
    if (activeArchiveImage) onSelectArchiveImage(activeArchiveImage, mergedItems)
  }

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
          <button onClick={onBack} style={{ border: '1px solid #ccd6eb', borderRadius: 10, padding: '8px 10px', background: 'white' }}>Home</button>
          <div style={{ fontWeight: 700, fontSize: 15 }}>Look up star</div>
          <button onClick={onOpenMetadata} style={{ border: '1px solid #ccd6eb', borderRadius: 10, padding: '8px 10px', background: canCompare ? '#eef4ff' : 'white' }}>Metadata</button>
        </div>
        <div style={{ color: '#667085', fontSize: 13, lineHeight: 1.35 }}>Browse by type, location, ID, and then observation date for that selected ID.</div>
      </div>

      <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 10 }}>
        <label style={{ display: 'grid', gap: 4 }}>
          <span style={{ fontSize: 13, color: '#667085' }}>Type</span>
          <select value={entityType} onChange={(e) => { setEntityType(e.target.value as 'gallery' | 'query'); setResult(null); setSelectedEncounter(''); setEncounterOptions([]) }} style={{ padding: 10, borderRadius: 10, border: '1px solid #ccd6eb' }}>
            <option value="gallery">Gallery</option>
            <option value="query">Queries</option>
          </select>
        </label>
        <label style={{ display: 'grid', gap: 4 }}>
          <span style={{ fontSize: 13, color: '#667085' }}>Location filter</span>
          <select value={location} onChange={(e) => { setLocation(e.target.value); setResult(null); setSelectedEncounter(''); setEncounterOptions([]) }} style={{ padding: 10, borderRadius: 10, border: '1px solid #ccd6eb' }}>
            <option value="">All locations</option>
            {locationOptions.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <label style={{ display: 'grid', gap: 4 }}>
          <span style={{ fontSize: 13, color: '#667085' }}>ID</span>
          <select value={entityId} onChange={(e) => { setEntityId(e.target.value); setResult(null); setSelectedEncounter('') }} style={{ padding: 10, borderRadius: 10, border: '1px solid #ccd6eb' }}>
            <option value="">Select an ID</option>
            {idOptions.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <label style={{ display: 'grid', gap: 4 }}>
          <span style={{ fontSize: 13, color: '#667085' }}>Observation date</span>
          <select value={selectedEncounter} onChange={(e) => setSelectedEncounter(e.target.value)} disabled={!entityId || encounterOptions.length === 0} style={{ padding: 10, borderRadius: 10, border: '1px solid #ccd6eb' }}>
            <option value="">All observation dates</option>
            {encounterOptions.map((item) => <option key={item.encounter} value={item.encounter}>{item.label}</option>)}
          </select>
        </label>
        <button onClick={() => void doLookup()} disabled={!entityId || loading} style={{ padding: 11, borderRadius: 10, background: '#2f6fed', color: 'white', border: '1px solid #2f6fed' }}>{loading ? 'Loading…' : 'Open selection'}</button>
      </div>

      {error && <div style={{ color: 'crimson', whiteSpace: 'pre-wrap' }}>{error}</div>}

      {result && (
        <>
          <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 6 }}>
            <div style={{ fontSize: 14 }}>Opened <strong>{result.entity_id}</strong></div>
            <div style={{ fontSize: 13, color: '#555' }}>{result.entity_type} • {selectedEncounter || 'all observation dates'} • {result.image_window.total} archive image{result.image_window.total === 1 ? '' : 's'} found • {result.image_window.items.length} loaded</div>
            {metadataRows.length > 0 && <details><summary>Metadata summary</summary><div style={{ display: 'grid', gap: 4, marginTop: 8, fontSize: 13 }}>{metadataRows.slice(0, 12).map(([key, value]) => <div key={key}><strong>{key}</strong>: {String(value)}</div>)}</div></details>}
          </div>
          <ZoomableImagePane compact title="Archive workspace" subtitle={activeArchiveImage?.label} src={activeArchiveImage?.fullres_url ?? activeArchiveImage?.preview_url} />
          <ArchiveImageStrip items={result.image_window.items} onSelect={(img) => onSelectArchiveImage(img, result.image_window.items)} selectedImageId={activeArchiveImage?.image_id ?? null} />
          {result.image_window.next_offset != null && <button onClick={loadMore}>Load more archive images</button>}
        </>
      )}
    </div>
  )
}
