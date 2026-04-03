import { useEffect, useMemo, useState } from 'react'
import { lookupEntity, getEntityImages, getEntityEncounters, getLookupOptions, type ArchiveEntityResponse, type ImageDescriptor, type EncounterOption } from '../api/client'
import { ArchiveImageStrip } from '../components/ArchiveImageStrip'
import { ZoomableImagePane } from '../components/ZoomableImagePane'

function imageIndex(image: ImageDescriptor | null | undefined): number {
  if (!image) return 0
  const parts = image.image_id.split(':')
  const value = Number(parts[parts.length - 1])
  return Number.isFinite(value) ? value : 0
}

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
  const currentWindow = result?.image_window.items ?? []
  const activeArchiveImage = selectedArchiveImage ?? currentWindow[0] ?? null
  const activeWindowIndex = currentWindow.findIndex((item) => item.image_id === activeArchiveImage?.image_id)
  const activeAbsoluteIndex = imageIndex(activeArchiveImage)

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

  async function doLookup(targetId?: string, encounterOverride?: string, offset = 0) {
    const id = (targetId ?? entityId).trim()
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const useEncounter = encounterOverride ?? selectedEncounter
      const data = offset === 0
        ? await lookupEntity(id, entityType, useEncounter)
        : {
            ...(result ?? { entity_type: entityType, entity_id: id, metadata_summary: {}, encounters: encounterOptions, selected_encounter: useEncounter, image_window: { offset: 0, count: 0, total: 0, items: [], next_offset: null } }),
            image_window: await getEntityImages(id, entityType, offset, 4, useEncounter),
            selected_encounter: useEncounter,
            encounters: encounterOptions,
          }
      setEntityId(id)
      setResult(data as ArchiveEntityResponse)
      if (offset === 0 && 'encounters' in data) {
        setEncounterOptions((data as ArchiveEntityResponse).encounters)
        setSelectedEncounter((data as ArchiveEntityResponse).selected_encounter || useEncounter || '')
      }
      const first = (data as ArchiveEntityResponse).image_window.items[0]
      if (first) onSelectArchiveImage(first, (data as ArchiveEntityResponse).image_window.items)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  async function shiftWindow(targetAbsoluteIndex: number) {
    if (!entityId) return
    const nextOffset = Math.max(0, targetAbsoluteIndex - 1)
    setLoading(true)
    setError(null)
    try {
      const imageWindow = await getEntityImages(entityId, entityType, nextOffset, 4, selectedEncounter)
      const nextResult: ArchiveEntityResponse = {
        entity_type: entityType,
        entity_id: entityId,
        metadata_summary: result?.metadata_summary ?? {},
        encounters: encounterOptions,
        selected_encounter: selectedEncounter,
        image_window: imageWindow,
      }
      setResult(nextResult)
      const chosen = imageWindow.items.find((item) => imageIndex(item) === targetAbsoluteIndex) ?? imageWindow.items[0]
      if (chosen) onSelectArchiveImage(chosen, imageWindow.items)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  function selectFromCurrentWindow(image: ImageDescriptor) {
    onSelectArchiveImage(image, currentWindow)
  }

  function prevImage() {
    if (!result || activeAbsoluteIndex <= 0) return
    const target = activeAbsoluteIndex - 1
    if (activeWindowIndex > 0) {
      selectFromCurrentWindow(currentWindow[activeWindowIndex - 1])
      return
    }
    void shiftWindow(target)
  }

  function nextImage() {
    if (!result || activeAbsoluteIndex >= result.image_window.total - 1) return
    const target = activeAbsoluteIndex + 1
    if (activeWindowIndex >= 0 && activeWindowIndex < currentWindow.length - 1) {
      selectFromCurrentWindow(currentWindow[activeWindowIndex + 1])
      return
    }
    void shiftWindow(target)
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
          <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 8 }}>
            <div style={{ fontSize: 14 }}>Opened <strong>{result.entity_id}</strong></div>
            <div style={{ fontSize: 13, color: '#555' }}>{result.entity_type} • {selectedEncounter || 'all observation dates'} • image {activeAbsoluteIndex + 1} of {result.image_window.total}</div>
            {metadataRows.length > 0 && <details><summary>Metadata summary</summary><div style={{ display: 'grid', gap: 4, marginTop: 8, fontSize: 13 }}>{metadataRows.slice(0, 12).map(([key, value]) => <div key={key}><strong>{key}</strong>: {String(value)}</div>)}</div></details>}
          </div>
          <ZoomableImagePane compact title="Archive workspace" subtitle={activeArchiveImage?.label} src={activeArchiveImage?.fullres_url ?? activeArchiveImage?.preview_url} />
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={prevImage} disabled={loading || activeAbsoluteIndex <= 0} style={{ flex: 1, padding: 11, borderRadius: 10, border: '1px solid #ccd6eb', background: 'white' }}>Previous</button>
            <button onClick={nextImage} disabled={loading || activeAbsoluteIndex >= result.image_window.total - 1} style={{ flex: 1, padding: 11, borderRadius: 10, border: '1px solid #ccd6eb', background: 'white' }}>Next</button>
          </div>
          <div style={{ color: '#667085', fontSize: 12 }}>Only a small rolling window of archive images is kept loaded at a time.</div>
          <ArchiveImageStrip items={currentWindow} onSelect={selectFromCurrentWindow} selectedImageId={activeArchiveImage?.image_id ?? null} />
        </>
      )}
    </div>
  )
}
