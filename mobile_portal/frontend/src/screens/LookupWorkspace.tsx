import { useEffect, useMemo, useState } from 'react'
import {
  lookupEntity,
  getEntityImages,
  getEntityEncounters,
  getLookupOptions,
  type ArchiveEntityResponse,
  type ImageDescriptor,
  type EncounterOption,
} from '../api/client'
import { ArchiveImageStrip } from '../components/ArchiveImageStrip'
import { ZoomableImagePane } from '../components/ZoomableImagePane'

const controlStyle = {
  width: '100%',
  maxWidth: '100%',
  minWidth: 0,
  display: 'block',
  padding: 10,
  borderRadius: 10,
  border: '1px solid #ccd6eb',
  boxSizing: 'border-box' as const,
  fontSize: 14,
}

const fieldStyle = {
  display: 'grid',
  gap: 4,
  minWidth: 0,
} as const

function truncateLabel(value: string, max = 24): string {
  if (value.length <= max) return value
  const head = Math.max(8, Math.floor((max - 1) / 2))
  const tail = Math.max(6, max - head - 1)
  return `${value.slice(0, head)}…${value.slice(-tail)}`
}

function displayEncounterLabel(item: EncounterOption): string {
  if (item.date) return item.date
  return truncateLabel(item.label, 24)
}

export function LookupWorkspace({
  selectedArchiveImage,
  onSelectArchiveImage,
  onBack,
  onOpenMetadata,
  canCompare,
  initialRequest,
}: {
  selectedArchiveImage?: ImageDescriptor | null
  onSelectArchiveImage: (image: ImageDescriptor, loadedItems?: ImageDescriptor[]) => void
  onBack: () => void
  onOpenMetadata: () => void
  canCompare: boolean
  initialRequest?: {
    entityType: 'gallery' | 'query'
    entityId: string
    encounter: string
    preferredImageId?: string | null
    nonce: number
  } | null
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

  const metadataRows = useMemo(
    () => Object.entries(result?.metadata_summary ?? {}).filter(([, value]) => String(value ?? '').trim() !== ''),
    [result],
  )
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
          setResult(null)
        }
      })
      .catch((err) => setError(String(err)))
  }, [entityType, location, entityId])

  useEffect(() => {
    if (!entityId) {
      setEncounterOptions([])
      setSelectedEncounter('')
      return
    }
    getEntityEncounters(entityId, entityType)
      .then((data) => {
        setEncounterOptions(data.encounters)
        setSelectedEncounter((prev) => (prev && data.encounters.some((item) => item.encounter === prev) ? prev : ''))
      })
      .catch((err) => setError(String(err)))
  }, [entityId, entityType])

  useEffect(() => {
    if (!initialRequest) return
    setEntityType(initialRequest.entityType)
    setEntityId(initialRequest.entityId)
    setLocation('')
    setSelectedEncounter(initialRequest.encounter)
    void doLookup(initialRequest.entityId, initialRequest.encounter, initialRequest.entityType, initialRequest.preferredImageId)
  }, [initialRequest?.nonce])

  async function doLookup(targetId?: string, encounterOverride?: string, entityTypeOverride?: 'gallery' | 'query', preferredImageId?: string | null) {
    const id = (targetId ?? entityId).trim()
    const nextEntityType = entityTypeOverride ?? entityType
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const useEncounter = encounterOverride ?? selectedEncounter
      const data = await lookupEntity(id, nextEntityType, useEncounter)
      setEntityType(nextEntityType)
      setEntityId(id)
      setResult(data)
      setEncounterOptions(data.encounters)
      setSelectedEncounter(data.selected_encounter || useEncounter || '')
      const preferredImage = preferredImageId ? data.image_window.items.find((item) => item.image_id === preferredImageId) : null
      const nextImage = preferredImage ?? data.image_window.items[0]
      if (nextImage) {
        onSelectArchiveImage(nextImage, data.image_window.items)
      }
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  async function loadMore() {
    if (!result?.image_window.next_offset) return
    const next = await getEntityImages(result.entity_id, entityType, result.image_window.next_offset, 4, selectedEncounter)
    const mergedItems = [...result.image_window.items, ...next.items]
    const mergedResult: ArchiveEntityResponse = {
      ...result,
      image_window: {
        ...next,
        items: mergedItems,
      },
    }
    setResult(mergedResult)
    if (activeArchiveImage) {
      const refreshedActiveImage = mergedItems.find((item) => item.image_id === activeArchiveImage.image_id) ?? activeArchiveImage
      onSelectArchiveImage(refreshedActiveImage, mergedItems)
    }
  }

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
          <button onClick={onBack} style={{ border: '1px solid #ccd6eb', borderRadius: 10, padding: '8px 10px', background: 'white' }}>Back</button>
          <div style={{ fontWeight: 700, fontSize: 15 }}>Look up star</div>
          <button onClick={onOpenMetadata} style={{ border: '1px solid #ccd6eb', borderRadius: 10, padding: '8px 10px', background: canCompare ? '#eef4ff' : 'white' }}>Metadata</button>
        </div>
        <div style={{ color: '#667085', fontSize: 13, lineHeight: 1.35 }}>Browse by type, location, ID, and then observation date for that selected ID.</div>
      </div>

      <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 10, display: 'grid', gap: 8, width: 'calc(100% - 12px)', maxWidth: 'calc(100vw - 32px)', minWidth: 0, overflow: 'hidden', justifySelf: 'center' }}>
        <label style={fieldStyle}>
          <span style={{ fontSize: 13, color: '#667085' }}>Type</span>
          <select
            value={entityType}
            onChange={(e) => {
              setEntityType(e.target.value as 'gallery' | 'query')
              setResult(null)
              setSelectedEncounter('')
              setEncounterOptions([])
            }}
            style={controlStyle}
          >
            <option value="gallery">Gallery</option>
            <option value="query">Queries</option>
          </select>
        </label>

        <label style={fieldStyle}>
          <span style={{ fontSize: 13, color: '#667085' }}>Location filter</span>
          <select
            value={location}
            onChange={(e) => {
              setLocation(e.target.value)
              setResult(null)
              setSelectedEncounter('')
              setEncounterOptions([])
            }}
            style={controlStyle}
          >
            <option value="">All locations</option>
            {locationOptions.map((item) => (
              <option key={item} value={item}>
                {truncateLabel(item, 26)}
              </option>
            ))}
          </select>
        </label>

        <label style={fieldStyle}>
          <span style={{ fontSize: 13, color: '#667085' }}>ID</span>
          <select
            value={entityId}
            onChange={(e) => {
              setEntityId(e.target.value)
              setResult(null)
              setSelectedEncounter('')
            }}
            style={controlStyle}
          >
            <option value="">Select an ID</option>
            {idOptions.map((item) => (
              <option key={item} value={item}>
                {truncateLabel(item, 24)}
              </option>
            ))}
          </select>
        </label>

        <label style={fieldStyle}>
          <span style={{ fontSize: 13, color: '#667085' }}>Observation date</span>
          <select
            value={selectedEncounter}
            onChange={(e) => setSelectedEncounter(e.target.value)}
            disabled={!entityId || encounterOptions.length === 0}
            style={controlStyle}
          >
            <option value="">All observation dates</option>
            {encounterOptions.map((item) => (
              <option key={item.encounter} value={item.encounter}>
                {displayEncounterLabel(item)}
              </option>
            ))}
          </select>
        </label>

        <button
          onClick={() => void doLookup()}
          disabled={!entityId || loading}
          style={{ padding: 11, borderRadius: 10, background: '#2f6fed', color: 'white', border: '1px solid #2f6fed' }}
        >
          {loading ? 'Loading…' : 'Open selection'}
        </button>
      </div>

      {error && <div style={{ color: 'crimson', whiteSpace: 'pre-wrap' }}>{error}</div>}

      {result && (
        <>
          <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 8 }}>
            <div style={{ fontSize: 14 }}>Opened <strong>{result.entity_id}</strong></div>
            <div style={{ fontSize: 13, color: '#555' }}>{result.entity_type} • {selectedEncounter || 'all observation dates'} • {result.image_window.total} archive image{result.image_window.total === 1 ? '' : 's'} found • {result.image_window.items.length} loaded</div>
            {metadataRows.length > 0 && (
              <details>
                <summary>Metadata summary</summary>
                <div style={{ display: 'grid', gap: 4, marginTop: 8, fontSize: 13 }}>
                  {metadataRows.slice(0, 12).map(([key, value]) => (
                    <div key={key}><strong>{key}</strong>: {String(value)}</div>
                  ))}
                </div>
              </details>
            )}
          </div>

          <ZoomableImagePane compact title="Archive workspace" subtitle={activeArchiveImage?.label} src={activeArchiveImage?.fullres_url ?? activeArchiveImage?.preview_url} />
          <ArchiveImageStrip items={result.image_window.items} onSelect={(img) => onSelectArchiveImage(img, result.image_window.items)} selectedImageId={activeArchiveImage?.image_id ?? null} />
          {result.image_window.next_offset != null && <button onClick={() => void loadMore()}>Load more archive images</button>}
        </>
      )}
    </div>
  )
}
