import { useEffect, useMemo, useState } from 'react'

import {
  discoverBatchUpload,
  executeBatchUpload,
  getLocationSites,
  previewBatchServerPath,
  uploadBatchZip,
  type BatchUploadDiscoverRequest,
  type BatchUploadDiscoverResponse,
  type BatchUploadDiscoverRow,
  type BatchUploadExecuteResponse,
  type BatchUploadServerPathPreviewResponse,
  type LocationSite,
} from '../api/client'
import { LocationSiteMap } from '../components/LocationSiteMap'

type DiscoverMode = 'auto' | 'flat' | 'encounters' | 'grouped'
type TargetArchive = 'gallery' | 'query'
type BusyState = 'preflight' | 'upload' | 'discover' | 'execute'

type ZipStructurePreview = {
  status: 'valid' | 'invalid'
  resolvedMode: 'flat' | 'encounters' | 'grouped' | 'empty'
  importableImages: number
  rootEntries: string[]
  message: string
}

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

function badge(action: BatchUploadDiscoverRow['action']) {
  if (action === 'append_existing') return { label: 'Append', color: '#8a5a00', bg: '#fff4d6' }
  if (action === 'create_new') return { label: 'New', color: '#0b6b2b', bg: '#dbfbe6' }
  return { label: 'Skip', color: '#7a1c1c', bg: '#ffe3e3' }
}

const IMPORTABLE_IMAGE_EXTENSIONS = new Set([
  '.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.tif', '.tiff', '.bmp', '.dib', '.gif', '.webp', '.heic', '.heif', '.avif', '.orf',
])

function isImportableImageName(name: string) {
  const lower = name.toLowerCase()
  return Array.from(IMPORTABLE_IMAGE_EXTENSIONS).some((ext) => lower.endsWith(ext))
}

function isDatedFolderName(name: string) {
  return /^\d{1,2}_\d{1,2}_\d{2,4}(?:_|$)/.test(name)
}

function uniqueSorted(values: string[]) {
  return Array.from(new Set(values.filter(Boolean))).sort((a, b) => a.localeCompare(b))
}

async function readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  if (typeof file.arrayBuffer === 'function') return file.arrayBuffer()
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as ArrayBuffer)
    reader.onerror = () => reject(reader.error ?? new Error('Could not read selected file'))
    reader.readAsArrayBuffer(file)
  })
}

function readZipEntryNames(buffer: ArrayBuffer) {
  const view = new DataView(buffer)
  const decoder = new TextDecoder()
  let eocdOffset = -1
  for (let i = view.byteLength - 22; i >= Math.max(0, view.byteLength - 65557); i -= 1) {
    if (view.getUint32(i, true) === 0x06054b50) {
      eocdOffset = i
      break
    }
  }
  if (eocdOffset < 0) throw new Error('Could not read zip directory')
  const entryCount = view.getUint16(eocdOffset + 10, true)
  let cursor = view.getUint32(eocdOffset + 16, true)
  const names: string[] = []
  for (let i = 0; i < entryCount; i += 1) {
    if (cursor + 46 > view.byteLength || view.getUint32(cursor, true) !== 0x02014b50) {
      throw new Error('Could not read zip entries')
    }
    const nameLen = view.getUint16(cursor + 28, true)
    const extraLen = view.getUint16(cursor + 30, true)
    const commentLen = view.getUint16(cursor + 32, true)
    const nameStart = cursor + 46
    const nameEnd = nameStart + nameLen
    names.push(decoder.decode(new Uint8Array(buffer, nameStart, nameLen)))
    cursor = nameEnd + extraLen + commentLen
  }
  return names
}

function analyzeZipEntries(names: string[], requestedMode: DiscoverMode): ZipStructurePreview {
  const fileNames = names
    .map((name) => name.replace(/^\/+/, ''))
    .filter((name) => name && !name.endsWith('/') && !name.startsWith('__MACOSX/'))
  const importable = fileNames.filter(isImportableImageName)
  if (importable.length === 0) {
    return { status: 'invalid', resolvedMode: 'empty', importableImages: 0, rootEntries: [], message: 'No importable images found in this zip.' }
  }

  let partsList = importable.map((name) => name.split('/').filter(Boolean))
  const rootEntriesBeforeUnwrap = uniqueSorted(partsList.map((parts) => parts[0]))
  const hasRootImages = partsList.some((parts) => parts.length === 1)
  if (!hasRootImages && rootEntriesBeforeUnwrap.length === 1 && partsList.every((parts) => parts.length > 1)) {
    const unwrapped = partsList.map((parts) => parts.slice(1))
    if (!unwrapped.some((parts) => parts.length === 1)) {
      partsList = unwrapped
    }
  }

  const rootEntries = uniqueSorted(partsList.map((parts) => parts[0]))
  const hasImagesAtRoot = partsList.some((parts) => parts.length === 1)
  if (hasImagesAtRoot) {
    return {
      status: 'invalid',
      resolvedMode: 'empty',
      importableImages: importable.length,
      rootEntries,
      message: 'Images are at the zip root. Put images inside ID folders before upload.',
    }
  }

  const hasDatedSecondLevel = partsList.some((parts) => parts.length >= 3 && isDatedFolderName(parts[1]))
  const hasDatedThirdLevel = partsList.some((parts) => parts.length >= 4 && isDatedFolderName(parts[2]))
  const resolvedMode = hasDatedThirdLevel ? 'grouped' : hasDatedSecondLevel ? 'encounters' : 'flat'
  const modeMismatch = requestedMode !== 'auto' && requestedMode !== resolvedMode
  return {
    status: modeMismatch ? 'invalid' : 'valid',
    resolvedMode,
    importableImages: importable.length,
    rootEntries,
    message: modeMismatch
      ? `Zip looks like ${resolvedMode} structure, but Discovery mode is ${requestedMode}. Switch to Auto or the matching mode.`
      : 'Zip structure looks valid.',
  }
}

export function BatchUploadPage() {
  const [targetArchive, setTargetArchive] = useState<TargetArchive>('gallery')
  const [discoveryMode, setDiscoveryMode] = useState<DiscoverMode>('auto')
  const [idPrefix, setIdPrefix] = useState('')
  const [idSuffix, setIdSuffix] = useState('')
  const [flatEncounterDate, setFlatEncounterDate] = useState(new Date().toISOString().slice(0, 10))
  const [flatEncounterSuffix, setFlatEncounterSuffix] = useState('')
  const [batchLocation, setBatchLocation] = useState({ location: '', latitude: '', longitude: '' })
  const [knownSites, setKnownSites] = useState<LocationSite[]>([])
  const [showNewLocationInput, setShowNewLocationInput] = useState(false)
  const [pickingCoordinates, setPickingCoordinates] = useState(false)
  const [sourceMode, setSourceMode] = useState<'zip' | 'server_path'>('server_path')
  const [serverPath, setServerPath] = useState('')
  const [serverPathPreview, setServerPathPreview] = useState<BatchUploadServerPathPreviewResponse | null>(null)
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [zipPreview, setZipPreview] = useState<ZipStructurePreview | null>(null)
  const [uploadToken, setUploadToken] = useState<string | null>(null)
  const [uploadInfo, setUploadInfo] = useState<{ file_count: number; root_entries: string[] } | null>(null)
  const [discoverResponse, setDiscoverResponse] = useState<BatchUploadDiscoverResponse | null>(null)
  const [executeResponse, setExecuteResponse] = useState<BatchUploadExecuteResponse | null>(null)
  const [selectedRowIds, setSelectedRowIds] = useState<string[]>([])
  const [busy, setBusy] = useState<BusyState | null>(null)
  const [operation, setOperation] = useState<{ label: string; startedAt: number } | null>(null)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [successReadout, setSuccessReadout] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [planStale, setPlanStale] = useState(false)

  const rows = discoverResponse?.rows ?? []
  const selectedRows = useMemo(() => rows.filter((row) => selectedRowIds.includes(row.row_id)), [rows, selectedRowIds])
  const showFlatEncounterControls = discoveryMode === 'auto' || discoveryMode === 'flat'
  const todayIso = new Date().toISOString().slice(0, 10)
  const canUploadZip = Boolean(zipFile && zipPreview?.status === 'valid')
  const canDiscover = sourceMode === 'server_path' ? serverPathPreview?.path === serverPath.trim() : Boolean(uploadToken)

  useEffect(() => {
    void (async () => {
      try {
        const response = await getLocationSites()
        setKnownSites(response.sites)
      } catch {
        setKnownSites([])
      }
    })()
  }, [])

  useEffect(() => {
    if (!operation) return undefined
    setElapsedSeconds(0)
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - operation.startedAt) / 1000))
    }, 500)
    return () => window.clearInterval(timer)
  }, [operation])

  function beginOperation(kind: BusyState, label: string) {
    setBusy(kind)
    setOperation({ label, startedAt: Date.now() })
    setElapsedSeconds(0)
    setSuccessReadout(null)
  }

  function endOperation() {
    setBusy(null)
    setOperation(null)
  }

  function invalidateDiscoveredPlan() {
    if (discoverResponse) {
      setDiscoverResponse(null)
      setExecuteResponse(null)
      setSelectedRowIds([])
      setPlanStale(true)
    }
  }

  function updateSavedLocation(value: string) {
    if (value === '__new__') {
      setShowNewLocationInput(true)
      setBatchLocation((cur) => ({ ...cur, location: '' }))
      invalidateDiscoveredPlan()
      return
    }
    setShowNewLocationInput(false)
    setBatchLocation((cur) => ({ ...cur, location: value }))
    invalidateDiscoveredPlan()
  }

  function handleMapPick(latitude: number, longitude: number) {
    setBatchLocation((cur) => ({
      ...cur,
      latitude: latitude.toFixed(6),
      longitude: longitude.toFixed(6),
    }))
    setPickingCoordinates(false)
    invalidateDiscoveredPlan()
  }

  function handleZipFileChange(file: File | null) {
    setZipFile(file)
    setZipPreview(null)
    setUploadToken(null)
    setUploadInfo(null)
    setDiscoverResponse(null)
    setExecuteResponse(null)
    setSelectedRowIds([])
  }

  async function handleTestZipStructure() {
    if (!zipFile) return
    beginOperation('preflight', 'Testing zip structure')
    setError(null)
    try {
      const names = readZipEntryNames(await readFileAsArrayBuffer(zipFile))
      const result = analyzeZipEntries(names, discoveryMode)
      setZipPreview(result)
      setSuccessReadout(result.status === 'valid' ? 'Zip preflight complete.' : null)
    } catch (err) {
      setZipPreview({
        status: 'invalid',
        resolvedMode: 'empty',
        importableImages: 0,
        rootEntries: [],
        message: err instanceof Error ? err.message : String(err),
      })
    } finally {
      endOperation()
    }
  }

  async function handleUpload() {
    if (!zipFile) return
    beginOperation('upload', 'Preparing preview source')
    setError(null)
    setDiscoverResponse(null)
    setExecuteResponse(null)
    try {
      const result = await uploadBatchZip(zipFile)
      setUploadToken(result.upload_token)
      setUploadInfo({ file_count: result.file_count, root_entries: result.root_entries })
      setSuccessReadout(`Preview source ready: ${result.file_count} file(s) available for review.`)
    } catch (err) {
      setError(String(err))
    } finally {
      endOperation()
    }
  }

  async function handlePreviewServerPath() {
    const trimmed = serverPath.trim()
    if (!trimmed) return
    beginOperation('discover', 'Previewing server path')
    setError(null)
    setDiscoverResponse(null)
    setExecuteResponse(null)
    try {
      const result = await previewBatchServerPath({ path: trimmed, discovery_mode: discoveryMode })
      setServerPathPreview(result)
      setPlanStale(false)
      setSuccessReadout(`Server path preview complete: ${result.importable_images} importable image(s).`)
    } catch (err) {
      setServerPathPreview(null)
      setError(String(err))
    } finally {
      endOperation()
    }
  }

  async function handleDiscover() {
    if (!canDiscover) return
    beginOperation('discover', 'Previewing IDs and metadata')
    setError(null)
    setExecuteResponse(null)
    try {
      const request: BatchUploadDiscoverRequest = {
        target_archive: targetArchive,
        discovery_mode: discoveryMode,
        id_prefix: idPrefix,
        id_suffix: idSuffix,
        flat_encounter_date: flatEncounterDate,
        flat_encounter_suffix: flatEncounterSuffix,
        batch_location: batchLocation,
        import_source: sourceMode === 'server_path'
          ? { type: 'server_path', path: serverPath.trim() }
          : { type: 'uploaded_bundle', upload_token: uploadToken ?? '' },
      }
      const result = await discoverBatchUpload(request)
      setDiscoverResponse(result)
      setSelectedRowIds(result.rows.map((row) => row.row_id))
      setPlanStale(false)
      setSuccessReadout(`Preview ready: ${result.summary.detected_rows} row(s), ${result.summary.detected_ids} ID(s), ${result.summary.total_images} image(s).`)
    } catch (err) {
      setError(String(err))
    } finally {
      endOperation()
    }
  }

  async function handleExecute() {
    if (!discoverResponse || selectedRowIds.length === 0) return
    const existingTargets = selectedRows.filter((row) => row.target_exists || row.action === 'append_existing')
    if (existingTargets.length > 0) {
      const preview = existingTargets.slice(0, 10).map((row) => `• ${row.transformed_target_id}`).join('\n')
      const more = existingTargets.length > 10 ? `\n... and ${existingTargets.length - 10} more` : ''
      const ok = window.confirm(
        `The following IDs already exist and selected rows will append images to them:\n\n${preview}${more}\n\nSubmit these IDs for upload?`,
      )
      if (!ok) return
    }
    beginOperation('execute', 'Submitting selected IDs')
    setError(null)
    try {
      const result = await executeBatchUpload({
        plan_id: discoverResponse.plan_id,
        accepted_row_ids: selectedRowIds,
      })
      setExecuteResponse(result)
      setSuccessReadout(`Upload ${result.status}: ${result.summary.accepted_images} accepted image(s), ${result.summary.executed_rows} submitted row(s).`)
    } catch (err) {
      setError(String(err))
    } finally {
      endOperation()
    }
  }

  function toggleRow(rowId: string) {
    setSelectedRowIds((current) => current.includes(rowId) ? current.filter((id) => id !== rowId) : [...current, rowId])
  }

  function toggleAllRows(checked: boolean) {
    setSelectedRowIds(checked ? rows.map((row) => row.row_id) : [])
  }

  return (
    <main style={{ maxWidth: 1180, margin: '0 auto', padding: 18, fontFamily: 'system-ui, sans-serif', color: '#152033', background: '#f7f9fc', minHeight: '100vh' }}>
      <div style={{ display: 'grid', gap: 16 }}>
        <section style={card}>
          <h1 style={{ marginTop: 0 }}>Batch Upload</h1>
          <p style={{ marginTop: 0, color: '#516070' }}>
            Browser workflow for desktop-style multi-ID batch upload into Gallery or Queries.
          </p>
          <details style={{ margin: '10px 0 14px', padding: 12, borderRadius: 8, background: '#f8fafc', border: '1px solid #d7deea' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 700 }}>How to use Batch Upload</summary>
            <ul style={{ margin: '10px 0 0 20px', padding: 0, color: '#405064' }}>
              <li>Choose a source: use a server folder path for local files already on this machine, or prepare a zip when the files are elsewhere.</li>
              <li>Pick the target archive, discovery mode, optional ID prefix/suffix, and location metadata before previewing.</li>
              <li>Preview IDs and metadata before writing anything to the archive.</li>
              <li>Review the detected IDs, encounters, image counts, target actions, warnings, and selected rows.</li>
              <li>Submit selected IDs only after the review table looks correct.</li>
            </ul>
          </details>
          <div style={{ margin: '10px 0 14px', padding: 12, borderRadius: 8, background: '#f8fafc', border: '1px solid #d7deea' }}>
            <b>Accepted source layouts</b>
            <ul style={{ margin: '8px 0 0 20px', padding: 0, color: '#405064' }}>
              <li><code>ID / images</code> for flat imports.</li>
              <li><code>ID / date / images</code> for one individual with encounter folders.</li>
              <li><code>group / ID / date / images</code> for grouped field exports.</li>
              <li>Zipped folders with one wrapper directory are okay; Auto mode unwraps the wrapper before preview.</li>
            </ul>
          </div>
          <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
            <label>
              <div>Target archive</div>
              <select aria-label="Target archive" value={targetArchive} onChange={(e) => { setTargetArchive(e.target.value as TargetArchive); invalidateDiscoveredPlan() }} style={input}>
                <option value="gallery">Gallery</option>
                <option value="query">Queries</option>
              </select>
            </label>
            <label>
              <div>Discovery mode</div>
              <select aria-label="Discovery mode" value={discoveryMode} onChange={(e) => { setDiscoveryMode(e.target.value as DiscoverMode); setZipPreview(null); invalidateDiscoveredPlan() }} style={input}>
                <option value="auto">Auto</option>
                <option value="flat">Flat (ID / images)</option>
                <option value="encounters">With Encounters (ID / date / images)</option>
                <option value="grouped">Grouped (group / ID / date / images)</option>
              </select>
            </label>
            <label>
              <div>ID prefix</div>
              <input aria-label="ID prefix" value={idPrefix} onChange={(e) => { setIdPrefix(e.target.value); invalidateDiscoveredPlan() }} style={input} />
            </label>
            <label>
              <div>ID suffix</div>
              <input aria-label="ID suffix" value={idSuffix} onChange={(e) => { setIdSuffix(e.target.value); invalidateDiscoveredPlan() }} style={input} />
            </label>
            {showFlatEncounterControls && (
              <>
                <label>
                  <div>Flat encounter date</div>
                  <input aria-label="Flat encounter date" type="date" value={flatEncounterDate} onChange={(e) => { setFlatEncounterDate(e.target.value); invalidateDiscoveredPlan() }} style={input} />
                </label>
                <label>
                  <div>Flat encounter suffix</div>
                  <input aria-label="Flat encounter suffix" value={flatEncounterSuffix} onChange={(e) => { setFlatEncounterSuffix(e.target.value); invalidateDiscoveredPlan() }} style={input} />
                </label>
              </>
            )}
          </div>
          {showFlatEncounterControls && flatEncounterDate === todayIso && (
            <div style={{ marginTop: 8, color: '#8a5a00', fontStyle: 'italic' }}>Today's date selected</div>
          )}
        </section>

        <section style={card}>
          <h2 style={{ marginTop: 0 }}>Location</h2>
          <div style={{ display: 'grid', gap: 12 }}>
            <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
              <label>
                <div>Saved locations</div>
                <select
                  aria-label="Saved locations"
                  style={input}
                  value={showNewLocationInput ? '__new__' : batchLocation.location}
                  onChange={(e) => updateSavedLocation(e.target.value)}
                >
                  <option value="__new__">Add new…</option>
                  <option value="">— choose —</option>
                  {knownSites.map((site) => (
                    <option key={`saved-location-${site.name}`} value={site.name}>{site.name}</option>
                  ))}
                </select>
              </label>
              <div style={{ display: 'flex', alignItems: 'end' }}>
                <button type="button" onClick={() => setShowNewLocationInput(true)} style={{ padding: '8px 12px' }}>Add new location</button>
              </div>
              {showNewLocationInput && (
                <label style={{ gridColumn: '1 / -1' }}>
                  <div>Location</div>
                  <input aria-label="Location" value={batchLocation.location} onChange={(e) => { setBatchLocation((cur) => ({ ...cur, location: e.target.value })); invalidateDiscoveredPlan() }} style={input} />
                </label>
              )}
              <label>
                <div>Latitude</div>
                <input aria-label="Latitude" type="number" value={batchLocation.latitude} onChange={(e) => { setBatchLocation((cur) => ({ ...cur, latitude: e.target.value })); invalidateDiscoveredPlan() }} style={input} />
              </label>
              <label>
                <div>Longitude</div>
                <input aria-label="Longitude" type="number" value={batchLocation.longitude} onChange={(e) => { setBatchLocation((cur) => ({ ...cur, longitude: e.target.value })); invalidateDiscoveredPlan() }} style={input} />
              </label>
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
                selectedLatitude={batchLocation.latitude ? Number(batchLocation.latitude) : undefined}
                selectedLongitude={batchLocation.longitude ? Number(batchLocation.longitude) : undefined}
                picking={pickingCoordinates}
                onPick={handleMapPick}
              />
            </div>
          </div>
        </section>

        <section style={card}>
          <h2 style={{ marginTop: 0 }}>1. Choose source</h2>
          <p style={{ marginTop: 0, color: '#516070' }}>
            Select the source first. starBoard will preview the detected IDs, encounters, image counts, and metadata before anything is pushed into the archive.
          </p>
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
            <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <input
                aria-label="Use zip upload"
                type="radio"
                checked={sourceMode === 'zip'}
                onChange={() => { setSourceMode('zip'); setServerPathPreview(null); invalidateDiscoveredPlan() }}
              />
              Upload zip
            </label>
            <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <input
                aria-label="Select server folder path source"
                type="radio"
                checked={sourceMode === 'server_path'}
                onChange={() => { setSourceMode('server_path'); invalidateDiscoveredPlan() }}
              />
              Server folder path
            </label>
          </div>
          {sourceMode === 'zip' && (
            <>
              <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                <label>
                  <span style={{ position: 'absolute', left: -10000 }}>Source zip bundle</span>
                  <input aria-label="Source zip bundle" type="file" accept=".zip,application/zip" onChange={(e) => handleZipFileChange(e.target.files?.[0] ?? null)} />
                </label>
                <button onClick={() => void handleTestZipStructure()} disabled={!zipFile || busy !== null} style={{ padding: '8px 12px' }}>
                  {busy === 'preflight' ? 'Testing…' : 'Test zip structure'}
                </button>
                <button onClick={() => void handleUpload()} disabled={!canUploadZip || busy !== null} style={{ padding: '8px 12px' }}>
                  {busy === 'upload' ? 'Preparing preview…' : 'Prepare zip for preview'}
                </button>
              </div>
              {zipPreview && (
                <div style={{ marginTop: 12, color: zipPreview.status === 'valid' ? '#0b6b2b' : '#7a1c1c' }}>
                  <b>{zipPreview.message}</b><br />
                  Resolved mode: <b>{zipPreview.resolvedMode}</b>; Importable images: <b>{zipPreview.importableImages}</b>; root entries: {zipPreview.rootEntries.join(', ') || 'none'}
                </div>
              )}
              {uploadInfo && uploadToken && (
                <div style={{ marginTop: 12, color: '#24354d' }}>
                  <div><b>Upload token:</b> <code>{uploadToken}</code></div>
                  <div><b>Files ready for preview:</b> {uploadInfo.file_count}</div>
                  <div><b>Root entries:</b> {uploadInfo.root_entries.join(', ') || 'none'}</div>
                </div>
              )}
            </>
          )}
          {sourceMode === 'server_path' && (
            <div style={{ display: 'grid', gap: 10 }}>
              <label>
                <div>Server folder path</div>
                <input
                  aria-label="Server folder path"
                  value={serverPath}
                  placeholder="/home/weertman/path/to/batch-folder"
                  onChange={(e) => { setServerPath(e.target.value); setServerPathPreview(null); invalidateDiscoveredPlan() }}
                  style={input}
                />
              </label>
              <div>
                <button onClick={() => void handlePreviewServerPath()} disabled={!serverPath.trim() || busy !== null} style={{ padding: '8px 12px' }}>
                  {busy === 'discover' ? 'Previewing…' : 'Preview server path'}
                </button>
              </div>
              {serverPathPreview && (
                <div style={{ color: '#24354d' }}>
                  <b>Server path ready:</b> {serverPathPreview.path}<br />
                  Resolved mode: <b>{serverPathPreview.resolved_discovery_mode}</b>; importable images: <b>{serverPathPreview.importable_images}</b>; entries: {serverPathPreview.immediate_entries.join(', ') || 'none'}
                </div>
              )}
            </div>
          )}
        </section>

        <section style={card}>
          <h2 style={{ marginTop: 0 }}>2. Preview IDs and metadata</h2>
          <p style={{ marginTop: 0, color: '#516070' }}>
            Build a review table from the selected source. This still does not write to the archive.
          </p>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <button onClick={() => void handleDiscover()} disabled={!canDiscover || busy !== null} style={{ padding: '8px 12px' }}>
              {busy === 'discover' ? 'Previewing…' : 'Preview IDs and metadata'}
            </button>
            {discoverResponse && (
              <span style={{ color: '#516070' }}>
                Resolved mode: <b>{discoverResponse.resolved_discovery_mode}</b>
              </span>
            )}
          </div>
          {discoverResponse && (
            <div style={{ marginTop: 12, display: 'flex', gap: 18, flexWrap: 'wrap', color: '#24354d' }}>
              <span>Rows: <b>{discoverResponse.summary.detected_rows}</b></span>
              <span>IDs: <b>{discoverResponse.summary.detected_ids}</b></span>
              <span>Images: <b>{discoverResponse.summary.total_images}</b></span>
              <span>New IDs: <b>{discoverResponse.summary.new_ids}</b></span>
              <span>Existing IDs: <b>{discoverResponse.summary.existing_ids}</b></span>
            </div>
          )}
          {discoverResponse && (discoverResponse.warnings.length > 0 || discoverResponse.errors.length > 0 || rows.some((row) => row.warnings.length > 0)) && (
            <div style={{ marginTop: 12, color: '#7a4f00' }}>
              {[...discoverResponse.warnings, ...discoverResponse.errors, ...rows.flatMap((row) => row.warnings)].map((warning, idx) => (
                <div key={`${warning.code}-${idx}`}>{warning.message}</div>
              ))}
            </div>
          )}
        </section>

        {planStale && (
          <section style={{ ...card, borderColor: '#d7a84a', background: '#fff8e7', color: '#6b4b00' }}>
            Settings changed. Preview IDs and metadata again before submitting IDs for upload.
          </section>
        )}

        {operation && (
          <section role="status" style={{ ...card, borderColor: '#b6d8ff', background: '#eaf4ff', color: '#163c63' }}>
            <b>{operation.label}</b> — Elapsed: {elapsedSeconds}s
          </section>
        )}

        {successReadout && !operation && (
          <section style={{ ...card, borderColor: '#9bd8ad', background: '#effaf1', color: '#0b6b2b' }}>
            <b>{successReadout}</b>
          </section>
        )}

        {error && (
          <section style={{ ...card, borderColor: '#e29a9a', background: '#fff5f5', color: '#7a1c1c' }}>
            <b>Error:</b> {error}
          </section>
        )}

        {rows.length > 0 && (
          <section style={card}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <h2 style={{ margin: 0 }}>Review selected IDs and metadata</h2>
              <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={selectedRowIds.length === rows.length && rows.length > 0}
                  onChange={(e) => toggleAllRows(e.target.checked)}
                />
                Select all
              </label>
            </div>
            <div style={{ overflowX: 'auto', marginTop: 12 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
                <thead>
                  <tr style={{ background: '#eef3fb' }}>
                    <th style={{ textAlign: 'left', padding: 8 }}></th>
                    <th style={{ textAlign: 'left', padding: 8 }}>Detected ID</th>
                    <th style={{ textAlign: 'left', padding: 8 }}>Target ID</th>
                    <th style={{ textAlign: 'left', padding: 8 }}>Action</th>
                    <th style={{ textAlign: 'left', padding: 8 }}>Encounter</th>
                    <th style={{ textAlign: 'left', padding: 8 }}>Images</th>
                    <th style={{ textAlign: 'left', padding: 8 }}>Sample files</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => {
                    const b = badge(row.action)
                    return (
                      <tr key={row.row_id} style={{ borderTop: '1px solid #e2e8f0' }}>
                        <td style={{ padding: 8 }}>
                          <input type="checkbox" checked={selectedRowIds.includes(row.row_id)} onChange={() => toggleRow(row.row_id)} />
                        </td>
                        <td style={{ padding: 8 }}>{row.original_detected_id}</td>
                        <td style={{ padding: 8 }}><code>{row.transformed_target_id}</code></td>
                        <td style={{ padding: 8 }}>
                          <span style={{ display: 'inline-block', padding: '3px 8px', borderRadius: 999, background: b.bg, color: b.color, fontWeight: 600 }}>{b.label}</span>
                        </td>
                        <td style={{ padding: 8 }}>{row.encounter_folder_name ?? row.encounter_date ?? '—'}</td>
                        <td style={{ padding: 8 }}>{row.image_count}</td>
                        <td style={{ padding: 8 }}>{row.sample_labels.join(', ') || '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}

        <section style={card}>
            <h2 style={{ marginTop: 0 }}>3. Submit IDs</h2>
            <p style={{ marginTop: 0, color: '#516070' }}>
              Only this final action submits the selected IDs and writes their selected rows into the archive.
            </p>
            <div style={{ color: '#24354d', marginBottom: 12 }}>
              Selected rows: <b>{selectedRows.length}</b> / {rows.length}
            </div>
            <button onClick={() => void handleExecute()} disabled={selectedRows.length === 0 || busy !== null} style={{ padding: '8px 12px' }}>
              {busy === 'execute' ? 'Submitting…' : 'Submit selected IDs for upload'}
            </button>
          </section>

        {executeResponse && (
          <section style={card}>
            <h2 style={{ marginTop: 0 }}>Result</h2>
            <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
              <span>Status: <b>{executeResponse.status}</b></span>
              <span>Batch ID: <code>{executeResponse.batch_id}</code></span>
              <span>Executed rows: <b>{executeResponse.summary.executed_rows}</b></span>
              <span>Created IDs: <b>{executeResponse.summary.created_ids}</b></span>
              <span>Appended IDs: <b>{executeResponse.summary.appended_ids}</b></span>
              <span>Accepted images: <b>{executeResponse.summary.accepted_images}</b></span>
            </div>
            <div style={{ marginTop: 10 }}>{executeResponse.message}</div>
            {executeResponse.rows.length > 0 && (
              <div style={{ marginTop: 14, overflowX: 'auto' }}>
                <h3 style={{ marginBottom: 8 }}>Row results</h3>
                <table aria-label="Batch upload row results" style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
                  <thead>
                    <tr style={{ background: '#eef3fb' }}>
                      <th style={{ textAlign: 'left', padding: 8 }}>Target ID</th>
                      <th style={{ textAlign: 'left', padding: 8 }}>Action</th>
                      <th style={{ textAlign: 'left', padding: 8 }}>Encounter</th>
                      <th style={{ textAlign: 'left', padding: 8 }}>Accepted</th>
                      <th style={{ textAlign: 'left', padding: 8 }}>Skipped</th>
                      <th style={{ textAlign: 'left', padding: 8 }}>Errors</th>
                    </tr>
                  </thead>
                  <tbody>
                    {executeResponse.rows.map((row) => (
                      <tr key={row.row_id} style={{ borderTop: '1px solid #e2e8f0' }}>
                        <td style={{ padding: 8 }}>{row.target_id}</td>
                        <td style={{ padding: 8 }}>{row.action}</td>
                        <td style={{ padding: 8 }}>{row.encounter_folder}</td>
                        <td style={{ padding: 8 }}>{row.accepted_images}</td>
                        <td style={{ padding: 8 }}>{row.skipped_images}</td>
                        <td style={{ padding: 8 }}>{row.errors.map((err) => err.message).join('; ') || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}
      </div>
    </main>
  )
}
