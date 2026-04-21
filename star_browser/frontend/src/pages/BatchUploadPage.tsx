import { useMemo, useState } from 'react'

import {
  discoverBatchUpload,
  executeBatchUpload,
  uploadBatchZip,
  type BatchUploadDiscoverRequest,
  type BatchUploadDiscoverResponse,
  type BatchUploadDiscoverRow,
  type BatchUploadExecuteResponse,
} from '../api/client'

type DiscoverMode = 'auto' | 'flat' | 'encounters' | 'grouped'
type TargetArchive = 'gallery' | 'query'

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

export function BatchUploadPage() {
  const [targetArchive, setTargetArchive] = useState<TargetArchive>('gallery')
  const [discoveryMode, setDiscoveryMode] = useState<DiscoverMode>('auto')
  const [idPrefix, setIdPrefix] = useState('')
  const [idSuffix, setIdSuffix] = useState('')
  const [flatEncounterDate, setFlatEncounterDate] = useState(new Date().toISOString().slice(0, 10))
  const [flatEncounterSuffix, setFlatEncounterSuffix] = useState('')
  const [batchLocation, setBatchLocation] = useState({ location: '', latitude: '', longitude: '' })
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [uploadToken, setUploadToken] = useState<string | null>(null)
  const [uploadInfo, setUploadInfo] = useState<{ file_count: number; root_entries: string[] } | null>(null)
  const [discoverResponse, setDiscoverResponse] = useState<BatchUploadDiscoverResponse | null>(null)
  const [executeResponse, setExecuteResponse] = useState<BatchUploadExecuteResponse | null>(null)
  const [selectedRowIds, setSelectedRowIds] = useState<string[]>([])
  const [busy, setBusy] = useState<'upload' | 'discover' | 'execute' | null>(null)
  const [error, setError] = useState<string | null>(null)

  const rows = discoverResponse?.rows ?? []
  const selectedRows = useMemo(() => rows.filter((row) => selectedRowIds.includes(row.row_id)), [rows, selectedRowIds])

  async function handleUpload() {
    if (!zipFile) return
    setBusy('upload')
    setError(null)
    setDiscoverResponse(null)
    setExecuteResponse(null)
    try {
      const result = await uploadBatchZip(zipFile)
      setUploadToken(result.upload_token)
      setUploadInfo({ file_count: result.file_count, root_entries: result.root_entries })
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(null)
    }
  }

  async function handleDiscover() {
    if (!uploadToken) return
    setBusy('discover')
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
        import_source: { type: 'uploaded_bundle', upload_token: uploadToken },
      }
      const result = await discoverBatchUpload(request)
      setDiscoverResponse(result)
      setSelectedRowIds(result.rows.map((row) => row.row_id))
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(null)
    }
  }

  async function handleExecute() {
    if (!discoverResponse || selectedRowIds.length === 0) return
    setBusy('execute')
    setError(null)
    try {
      const result = await executeBatchUpload({
        plan_id: discoverResponse.plan_id,
        accepted_row_ids: selectedRowIds,
      })
      setExecuteResponse(result)
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(null)
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
          <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
            <label>
              <div>Target archive</div>
              <select value={targetArchive} onChange={(e) => setTargetArchive(e.target.value as TargetArchive)} style={input}>
                <option value="gallery">Gallery</option>
                <option value="query">Queries</option>
              </select>
            </label>
            <label>
              <div>Discovery mode</div>
              <select value={discoveryMode} onChange={(e) => setDiscoveryMode(e.target.value as DiscoverMode)} style={input}>
                <option value="auto">Auto</option>
                <option value="flat">Flat (ID / images)</option>
                <option value="encounters">With Encounters (ID / date / images)</option>
                <option value="grouped">Grouped (group / ID / date / images)</option>
              </select>
            </label>
            <label>
              <div>ID prefix</div>
              <input value={idPrefix} onChange={(e) => setIdPrefix(e.target.value)} style={input} />
            </label>
            <label>
              <div>ID suffix</div>
              <input value={idSuffix} onChange={(e) => setIdSuffix(e.target.value)} style={input} />
            </label>
            <label>
              <div>Flat encounter date</div>
              <input type="date" value={flatEncounterDate} onChange={(e) => setFlatEncounterDate(e.target.value)} style={input} />
            </label>
            <label>
              <div>Flat encounter suffix</div>
              <input value={flatEncounterSuffix} onChange={(e) => setFlatEncounterSuffix(e.target.value)} style={input} />
            </label>
            <label>
              <div>Batch location</div>
              <input value={batchLocation.location} onChange={(e) => setBatchLocation((cur) => ({ ...cur, location: e.target.value }))} style={input} />
            </label>
            <label>
              <div>Latitude</div>
              <input value={batchLocation.latitude} onChange={(e) => setBatchLocation((cur) => ({ ...cur, latitude: e.target.value }))} style={input} />
            </label>
            <label>
              <div>Longitude</div>
              <input value={batchLocation.longitude} onChange={(e) => setBatchLocation((cur) => ({ ...cur, longitude: e.target.value }))} style={input} />
            </label>
          </div>
        </section>

        <section style={card}>
          <h2 style={{ marginTop: 0 }}>1. Upload source bundle</h2>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <input type="file" accept=".zip,application/zip" onChange={(e) => setZipFile(e.target.files?.[0] ?? null)} />
            <button onClick={() => void handleUpload()} disabled={!zipFile || busy !== null} style={{ padding: '8px 12px' }}>
              {busy === 'upload' ? 'Uploading…' : 'Upload zip'}
            </button>
          </div>
          {uploadInfo && uploadToken && (
            <div style={{ marginTop: 12, color: '#24354d' }}>
              <div><b>Upload token:</b> <code>{uploadToken}</code></div>
              <div><b>Files staged:</b> {uploadInfo.file_count}</div>
              <div><b>Root entries:</b> {uploadInfo.root_entries.join(', ') || 'none'}</div>
            </div>
          )}
        </section>

        <section style={card}>
          <h2 style={{ marginTop: 0 }}>2. Discover IDs</h2>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            <button onClick={() => void handleDiscover()} disabled={!uploadToken || busy !== null} style={{ padding: '8px 12px' }}>
              {busy === 'discover' ? 'Discovering…' : 'Discover IDs'}
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
        </section>

        {error && (
          <section style={{ ...card, borderColor: '#e29a9a', background: '#fff5f5', color: '#7a1c1c' }}>
            <b>Error:</b> {error}
          </section>
        )}

        {rows.length > 0 && (
          <section style={card}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <h2 style={{ margin: 0 }}>3. Preview plan</h2>
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

        {discoverResponse && (
          <section style={card}>
            <h2 style={{ marginTop: 0 }}>4. Execute</h2>
            <div style={{ color: '#24354d', marginBottom: 12 }}>
              Selected rows: <b>{selectedRows.length}</b> / {rows.length}
            </div>
            <button onClick={() => void handleExecute()} disabled={selectedRows.length === 0 || busy !== null} style={{ padding: '8px 12px' }}>
              {busy === 'execute' ? 'Executing…' : 'Start batch upload'}
            </button>
          </section>
        )}

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
          </section>
        )}
      </div>
    </main>
  )
}
