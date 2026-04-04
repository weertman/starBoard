import { ZoomableImagePane } from '../components/ZoomableImagePane'
import { LocalImageQueue } from '../components/LocalImageQueue'
import type { ImageDescriptor, MegaStarCapabilityInfo, MegaStarLookupCandidate, MegaStarLookupResponse } from '../api/client'

function statusTone(status: MegaStarLookupResponse['status']) {
  if (status === 'ok') return { background: '#eff8ff', border: '#b2ddff', color: '#175cd3', title: 'MegaStar candidates ready' }
  if (status === 'weak') return { background: '#fffaeb', border: '#fedf89', color: '#b54708', title: 'MegaStar found weak candidates' }
  if (status === 'empty') return { background: '#f8f9fc', border: '#d0d5dd', color: '#344054', title: 'MegaStar found no candidates' }
  return { background: '#fef3f2', border: '#fecdca', color: '#b42318', title: 'MegaStar unavailable' }
}

function candidateSubtitle(candidate: MegaStarLookupCandidate) {
  const date = candidate.encounter_date || candidate.encounter || 'date unknown'
  return `${candidate.entity_id} • score ${candidate.retrieval_score.toFixed(3)} • ${date}`
}

function capabilityMessage(info: MegaStarCapabilityInfo) {
  if (info.enabled) return 'MegaStar is available for the currently selected local image.'
  const reason = info.reason ? info.reason.split('_').join(' ') : 'service unavailable'
  return `MegaStar is unavailable right now: ${reason}.`
}

export function ObservationWorkspace({
  localPreviews,
  selectedLocalIndex,
  selectLocalIndex,
  removeLocalAt,
  addFiles,
  archiveImages,
  selectedArchiveIndex,
  selectArchiveIndex,
  metadataReady,
  metadataSummary,
  submitDisabled,
  submitLabel,
  submitError,
  submitMessage,
  megastarEnabled,
  megastarInfo,
  megastarLoading,
  megastarResult,
  megastarError,
  onMegaStarLookup,
  onClearMegaStar,
  onRetryMegaStar,
  onCompareMegaStarCandidate,
  onOpenMegaStarCandidate,
  onSubmit,
  onBack,
  onOpenMetadata,
  onOpenLookup,
}: {
  localPreviews: { file: File; url: string }[]
  selectedLocalIndex: number
  selectLocalIndex: (index: number) => void
  removeLocalAt: (index: number) => void
  addFiles: (files: FileList | null) => void
  archiveImages: ImageDescriptor[]
  selectedArchiveIndex: number
  selectArchiveIndex: (index: number) => void
  metadataReady: boolean
  metadataSummary: string
  submitDisabled: boolean
  submitLabel: string
  submitError: string | null
  submitMessage: string | null
  megastarEnabled: boolean
  megastarInfo: MegaStarCapabilityInfo
  megastarLoading: boolean
  megastarResult: MegaStarLookupResponse | null
  megastarError: string | null
  onMegaStarLookup: () => void
  onClearMegaStar: () => void
  onRetryMegaStar: () => void
  onCompareMegaStarCandidate: (candidate: MegaStarLookupCandidate) => void
  onOpenMegaStarCandidate: (candidate: MegaStarLookupCandidate) => void
  onSubmit: () => void
  onBack: () => void
  onOpenMetadata: () => void
  onOpenLookup: () => void
}) {
  const localPreview = localPreviews[selectedLocalIndex]
  const archiveImage = archiveImages[selectedArchiveIndex]
  const compareMode = !!archiveImage && !!localPreview
  const megastarTone = megastarResult ? statusTone(megastarResult.status) : null
  const hasMegaStarState = !!megastarResult || !!megastarError || megastarLoading

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
          <button onClick={onBack} style={{ border: '1px solid #ccd6eb', borderRadius: 10, padding: '8px 10px', background: 'white' }}>Home</button>
          <div style={{ fontWeight: 700, fontSize: 15 }}>New observation</div>
          <button onClick={onOpenMetadata} style={{ border: metadataReady ? '1px solid #108a5b' : '1px solid #2f6fed', borderRadius: 10, padding: '8px 10px', background: metadataReady ? '#e9f7f0' : '#2f6fed', color: metadataReady ? '#108a5b' : 'white' }}>{metadataReady ? 'Metadata ready' : 'Metadata'}</button>
        </div>
        <div style={{ color: '#667085', fontSize: 13, lineHeight: 1.35 }}>Capture or select photos first. Compare with a looked-up star only when needed. Final submission happens below the local image workspace after metadata is marked ready.</div>
      </div>

      <div style={{ display: 'grid', gap: 8 }}>
        <label style={{ display: 'grid', gap: 4, background: 'white', border: '1px solid #d6dae1', borderRadius: 14, padding: 10 }}>
          <span style={{ fontSize: 13, color: '#667085' }}>Camera capture</span>
          <input type="file" accept="image/*" capture="environment" multiple onChange={(e) => addFiles(e.target.files)} />
        </label>
        <label style={{ display: 'grid', gap: 4, background: 'white', border: '1px solid #d6dae1', borderRadius: 14, padding: 10 }}>
          <span style={{ fontSize: 13, color: '#667085' }}>Photo library</span>
          <input type="file" accept="image/*" multiple onChange={(e) => addFiles(e.target.files)} />
        </label>
      </div>

      {!compareMode && <ZoomableImagePane compact title="Local image workspace" subtitle={localPreview?.file.name ?? 'Choose or capture a photo'} src={localPreview?.url} />}

      {compareMode && (
        <div style={{ display: 'grid', gap: 10 }}>
          <ZoomableImagePane compact title="Local image" subtitle={localPreview?.file.name} src={localPreview?.url} />
          <ZoomableImagePane compact title="Archive comparison" subtitle={archiveImage?.label} src={archiveImage?.fullres_url ?? archiveImage?.preview_url} />
        </div>
      )}

      <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
          <div>
            <strong>Local photos</strong>
            <div style={{ color: '#667085', fontSize: 13 }}>{localPreviews.length} loaded</div>
          </div>
          <button onClick={onOpenLookup} style={{ border: '1px solid #ccd6eb', borderRadius: 10, padding: '8px 10px', background: '#eef4ff' }}>Look up star</button>
        </div>
        <LocalImageQueue previews={localPreviews} onRemove={removeLocalAt} onSelect={selectLocalIndex} selectedIndex={selectedLocalIndex} />
        <div style={{ padding: 10, border: '1px solid #d6dae1', borderRadius: 12, background: megastarEnabled ? '#fbfcfe' : '#fff7ed', display: 'grid', gap: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontWeight: 600, fontSize: 14 }}>MegaStar lookup</div>
              <div style={{ color: '#667085', fontSize: 13 }}>{capabilityMessage(megastarInfo)}</div>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button onClick={onMegaStarLookup} disabled={!localPreview || megastarLoading || !megastarEnabled} style={{ border: '1px solid #7f56d9', borderRadius: 10, padding: '8px 10px', background: megastarLoading ? '#eaecf0' : megastarEnabled ? '#f4f3ff' : '#f2f4f7', color: megastarEnabled ? '#6941c6' : '#667085' }}>
                {megastarLoading ? 'MegaStar searching…' : 'MegaStar Lookup'}
              </button>
              {hasMegaStarState && (
                <button onClick={onClearMegaStar} disabled={megastarLoading} style={{ border: '1px solid #d0d5dd', borderRadius: 10, padding: '8px 10px', background: 'white' }}>
                  Clear
                </button>
              )}
            </div>
          </div>
          {!localPreview && <div style={{ color: '#667085', fontSize: 13 }}>Select a local image first to run MegaStar.</div>}
          {!megastarEnabled && megastarInfo.model_key && <div style={{ color: '#667085', fontSize: 12 }}>Configured model: {megastarInfo.model_key}</div>}
          {megastarError && (
            <div style={{ padding: 10, borderRadius: 12, border: '1px solid #fecdca', background: '#fef3f2', color: '#b42318', display: 'grid', gap: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 14 }}>MegaStar lookup failed</div>
              <div style={{ whiteSpace: 'pre-wrap', fontSize: 13 }}>{megastarError}</div>
              <div>
                <button onClick={onRetryMegaStar} disabled={!localPreview || megastarLoading || !megastarEnabled} style={{ border: '1px solid #f97066', borderRadius: 10, padding: '8px 10px', background: 'white' }}>
                  Retry
                </button>
              </div>
            </div>
          )}
          {megastarResult && megastarTone && (
            <div style={{ padding: 10, borderRadius: 12, border: `1px solid ${megastarTone.border}`, background: megastarTone.background, display: 'grid', gap: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 14, color: megastarTone.color }}>{megastarTone.title}</div>
                  <div style={{ color: '#475467', fontSize: 13 }}>{megastarResult.query_image_name} • {megastarResult.processing_ms} ms</div>
                </div>
                <button onClick={onRetryMegaStar} disabled={!localPreview || megastarLoading || !megastarEnabled} style={{ border: '1px solid #d0d5dd', borderRadius: 10, padding: '8px 10px', background: 'white' }}>
                  Retry
                </button>
              </div>
              {megastarResult.availability_reason && <div style={{ fontSize: 13, color: '#475467' }}>{megastarResult.availability_reason}</div>}
              {megastarResult.candidates.length === 0 && (
                <div style={{ fontSize: 13, color: '#475467' }}>No archive candidates were returned for this local image.</div>
              )}
              {megastarResult.candidates.length > 0 && (
                <div style={{ display: 'grid', gap: 10 }}>
                  {megastarResult.candidates.map((candidate) => (
                    <div key={`${candidate.entity_id}-${candidate.rank}`} style={{ border: '1px solid rgba(16,24,40,0.08)', borderRadius: 12, background: 'white', padding: 10, display: 'grid', gap: 8 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'flex-start' }}>
                        <div>
                          <div style={{ fontWeight: 600, fontSize: 14 }}>#{candidate.rank} {candidate.best_match_label || candidate.best_match_image.label}</div>
                          <div style={{ fontSize: 13, color: '#475467' }}>{candidateSubtitle(candidate)}</div>
                        </div>
                        <div style={{ fontSize: 12, color: '#667085' }}>{candidate.entity_type}</div>
                      </div>
                      <img src={candidate.best_match_image.preview_url} alt={candidate.best_match_label || candidate.best_match_image.label} style={{ width: '100%', borderRadius: 10 }} />
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        <button onClick={() => onCompareMegaStarCandidate(candidate)} style={{ border: '1px solid #2f6fed', borderRadius: 10, padding: '8px 10px', background: '#eef4ff', color: '#175cd3' }}>
                          Compare here
                        </button>
                        <button onClick={() => onOpenMegaStarCandidate(candidate)} style={{ border: '1px solid #d0d5dd', borderRadius: 10, padding: '8px 10px', background: 'white' }}>
                          Open in archive browser
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
        <div style={{ padding: 10, border: '1px solid #e4e7ec', borderRadius: 12, background: metadataReady ? '#f0faf5' : '#fbfcfe', display: 'grid', gap: 4 }}>
          <div style={{ fontWeight: 600, fontSize: 14 }}>{metadataReady ? 'Metadata ready' : 'Metadata required before submit'}</div>
          <div style={{ color: '#667085', fontSize: 13 }}>{metadataSummary}</div>
        </div>
        <button onClick={onSubmit} disabled={submitDisabled} style={{ padding: 12, borderRadius: 12, background: submitDisabled ? '#d6dae1' : '#2f6fed', color: submitDisabled ? '#667085' : 'white', border: '1px solid ' + (submitDisabled ? '#d6dae1' : '#2f6fed'), fontWeight: 600 }}>
          {submitLabel}
        </button>
        {submitMessage && <div style={{ color: 'green', fontSize: 13 }}>{submitMessage}</div>}
        {submitError && <div style={{ color: 'crimson', whiteSpace: 'pre-wrap', fontSize: 13 }}>{submitError}</div>}
      </div>

      {archiveImages.length > 0 && (
        <details style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12 }}>
          <summary style={{ fontWeight: 600, cursor: 'pointer' }}>Loaded archive comparison images ({archiveImages.length})</summary>
          <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
            {archiveImages.map((image, index) => (
              <button key={image.image_id} onClick={() => selectArchiveIndex(index)} style={{ textAlign: 'left', border: index === selectedArchiveIndex ? '2px solid #2f6fed' : '1px solid #ddd', borderRadius: 10, padding: 8, background: index === selectedArchiveIndex ? '#f5f8ff' : 'white' }}>
                <div style={{ fontSize: 12, marginBottom: 8 }}>{image.label}</div>
                <img src={image.preview_url} alt={image.label} style={{ width: '100%', borderRadius: 8 }} />
              </button>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}
