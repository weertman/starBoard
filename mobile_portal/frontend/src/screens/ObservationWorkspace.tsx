import { ZoomableImagePane } from '../components/ZoomableImagePane'
import { LocalImageQueue } from '../components/LocalImageQueue'
import type { ImageDescriptor, SubmissionResponse } from '../api/client'

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
  onSubmit: () => void
  onBack: () => void
  onOpenMetadata: () => void
  onOpenLookup: () => void
}) {
  const localPreview = localPreviews[selectedLocalIndex]
  const archiveImage = archiveImages[selectedArchiveIndex]
  const compareMode = !!archiveImage && !!localPreview

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
