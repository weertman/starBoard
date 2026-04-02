import { ZoomableImagePane } from '../components/ZoomableImagePane'
import { LocalImageQueue } from '../components/LocalImageQueue'
import type { ImageDescriptor } from '../api/client'

export function ObservationWorkspace({
  localPreviews,
  selectedLocalIndex,
  selectLocalIndex,
  removeLocalAt,
  addFiles,
  archiveImages,
  selectedArchiveIndex,
  selectArchiveIndex,
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
  onBack: () => void
  onOpenMetadata: () => void
  onOpenLookup: () => void
}) {
  const localPreview = localPreviews[selectedLocalIndex]
  const archiveImage = archiveImages[selectedArchiveIndex]
  const compareMode = !!archiveImage && !!localPreview

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
          <button onClick={onBack} style={{ border: '1px solid #ccd6eb', borderRadius: 10, padding: '8px 10px', background: 'white' }}>Home</button>
          <div style={{ fontWeight: 700 }}>New observation</div>
          <button onClick={onOpenMetadata} style={{ border: '1px solid #2f6fed', borderRadius: 10, padding: '8px 10px', background: '#2f6fed', color: 'white' }}>Metadata</button>
        </div>
        <div style={{ color: '#667085', fontSize: 13 }}>Capture/select photos first. Compare with a looked-up star only when needed.</div>
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <label style={{ flex: 1, display: 'grid', gap: 4, background: 'white', border: '1px solid #d6dae1', borderRadius: 14, padding: 10 }}>
          <span style={{ fontSize: 13, color: '#667085' }}>Camera capture</span>
          <input type="file" accept="image/*" capture="environment" multiple onChange={(e) => addFiles(e.target.files)} />
        </label>
        <label style={{ flex: 1, display: 'grid', gap: 4, background: 'white', border: '1px solid #d6dae1', borderRadius: 14, padding: 10 }}>
          <span style={{ fontSize: 13, color: '#667085' }}>Photo library</span>
          <input type="file" accept="image/*" multiple onChange={(e) => addFiles(e.target.files)} />
        </label>
      </div>

      {!compareMode && <ZoomableImagePane title="Local image workspace" subtitle={localPreview?.file.name ?? 'Choose or capture a photo'} src={localPreview?.url} />}

      {compareMode && (
        <div style={{ display: 'grid', gap: 12 }}>
          <ZoomableImagePane title="Local image" subtitle={localPreview?.file.name} src={localPreview?.url} />
          <ZoomableImagePane title="Archive comparison" subtitle={archiveImage?.label} src={archiveImage?.fullres_url ?? archiveImage?.preview_url} />
        </div>
      )}

      <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12, display: 'grid', gap: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
          <div>
            <strong>Local photos</strong>
            <div style={{ color: '#667085', fontSize: 13 }}>{localPreviews.length} loaded</div>
          </div>
          <button onClick={onOpenLookup} style={{ border: '1px solid #ccd6eb', borderRadius: 10, padding: '8px 10px', background: '#eef4ff' }}>Look up star</button>
        </div>
        <LocalImageQueue previews={localPreviews} onRemove={removeLocalAt} onSelect={selectLocalIndex} selectedIndex={selectedLocalIndex} />
      </div>

      {archiveImages.length > 0 && (
        <details style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 12 }}>
          <summary style={{ fontWeight: 600, cursor: 'pointer' }}>Loaded archive comparison images ({archiveImages.length})</summary>
          <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
            {archiveImages.map((image, index) => (
              <button key={image.image_id} onClick={() => selectArchiveIndex(index)} style={{ textAlign: 'left', border: index == selectedArchiveIndex ? '2px solid #2f6fed' : '1px solid #ddd', borderRadius: 10, padding: 8, background: index == selectedArchiveIndex ? '#f5f8ff' : 'white' }}>
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
