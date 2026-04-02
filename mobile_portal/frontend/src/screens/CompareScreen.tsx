import type { ImageDescriptor } from '../api/client'
import { ZoomableImagePane } from '../components/ZoomableImagePane'
import { LocalImageQueue } from '../components/LocalImageQueue'

export function CompareScreen({
  localPreviews,
  selectedLocalIndex,
  selectLocalIndex,
  removeLocalAt,
  archiveImages,
  selectedArchiveIndex,
  selectArchiveIndex,
}: {
  localPreviews: { file: File; url: string }[]
  selectedLocalIndex: number
  selectLocalIndex: (index: number) => void
  removeLocalAt: (index: number) => void
  archiveImages: ImageDescriptor[]
  selectedArchiveIndex: number
  selectArchiveIndex: (index: number) => void
}) {
  const localPreview = localPreviews[selectedLocalIndex]
  const archiveImage = archiveImages[selectedArchiveIndex]
  const hasPrevLocal = selectedLocalIndex > 0
  const hasNextLocal = selectedLocalIndex < localPreviews.length - 1
  const hasPrevArchive = selectedArchiveIndex > 0
  const hasNextArchive = selectedArchiveIndex < archiveImages.length - 1

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <h2>Compare</h2>
      <div style={{ color: '#555', fontSize: 14 }}>Compare the currently selected local phone image against a selected archive image before deciding whether to submit. Use Previous/Next to move quickly through candidates.</div>
      <div style={{ display: 'grid', gap: 12 }}>
        <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid #ddd', borderRadius: 10, background: 'white' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
            <strong>Local image</strong>
            <div style={{ display: 'flex', gap: 8 }}>
              <button onClick={() => hasPrevLocal && selectLocalIndex(selectedLocalIndex - 1)} disabled={!hasPrevLocal}>Previous</button>
              <button onClick={() => hasNextLocal && selectLocalIndex(selectedLocalIndex + 1)} disabled={!hasNextLocal}>Next</button>
            </div>
          </div>
          <ZoomableImagePane title="Local image" subtitle={localPreview?.file.name} src={localPreview?.url} />
        </div>
        <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid #ddd', borderRadius: 10, background: 'white' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center' }}>
            <strong>Archive image</strong>
            <div style={{ display: 'flex', gap: 8 }}>
              <button onClick={() => hasPrevArchive && selectArchiveIndex(selectedArchiveIndex - 1)} disabled={!hasPrevArchive}>Previous</button>
              <button onClick={() => hasNextArchive && selectArchiveIndex(selectedArchiveIndex + 1)} disabled={!hasNextArchive}>Next</button>
            </div>
          </div>
          <ZoomableImagePane title="Archive image" subtitle={archiveImage?.label} src={archiveImage?.fullres_url ?? archiveImage?.preview_url} />
        </div>
      </div>
      <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid #ddd', borderRadius: 10, background: 'white' }}>
        <div><strong>Comparison status</strong></div>
        <div style={{ fontSize: 13, color: '#555' }}>Local image: {localPreview?.file.name ?? 'none selected'} ({localPreviews.length > 0 ? `${selectedLocalIndex + 1}/${localPreviews.length}` : '0/0'})</div>
        <div style={{ fontSize: 13, color: '#555' }}>Archive image: {archiveImage?.label ?? 'none selected'} ({archiveImages.length > 0 ? `${selectedArchiveIndex + 1}/${archiveImages.length}` : '0/0'})</div>
      </div>
      {localPreviews.length > 0 && (
        <details>
          <summary>Choose local comparison image ({localPreviews.length})</summary>
          <div style={{ marginTop: 10 }}>
            <LocalImageQueue previews={localPreviews} onRemove={removeLocalAt} onSelect={selectLocalIndex} selectedIndex={selectedLocalIndex} />
          </div>
        </details>
      )}
      {archiveImages.length > 1 && (
        <details>
          <summary>Choose archive comparison image ({archiveImages.length} loaded)</summary>
          <div style={{ marginTop: 10, display: 'grid', gap: 8 }}>
            {archiveImages.map((image, index) => (
              <button key={image.image_id} onClick={() => selectArchiveIndex(index)} style={{ textAlign: 'left', border: index === selectedArchiveIndex ? '2px solid #2f6fed' : '1px solid #ddd', borderRadius: 10, padding: 8, background: index === selectedArchiveIndex ? '#f5f8ff' : 'white' }}>
                <div style={{ fontSize: 12, marginBottom: 8 }}>{image.label}</div>
                <img src={image.preview_url} alt={image.label} style={{ width: '100%', borderRadius: 8 }} />
              </button>
            ))}
          </div>
        </details>
      )}
      {archiveImages.length === 0 && <div style={{ color: '#666' }}>Open an archive ID in the Lookup tab to choose a comparison image.</div>}
    </div>
  )
}
