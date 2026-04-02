import type { ImageDescriptor } from '../api/client'
import { ZoomableImagePane } from '../components/ZoomableImagePane'
import { LocalImageQueue } from '../components/LocalImageQueue'

export function CompareScreen({ localPreviews, selectedLocalIndex, selectLocalIndex, removeLocalAt, archiveImage }: { localPreviews: { file: File; url: string }[]; selectedLocalIndex: number; selectLocalIndex: (index: number) => void; removeLocalAt: (index: number) => void; archiveImage?: ImageDescriptor | null }) {
  const localPreview = localPreviews[selectedLocalIndex]
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <h2>Compare</h2>
      <div style={{ color: '#555', fontSize: 14 }}>Compare the currently selected local phone image against an archive image before deciding whether to submit. Tap or pinch in your phone browser to inspect details.</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
        <ZoomableImagePane title="Local image" subtitle={localPreview?.file.name} src={localPreview?.url} />
        <ZoomableImagePane title="Archive image" subtitle={archiveImage?.label} src={archiveImage?.fullres_url ?? archiveImage?.preview_url} />
      </div>
      <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid #ddd', borderRadius: 10, background: 'white' }}>
        <div><strong>Comparison status</strong></div>
        <div style={{ fontSize: 13, color: '#555' }}>Local image: {localPreview?.file.name ?? 'none selected'}</div>
        <div style={{ fontSize: 13, color: '#555' }}>Archive image: {archiveImage?.label ?? 'none selected'}</div>
      </div>
      {localPreviews.length > 0 && (
        <details>
          <summary>Choose local comparison image ({localPreviews.length})</summary>
          <div style={{ marginTop: 10 }}>
            <LocalImageQueue previews={localPreviews} onRemove={removeLocalAt} onSelect={selectLocalIndex} selectedIndex={selectedLocalIndex} />
          </div>
        </details>
      )}
      {!archiveImage && <div style={{ color: '#666' }}>Open an archive ID in the Lookup tab to choose a comparison image.</div>}
    </div>
  )
}
