import type { ImageDescriptor } from '../api/client'
import { ZoomableImagePane } from '../components/ZoomableImagePane'

export function CompareScreen({ localSrc, archiveImage }: { localSrc?: string; archiveImage?: ImageDescriptor | null }) {
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <h2>Compare</h2>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
        <ZoomableImagePane title="Local image" src={localSrc} />
        <ZoomableImagePane title="Archive image" src={archiveImage?.fullres_url ?? archiveImage?.preview_url} />
      </div>
    </div>
  )
}
