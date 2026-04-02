import { LocalImageQueue } from '../components/LocalImageQueue'

export function CaptureScreen({ previews, addFiles, removeAt, select, selectedIndex }: { previews: { file: File; url: string }[]; addFiles: (files: FileList | null) => void; removeAt: (index: number) => void; select: (index: number) => void; selectedIndex: number }) {
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <h2>Capture / Select Photos</h2>
      <div style={{ color: '#555', fontSize: 14 }}>Take field photos or select them from your phone. Images stay local in the browser until you submit.</div>
      <label>
        Camera capture<br />
        <input type="file" accept="image/*" capture="environment" multiple onChange={(e) => addFiles(e.target.files)} />
      </label>
      <label>
        Photo library<br />
        <input type="file" accept="image/*" multiple onChange={(e) => addFiles(e.target.files)} />
      </label>
      <div style={{ fontSize: 13, color: '#555' }}>{previews.length} local image{previews.length === 1 ? '' : 's'} selected</div>
      <LocalImageQueue previews={previews} onRemove={removeAt} onSelect={select} selectedIndex={selectedIndex} />
    </div>
  )
}
