import { LocalImageQueue } from '../components/LocalImageQueue'

export function CaptureScreen({ previews, addFiles, removeAt }: { previews: { file: File; url: string }[]; addFiles: (files: FileList | null) => void; removeAt: (index: number) => void }) {
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <h2>Capture / Select Photos</h2>
      <label>
        Camera capture<br />
        <input type="file" accept="image/*" capture="environment" multiple onChange={(e) => addFiles(e.target.files)} />
      </label>
      <label>
        Photo library<br />
        <input type="file" accept="image/*" multiple onChange={(e) => addFiles(e.target.files)} />
      </label>
      <LocalImageQueue previews={previews} onRemove={removeAt} />
    </div>
  )
}
