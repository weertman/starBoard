type Preview = { file: File; url: string }

export function LocalImageQueue({ previews, onRemove }: { previews: Preview[]; onRemove: (index: number) => void }) {
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {previews.map((preview, index) => (
        <div key={`${preview.file.name}-${index}`} style={{ border: '1px solid #ddd', borderRadius: 8, padding: 8 }}>
          <div style={{ fontSize: 12, marginBottom: 8 }}>{preview.file.name}</div>
          <img src={preview.url} alt={preview.file.name} style={{ width: '100%', borderRadius: 8 }} />
          <button onClick={() => onRemove(index)} style={{ marginTop: 8 }}>Remove</button>
        </div>
      ))}
    </div>
  )
}
