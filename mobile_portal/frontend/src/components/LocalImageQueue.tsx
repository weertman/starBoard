type Preview = { file: File; url: string }

export function LocalImageQueue({ previews, onRemove, onSelect, selectedIndex }: { previews: Preview[]; onRemove: (index: number) => void; onSelect?: (index: number) => void; selectedIndex?: number }) {
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {previews.map((preview, index) => {
        const selected = selectedIndex === index
        return (
          <div key={`${preview.file.name}-${index}`} style={{ border: selected ? '2px solid #2f6fed' : '1px solid #ddd', borderRadius: 10, padding: 8, background: selected ? '#f5f8ff' : 'white' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'center', marginBottom: 8 }}>
              <div style={{ fontSize: 12, wordBreak: 'break-word' }}>{preview.file.name}</div>
              <div style={{ display: 'flex', gap: 8 }}>
                {onSelect && <button onClick={() => onSelect(index)}>Use</button>}
                <button onClick={() => onRemove(index)}>Remove</button>
              </div>
            </div>
            <img src={preview.url} alt={preview.file.name} style={{ width: '100%', borderRadius: 8 }} />
          </div>
        )
      })}
    </div>
  )
}
