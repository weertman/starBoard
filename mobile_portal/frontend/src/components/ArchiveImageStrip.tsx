import type { ImageDescriptor } from '../api/client'

export function ArchiveImageStrip({ items, onSelect, selectedImageId }: { items: ImageDescriptor[]; onSelect: (item: ImageDescriptor) => void; selectedImageId?: string | null }) {
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {items.map((item) => {
        const selected = item.image_id === selectedImageId
        return (
          <button key={item.image_id} onClick={() => onSelect(item)} style={{ textAlign: 'left', border: selected ? '2px solid #2f6fed' : '1px solid #ddd', borderRadius: 10, padding: 8, background: selected ? '#f5f8ff' : 'white' }}>
            <div style={{ fontSize: 12, marginBottom: 8, wordBreak: 'break-word' }}>{item.label}</div>
            <img src={item.preview_url} alt={item.label} style={{ width: '100%', borderRadius: 8 }} />
          </button>
        )
      })}
    </div>
  )
}
