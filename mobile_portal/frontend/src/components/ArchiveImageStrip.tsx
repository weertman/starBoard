import type { ImageDescriptor } from '../api/client'

export function ArchiveImageStrip({ items, onSelect }: { items: ImageDescriptor[]; onSelect: (item: ImageDescriptor) => void }) {
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {items.map((item) => (
        <button key={item.image_id} onClick={() => onSelect(item)} style={{ textAlign: 'left', border: '1px solid #ddd', borderRadius: 8, padding: 8, background: 'white' }}>
          <div style={{ fontSize: 12, marginBottom: 8 }}>{item.label}</div>
          <img src={item.preview_url} alt={item.label} style={{ width: '100%', borderRadius: 8 }} />
        </button>
      ))}
    </div>
  )
}
