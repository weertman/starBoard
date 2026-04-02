export function ZoomableImagePane({ src, title, subtitle, compact = false }: { src?: string; title: string; subtitle?: string; compact?: boolean }) {
  return (
    <div style={{ border: '1px solid #ddd', borderRadius: 10, padding: 8, minHeight: compact ? 180 : 220, background: 'white' }}>
      <div style={{ fontWeight: 600, marginBottom: 4, fontSize: 15 }}>{title}</div>
      {subtitle && <div style={{ fontSize: 12, color: '#555', marginBottom: 8, wordBreak: 'break-word' }}>{subtitle}</div>}
      {src ? (
        <div style={{ overflow: 'auto', maxHeight: compact ? 260 : 360, borderRadius: 8, touchAction: 'pan-x pan-y pinch-zoom', background: '#f7f7f7' }}>
          <img src={src} alt={title} style={{ width: '100%', height: 'auto', display: 'block' }} />
        </div>
      ) : (
        <div style={{ color: '#666', fontSize: 13 }}>No image selected.</div>
      )}
    </div>
  )
}
