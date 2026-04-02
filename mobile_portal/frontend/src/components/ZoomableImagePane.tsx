export function ZoomableImagePane({ src, title, subtitle }: { src?: string; title: string; subtitle?: string }) {
  return (
    <div style={{ border: '1px solid #ddd', borderRadius: 10, padding: 8, minHeight: 260, background: 'white' }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{title}</div>
      {subtitle && <div style={{ fontSize: 12, color: '#555', marginBottom: 8 }}>{subtitle}</div>}
      {src ? (
        <div style={{ overflow: 'auto', maxHeight: 560, borderRadius: 8, touchAction: 'pan-x pan-y pinch-zoom', background: '#f7f7f7' }}>
          <img src={src} alt={title} style={{ width: '100%', height: 'auto', display: 'block' }} />
        </div>
      ) : (
        <div style={{ color: '#666' }}>No image selected.</div>
      )}
    </div>
  )
}
