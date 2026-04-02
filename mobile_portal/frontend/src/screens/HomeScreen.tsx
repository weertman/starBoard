export function HomeScreen({
  localCount,
  onNewObservation,
  onLookup,
}: {
  localCount: number
  onNewObservation: () => void
  onLookup: () => void
}) {
  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 16, display: 'grid', gap: 6 }}>
        <div style={{ fontSize: 22, fontWeight: 700 }}>starBoard</div>
        <div style={{ color: '#667085', fontSize: 14 }}>Choose one clear starting point. Images stay central; metadata stays secondary until you need it.</div>
      </div>

      <button onClick={onNewObservation} style={{ textAlign: 'left', background: 'white', border: '1px solid #d6dae1', borderRadius: 18, padding: 18, display: 'grid', gap: 8 }}>
        <div style={{ fontSize: 12, textTransform: 'uppercase', color: '#667085', letterSpacing: '.08em' }}>Mode 1</div>
        <div style={{ fontSize: 20, fontWeight: 700 }}>New observation</div>
        <div style={{ color: '#667085', fontSize: 14 }}>Capture or select local photos, compare if needed, then slide up metadata and submit.</div>
        <div style={{ color: '#2f6fed', fontWeight: 600, fontSize: 14 }}>{localCount} local image{localCount === 1 ? '' : 's'} currently loaded</div>
      </button>

      <button onClick={onLookup} style={{ textAlign: 'left', background: 'white', border: '1px solid #d6dae1', borderRadius: 18, padding: 18, display: 'grid', gap: 8 }}>
        <div style={{ fontSize: 12, textTransform: 'uppercase', color: '#667085', letterSpacing: '.08em' }}>Mode 2</div>
        <div style={{ fontSize: 20, fontWeight: 700 }}>Look up star</div>
        <div style={{ color: '#667085', fontSize: 14 }}>Pull up Anchovy or another known individual first. Compare without submitting anything.</div>
        <div style={{ color: '#2f6fed', fontWeight: 600, fontSize: 14 }}>Archive-first workflow</div>
      </button>
    </div>
  )
}
