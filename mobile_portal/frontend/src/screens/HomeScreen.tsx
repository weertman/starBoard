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
    <div style={{ display: 'grid', gap: 12 }}>
      <div style={{ background: 'white', border: '1px solid #d6dae1', borderRadius: 16, padding: 14, display: 'grid', gap: 10 }}>
        <div style={{ fontSize: 18, fontWeight: 700 }}>How to use this app</div>
        <div style={{ color: '#667085', fontSize: 13 }}>Expand the sections below for instructions. The app is designed so images stay primary and metadata stays secondary until you need it.</div>

        <details open style={{ border: '1px solid #e3e7ee', borderRadius: 12, padding: 10, background: '#fbfcfe' }}>
          <summary style={{ fontWeight: 600, cursor: 'pointer' }}>1. New observation</summary>
          <div style={{ marginTop: 8, color: '#667085', fontSize: 13, lineHeight: 1.4 }}>
            Use this mode when you have new photos on your phone. Capture or select local images first. Then, if needed, compare them against archive images. When ready, open the metadata sheet and submit.
          </div>
        </details>

        <details style={{ border: '1px solid #e3e7ee', borderRadius: 12, padding: 10, background: '#fbfcfe' }}>
          <summary style={{ fontWeight: 600, cursor: 'pointer' }}>2. Look up star</summary>
          <div style={{ marginTop: 8, color: '#667085', fontSize: 13, lineHeight: 1.4 }}>
            Use filters to choose a type, location, ID, and optionally an observation date. This lets you inspect archive images without submitting anything first.
          </div>
        </details>

        <details style={{ border: '1px solid #e3e7ee', borderRadius: 12, padding: 10, background: '#fbfcfe' }}>
          <summary style={{ fontWeight: 600, cursor: 'pointer' }}>3. Compare images</summary>
          <div style={{ marginTop: 8, color: '#667085', fontSize: 13, lineHeight: 1.4 }}>
            After you load local or archive images, tap the images to inspect them more closely. Compare before deciding whether the current animal matches a known star.
          </div>
        </details>

        <details style={{ border: '1px solid #e3e7ee', borderRadius: 12, padding: 10, background: '#fbfcfe' }}>
          <summary style={{ fontWeight: 600, cursor: 'pointer' }}>4. Metadata and submit</summary>
          <div style={{ marginTop: 8, color: '#667085', fontSize: 13, lineHeight: 1.4 }}>
            Metadata lives in the slide-up sheet. Fill in the starBoard fields there, choose query or gallery target behavior, and submit once the observation is ready.
          </div>
        </details>
      </div>

      <button onClick={onNewObservation} style={{ textAlign: 'left', background: 'white', border: '1px solid #d6dae1', borderRadius: 18, padding: 16, display: 'grid', gap: 8 }}>
        <div style={{ fontSize: 11, textTransform: 'uppercase', color: '#667085', letterSpacing: '.08em' }}>Mode 1</div>
        <div style={{ fontSize: 19, fontWeight: 700 }}>New observation</div>
        <div style={{ color: '#667085', fontSize: 13, lineHeight: 1.35 }}>Capture or select local photos, compare if needed, then slide up metadata and submit.</div>
        <div style={{ color: '#2f6fed', fontWeight: 600, fontSize: 13 }}>{localCount} local image{localCount === 1 ? '' : 's'} currently loaded</div>
      </button>

      <button onClick={onLookup} style={{ textAlign: 'left', background: 'white', border: '1px solid #d6dae1', borderRadius: 18, padding: 16, display: 'grid', gap: 8 }}>
        <div style={{ fontSize: 11, textTransform: 'uppercase', color: '#667085', letterSpacing: '.08em' }}>Mode 2</div>
        <div style={{ fontSize: 19, fontWeight: 700 }}>Look up star</div>
        <div style={{ color: '#667085', fontSize: 13, lineHeight: 1.35 }}>Pull up Anchovy or another known individual first. Compare without submitting anything.</div>
        <div style={{ color: '#2f6fed', fontWeight: 600, fontSize: 13 }}>Archive-first workflow</div>
      </button>
    </div>
  )
}
