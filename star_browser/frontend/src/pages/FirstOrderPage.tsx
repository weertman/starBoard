import { useState } from 'react'

import { runFirstOrderSearch, type FirstOrderSearchResponse } from '../api/client'

const card: React.CSSProperties = {
  background: '#fff',
  border: '1px solid #d7deea',
  borderRadius: 10,
  padding: 12,
  boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
}

const input: React.CSSProperties = {
  width: '100%',
  padding: '8px 10px',
  borderRadius: 8,
  border: '1px solid #c8d0dd',
  boxSizing: 'border-box',
}

export function FirstOrderPage() {
  const [queryId, setQueryId] = useState('')
  const [topK, setTopK] = useState(10)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<FirstOrderSearchResponse | null>(null)

  async function handleSearch() {
    if (!queryId.trim()) return
    setBusy(true)
    setError(null)
    try {
      const next = await runFirstOrderSearch({ query_id: queryId.trim(), top_k: topK })
      setResult(next)
    } catch (err) {
      setError(String(err))
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main style={{ maxWidth: 1180, margin: '0 auto', padding: 18, fontFamily: 'system-ui, sans-serif', color: '#152033', background: '#f7f9fc', minHeight: '100vh' }}>
      <div style={{ display: 'grid', gap: 16 }}>
        <section style={card}>
          <h1 style={{ marginTop: 0 }}>First-order Search</h1>
          <p style={{ marginTop: 0, color: '#516070' }}>Run a minimal first-order search using the current backend ranking defaults.</p>
          <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'minmax(220px, 1fr) 120px auto' }}>
            <input value={queryId} onChange={(e) => setQueryId(e.target.value)} placeholder="Enter query ID" style={input} />
            <input type="number" min={1} max={50} value={topK} onChange={(e) => setTopK(Number(e.target.value) || 10)} style={input} />
            <button onClick={() => void handleSearch()} disabled={busy || !queryId.trim()} style={{ padding: '8px 12px' }}>
              {busy ? 'Searching…' : 'Search'}
            </button>
          </div>
        </section>

        {error && <section style={{ ...card, borderColor: '#e29a9a', background: '#fff5f5', color: '#7a1c1c' }}><b>Error:</b> {error}</section>}

        {result && (
          <section style={card}>
            <h2 style={{ marginTop: 0 }}>Results for {result.query_id}</h2>
            {result.candidates.length === 0 ? (
              <div style={{ color: '#516070' }}>No ranked candidates returned.</div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
                  <thead>
                    <tr style={{ background: '#eef3fb' }}>
                      <th style={{ textAlign: 'left', padding: 8 }}>Rank</th>
                      <th style={{ textAlign: 'left', padding: 8 }}>Gallery ID</th>
                      <th style={{ textAlign: 'left', padding: 8 }}>Score</th>
                      <th style={{ textAlign: 'left', padding: 8 }}>Contributing fields</th>
                      <th style={{ textAlign: 'left', padding: 8 }}>Field breakdown</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.candidates.map((candidate, idx) => (
                      <tr key={`${candidate.entity_id}-${idx}`} style={{ borderTop: '1px solid #e2e8f0' }}>
                        <td style={{ padding: 8 }}>{idx + 1}</td>
                        <td style={{ padding: 8 }}><code>{candidate.entity_id}</code></td>
                        <td style={{ padding: 8 }}>{candidate.score.toFixed(4)}</td>
                        <td style={{ padding: 8 }}>{candidate.k_contrib}</td>
                        <td style={{ padding: 8 }}>
                          {Object.keys(candidate.field_breakdown).length === 0
                            ? '—'
                            : Object.entries(candidate.field_breakdown).map(([k, v]) => `${k}: ${v.toFixed(3)}`).join(', ')}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}
      </div>
    </main>
  )
}
