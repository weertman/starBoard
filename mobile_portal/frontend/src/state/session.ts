import { useEffect, useState } from 'react'
import { getSession, type SessionResponse } from '../api/client'

export function useSession() {
  const [session, setSession] = useState<SessionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getSession()
      .then(setSession)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false))
  }, [])

  return { session, error, loading }
}
