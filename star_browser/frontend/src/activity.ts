const SESSION_STORAGE_KEY = 'starboard.activity.session_id'

function makeSessionId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `session-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export function getActivitySessionId(): string {
  if (typeof window === 'undefined') return makeSessionId()
  const existing = window.sessionStorage.getItem(SESSION_STORAGE_KEY)
  if (existing) return existing
  const next = makeSessionId()
  window.sessionStorage.setItem(SESSION_STORAGE_KEY, next)
  return next
}

export function activityHeaders(): Record<string, string> {
  return { 'X-Starboard-Session-Id': getActivitySessionId() }
}

export type ActivityEvent = {
  event_type: string
  client_timestamp_utc?: string
  workflow?: string
  entity_type?: string
  entity_id?: string
  query_id?: string
  gallery_id?: string
  success?: boolean
  duration_ms?: number
  details?: Record<string, unknown>
}

export function trackActivity(event: ActivityEvent): void {
  const payload = {
    session_id: getActivitySessionId(),
    events: [
      {
        client_timestamp_utc: new Date().toISOString(),
        ...event,
      },
    ],
  }

  void fetch('/api/activity/events', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...activityHeaders() },
    body: JSON.stringify(payload),
    keepalive: true,
  }).catch(() => {
    // Activity tracking must never block primary archive workflows.
  })
}
